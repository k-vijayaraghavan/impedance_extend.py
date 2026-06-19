import warnings
import os
import inspect
import random
from warnings import warn

import numpy as np
from scipy.linalg import inv, svd
from scipy.optimize import curve_fit, basinhopping

from .elements import circuit_elements, get_element_from_name


def _custom_formatwarning(msg, category, filename, lineno, line=None):
    norm_path = os.path.normpath(filename)
    path_parts = norm_path.split(os.sep)
    if 'impedance' in path_parts:
        display_name = ".".join(path_parts[path_parts.index('impedance'):])
        if display_name.endswith('.py'):
            display_name = display_name[:-3]
    else:
        display_name = os.path.basename(filename)
    return f"{display_name}:{lineno}: {category.__name__}: {msg}\n"


warnings.formatwarning = _custom_formatwarning

ints = '0123456789'


def rmse(a, b):
    """
    A function which calculates the root mean squared error
    between two vectors.

    Notes
    ---------
    .. math::

        RMSE = \\sqrt{\\frac{1}{n}(a-b)^2}
    """

    n = len(a)
    return np.linalg.norm(a - b) / np.sqrt(n)


def set_default_bounds(circuit, constants={}):
    """ This function sets default bounds for optimization.

    set_default_bounds sets bounds of 0 and np.inf for all parameters,
    except the CPE and La alphas which have an upper bound of 1.

    Parameters
    -----------------
    circuit : string
        String defining the equivalent circuit to be fit

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    Returns
    ------------
    bounds : 2-tuple of array_like
        Lower and upper bounds on parameters.
    """

    # extract the elements from the circuit
    extracted_elements = extract_circuit_elements(circuit)

    # loop through bounds
    lower_bounds, upper_bounds = [], []
    for elem in extracted_elements:
        raw_element = get_element_from_name(elem)
        for i in range(check_and_eval(raw_element).num_params):
            if elem in constants or elem + f'_{i}' in constants:
                continue
            if (raw_element in ['CPE', 'La'] and i == 1) or \
               (raw_element in ['TLMQ'] and i == 2):
                upper_bounds.append(1)
            else:
                upper_bounds.append(np.inf)
            lower_bounds.append(0)

    bounds = ((lower_bounds), (upper_bounds))
    return bounds


def scale_bounds(bounds, guess, scale, ubound_finite=False):
    # We should either accept [(0, 0.1), (0, 100), (0, 0.01), (0, 10), (0, 1)]
    # or [(0, 0, ...), [0.1, 100, ...]] or [0, [0.1, 100, ...]];
    # This should also work when n_guess=2
    n_guess = len(guess)
    is_format_1 = False

    if len(bounds) == n_guess and len(bounds) != 2:
        is_format_1 = True
    elif len(bounds) == 2:
        if n_guess == 2:
            if isinstance(bounds, list) and isinstance(bounds[0], tuple):
                is_format_1 = True
            else:
                is_format_1 = False
        else:
            is_format_1 = False
    else:
        is_format_1 = True

    if is_format_1:
        b0 = np.array([b[0] for b in bounds])
        b1 = np.array([b[1] for b in bounds])
    else:
        b0 = np.atleast_1d(bounds[0])
        b1 = np.atleast_1d(bounds[1])
        if len(b0) == 1:
            b0 = np.repeat(b0[0], n_guess)
        if len(b1) == 1:
            b1 = np.repeat(b1[0], n_guess)
        if ubound_finite:
            flag = False
            ubf = 10
            b1_ = ubf*pow_of_10(guess, 1)
            for ib, b in enumerate(b1):
                if np.isposinf(b):
                    flag = True
                    b1[ib] = b1_[ib]
            if flag:
                warn("Forcing finite upper bound")

    if scale is None:
        scale = np.ones(n_guess)
    else:
        scale = np.array(scale, dtype=float)

    scaled_low = np.array(b0, dtype=float) / scale
    scaled_high = np.array(b1, dtype=float) / scale
    return scaled_low, scaled_high


def is_scalarval(var, val):
    if isinstance(var, (list, np.ndarray)):
        return False
    if var != val:
        return False
    return True


def calc_perror(res, df, scale=1, name=""):
    try:
        pcov = res.hess_inv
        if hasattr(pcov, "todense"):
            pcov = pcov.todense()
        pcov = pcov * np.outer(scale, scale)
        return np.sqrt(np.diag(pcov))
    except:
        try:
            res_jac = res['jac'] if isinstance(res,dict) else res.jac 
            res_cost = res['cost'] if isinstance(res,dict) else res.cost 
        except:
            warnings.warn('Failed to compute perror as this version of '
                        'algorithm returns neither "hess_inv" nor "jac"')
            return None
    if res_jac is not None:
        _, s, VT = svd(res_jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res_jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        cost = 2 * res_cost
        # df = len(target_Z) - len(popt)
        if df > 0:
            pcov = pcov * (cost / df)
        pcov = pcov * np.outer(scale, scale)
    return np.sqrt(np.diag(pcov))


def pow_of_10(num, dirn=0):
    # ret = 10 ** np.round(np.log10(np.abs(num) +
    #                                 np.finfo(float).eps))
    # Set dirn = 1 for getting next power of 10, 0 for closest
    if dirn == 1:
        ret = 10 ** np.ceil(np.log10(np.abs(num) +
                            np.finfo(float).eps))
    else:
        ret = 10 ** np.floor(np.log10(np.abs(num) +
                             np.finfo(float).eps))
        if dirn == 0:
            idx = num >= 5*ret
            ret[idx] *= 10
    return ret


def circuit_fit(frequencies, impedances, circuit, initial_guess,
                constants={}, bounds=None, weight_by_modulus=False,
                global_opt=False, optimizations=[], scale=None,
                **kwargs):

    """ Main function for fitting an equivalent circuit to data.

    By default, this function uses `scipy.optimize.curve_fit
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
    to fit the equivalent circuit. This function generally works well for
    simple circuits. However, the final results may be sensitive to
    the initial conditions for more complex circuits. In these cases,
    the `scipy.optimize.basinhopping
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_
    global optimization algorithm can be used to attempt a better fit.

    Parameters
    -----------------
    frequencies : numpy array
        Frequencies

    impedances : numpy array of dtype 'complex128'
        Impedances

    circuit : string
        String defining the equivalent circuit to be fit

    initial_guess : list of floats
        Initial guesses for the fit parameters

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to bounds on all
        parameters of 0 and np.inf, except the CPE alpha
        which has an upper bound of 1

    weight_by_modulus : bool, optional
        Uses the modulus of each data (|Z|) as the weighting factor.
        Standard weighting scheme when experimental variances are unavailable.
        Only applicable when global_opt = False

    global_opt : bool, optional
        If global optimization should be used (uses the basinhopping
        algorithm). Defaults to False

    optimizations : dict or list of dicts, optional
        If global_opt is True, gets set to "basinhopping"
        Else
            If optimizations is not passed, curve_fit is used.
            If dict(s), it must algorithm + algorithm options
            "algorithm" is mandatory.
            Acceptable algorithm are 'curve_fit', 'basinhopping',
            'least_squares', 'pygad', 'pyswarms' or a callable func*
                eg : {"algorithm" : 'pygad', 'gene_space' : ...} or
                {"algorithm" : 'pyswarms', 'options' : ...}
                or {"algorithm" : func, 'cost_asvector' : True, ...}
                Callable func supports 'vector_residuals':
                    setting vector_residuals to True passes a vector of
                    residuals to the cost funtion that is useful for
                    scipy.optimize::least_squares

            List could be a list of dics (in the above format). This is used
            for sequential optimizations, particularly useful for GA.
            For 'least_squares', we use jacobian (unless we explicitly set 
            use_jac=False). For callable func, it is possible to get jacobian 
            by setting use_jac=True

    scale : list, optional
        Used to denote "scale" of prameters to improve convergence.
        Consider a p(R,C) or R-C circuit. Suppose C-s is in μF while,
        R-s might be in ~0.1 ohms; we can pass [0.1,1e-6].
        Internally the parameters are divided by scale during optimization.

    kwargs :
        Keyword arguments passed to scipy.optimize.curve_fit or
        scipy.optimize.basinhopping
        One can also pass a callable soft_constraint that adds a penalty
        for arbitary constraints.

    Returns
    ------------
    p_values : list of floats
        best fit parameters for specified equivalent circuit

    p_errors : list of floats
        one standard deviation error estimates for fit parameters

   result_objects : For pygad, pyswarms and callable func) \
                    object or list of objects
        result objec or list of result objects

    Notes
    ---------
    Need to do a better job of handling errors in fitting.
    Currently, an error of -1 is returned.

    """
    kwargs_org = kwargs.copy()
    f = np.array(frequencies, dtype=float)
    Z = np.array(impedances, dtype=complex)

    if global_opt:
        warn('global_opt has been deprecated. Use optimizations='
             '{"algorithm": "basinhopping"} OR '
             'optimizations="basinhopping"',
             DeprecationWarning, stacklevel=2)
        opt = {"algorithm": 'basinhopping'}
        optimizations = []
    elif optimizations == [] or optimizations == {}:
        opt = {"algorithm": 'curve_fit'}
    elif isinstance(optimizations, list):
        opt = optimizations.pop(0)
    else:
        opt = optimizations
        optimizations = []

    if not isinstance(opt, dict):
        opt = {"algorithm": opt}
    kwargs.update(opt)
    algo = kwargs.pop("algorithm")

    # set upper and lower bounds on a per-element basis
    bounds_ = bounds
    bounds = kwargs.pop("bounds", bounds)
    if bounds is None:
        bounds = set_default_bounds(circuit, constants=constants)

    seed_val = kwargs.pop('seed', None)
    random_seed_val = kwargs.pop('random_seed', None)
    seed = seed_val if seed_val is not None else random_seed_val
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        if algo == 'pygad':
            kwargs['random_seed'] = seed
        elif algo == 'basinhopping':
            kwargs['seed'] = seed

    use_jac = kwargs.pop("use_jac", not callable(algo)) \
             if callable(algo) or algo in ('least_squares') \
                else False
        
    if callable(algo) or algo in ('pygad', 'pyswarms', 'least_squares') :
        n_soft_constraint = 0
        if 'soft_constraint' in kwargs:
            soft_constraint = kwargs.pop('soft_constraint')
            n_soft_constraint = 1
        if use_jac:
                if n_soft_constraint > 0 :
                    if 'soft_constraint_jac' in kwargs:
                        soft_constraint_jac = kwargs.pop('soft_constraint_jac')
                    else:
                        warnings.warn("soft_constraint_jac is missing.")
                        use_jac = False

        if weight_by_modulus:
            abs_Z = np.abs(Z)
            kwargs['sigma'] = np.hstack([abs_Z, abs_Z, [1]*n_soft_constraint])
            if 'sigma' in kwargs:
                warn("weight_by_modulus==True over-rode sigma values")
        else:
            sigma = kwargs.pop('sigma', 1)
        if isinstance(sigma, (list, np.ndarray)):
            sigma = np.hstack([sigma, [1]*n_soft_constraint])

        vector_residuals = kwargs.pop('vector_residuals', False) \
            if callable(algo) else algo == 'least_squares'

        if scale is None:
            if not callable(algo):
                scale = pow_of_10(initial_guess, 0)
                warnings.warn(f"'scale' is recommeded for {str(algo)}. "
                              "Using scale from initial_guess.")
            else:
                scale = 1
        scaled_low, scaled_high = scale_bounds(bounds, initial_guess,
                                               scale, ubound_finite=algo
                                               in ('pygad', 'pyswarms'))
        scaled_initial = np.array(initial_guess) / scale

        def obj_fn(p_scaled):
            p_unscaled = p_scaled * scale
            try:
                pred_Z = wrapedCircuit(f, *p_unscaled)
                if n_soft_constraint > 0:
                    pred_Z[-1] = soft_constraint(p_unscaled) * len(f) #Check
                error = (pred_Z - target_Z) / sigma
            except Exception:
                return np.inf
            return error if vector_residuals else 0.5 * np.dot(error, error)
        pbar = None
        show_progress = kwargs.pop('show_progress',
                                   kwargs.get('show_progress', True))
        if show_progress:
            if algo == 'pygad':
                maxiter = kwargs.get('num_generations', 1000)
            elif algo == 'pyswarms':
                maxiter = kwargs.get('iters', 1000)
            else:
                maxiter = kwargs.get('options', {}).get('maxiter', None)
            try:
                from tqdm.auto import tqdm
                method = f" ({kwargs.get('method', 'default')} method)"
                pbar = tqdm(total=maxiter, desc="Circuit fit using " +
                            (algo.__name__ if callable(algo)
                             else str(algo)) + method, leave=False)
            except ImportError:
                warn('tqdm not found, progress cannot be plotted !!!')
        ret_obj = True
    else:
        ret_obj = False
        n_soft_constraint = 0

    wrapedCircuit, wrapedJac = wrapCircuit(circuit, constants, n_soft_constraint, retun_jac=True if use_jac else 2)
    target_Z = np.hstack([Z.real, Z.imag, [0]*n_soft_constraint])
    if use_jac:
        def jac(p_scaled):
            # We jeed jac to be (an m-by-n matrix, where jac(i, j) is the 
            # partial derivative of f[i] with respect to x[j]).
            p_unscaled = p_scaled * scale
            try:
                pred_jac = wrapedJac(f, *p_unscaled)
                if n_soft_constraint > 0:
                     pred_jac[-1] = soft_constraint_jac(p_unscaled) * len(f)
            except Exception:
                return np.inf
            return pred_jac
    else:
        jac = '2-point'

    if algo == 'curve_fit':
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = 1e5
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13

        # weighting scheme for fitting
        if weight_by_modulus:
            abs_Z = np.abs(Z)
            kwargs['sigma'] = np.hstack([abs_Z, abs_Z])

        popt, pcov = curve_fit(wrapedCircuit, f,
                               target_Z,
                               p0=initial_guess, bounds=bounds, **kwargs)

        # Calculate one standard deviation error estimates for fit parameters,
        # defined as the square root of the diagonal of the covariance matrix.
        # https://stackoverflow.com/a/52275674/5144795
        perror = np.sqrt(np.diag(pcov))

    elif algo == 'basinhopping':
        if 'seed' not in kwargs:
            kwargs['seed'] = 0

        def opt_function(x):
            """ Short function for basinhopping to optimize over.
            We want to minimize the RMSE between the fit and the data.

            Parameters
            ----------
            x : args
                Parameters for optimization.

            Returns
            -------
            function
                Returns a function (RMSE as a function of parameters).
            """
            return rmse(wrapedCircuit(f, *x),
                        np.hstack([Z.real, Z.imag]))

        class BasinhoppingBounds(object):
            """ Adapted from the basinhopping documetation
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
            """

            def __init__(self, xmin, xmax):
                self.xmin = np.array(xmin)
                self.xmax = np.array(xmax)

            def __call__(self, **kwargs):
                x = kwargs['x_new']
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin

        basinhopping_bounds = BasinhoppingBounds(xmin=bounds[0],
                                                 xmax=bounds[1])
        results = basinhopping(opt_function, x0=initial_guess,
                               accept_test=basinhopping_bounds, **kwargs)
        popt = results.x

        # Calculate perror
        jac = results.lowest_optimization_result['jac'][np.newaxis]
        try:
            # jacobian -> covariance
            # https://stats.stackexchange.com/q/231868
            pcov = inv(np.dot(jac.T, jac)) * opt_function(popt) ** 2
            # covariance -> perror (one standard deviation
            # error estimates for fit parameters)
            perror = np.sqrt(np.diag(pcov))
        except (ValueError, np.linalg.LinAlgError):
            warnings.warn('Failed to compute perror')
            perror = None

    elif algo == 'least_squares':
        method = kwargs.pop('method', 'trf')

        from scipy.optimize import least_squares
        user_callback = kwargs.pop('callback', None)

        def combined_callback(xk, *args, **kwds):
            if pbar is not None:
                pbar.update(1)
            if user_callback is not None:
                user_callback(xk, *args, **kwds)

        min_bounds_scaled = (scaled_low, scaled_high)
        if 'callback' in inspect.signature(least_squares).parameters.keys():
            kwargs['callback'] = combined_callback
        else:
            warn('This version of least_squares does not support "callback"')
        res = least_squares(obj_fn, scaled_initial, method=method, jac=jac,
                            bounds=min_bounds_scaled, **kwargs)
        if pbar is not None:
            pbar.close()
        popt = res.x * scale
        perror = calc_perror(res, len(target_Z) - len(popt), scale,
                             name=algo)

    elif algo == 'pygad':
        import pygad

        def fitness_func(ga_instance, solution, solution_idx):
            return 1. / obj_fn(solution)

        gene_space = kwargs.pop('gene_space', None)
        if gene_space is None and scaled_low is not None:
            gene_space = [{'low': low, 'high': high}
                          for low, high in zip(scaled_low, scaled_high)]

        num_generations = kwargs.pop('num_generations', 1000)
        num_parents_mating = kwargs.pop('num_parents_mating', 4)
        sol_per_pop = kwargs.pop('sol_per_pop', 20)
        num_genes = len(initial_guess)

        initial_population = kwargs.pop('initial_population', None)
        if initial_population is None:
            initial_population = np.empty((sol_per_pop, num_genes))
            initial_population[0] = scaled_initial
            for i in range(1, sol_per_pop):
                initial_population[i] = np.random.uniform(scaled_low,
                                                          scaled_high)
            initial_population = np.clip(initial_population, scaled_low,
                                         scaled_high)
        else:
            initial_population = np.array(initial_population) / scale

        plot_pygad = kwargs.pop('plot', False)
        user_on_generation = kwargs.pop('on_generation', None)

        def on_generation(ga_instance):
            if pbar is not None:
                pbar.update(1)
            if user_on_generation:
                return user_on_generation(ga_instance)

        parent_selection_type = kwargs.pop('parent_selection_type', 'sss')
        keep_elitism = kwargs.pop('keep_elitism', 1)
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func,
                               initial_population=initial_population,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               gene_space=gene_space,
                               on_generation=on_generation,
                               parent_selection_type=parent_selection_type,
                               keep_elitism=keep_elitism,
                               **kwargs)
        ga_instance.run()
        if pbar is not None:
            pbar.close()

        if plot_pygad:
            ga_instance.plot_fitness()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        res = ga_instance
        popt = solution * scale
        perror = None

    elif algo == 'pyswarms':
        import pyswarms as ps

        def fitness_func(x):
            n_particles = x.shape[0]
            j = [obj_fn(x[i]) for i in range(n_particles)]
            if pbar is not None:
                pbar.update(1)
            return np.array(j)

        if scaled_low is None:
            raise ValueError("Bounds must be provided for pyswarms "
                             "optimization.")

        if np.any(np.isinf(scaled_low)) or np.any(np.isinf(scaled_high)):
            raise ValueError("Bounds must be finite for pyswarms "
                             "optimization.")

        bounds_ps = (np.array(scaled_low), np.array(scaled_high))

        options = kwargs.pop('options', {'c1': 0.5, 'c2': 0.3, 'w': 0.9})
        n_particles = kwargs.pop('n_particles', 20)
        iters = kwargs.pop('iters', 1000)
        num_dimensions = len(initial_guess)

        initial_population = kwargs.pop('initial_population', None)
        if initial_population is None:
            initial_population = np.empty((n_particles, num_dimensions))
            initial_population[0] = scaled_initial

            for i in range(1, n_particles):
                initial_population[i] = np.random.uniform(scaled_low,
                                                          scaled_high)

            initial_population = np.clip(initial_population, scaled_low,
                                         scaled_high)
        else:
            initial_population = np.array(initial_population) / scale

        verbose = kwargs.pop('verbose', pbar is None)
        plot_pyswarms = kwargs.pop('plot', False)
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                            dimensions=num_dimensions,
                                            options=options,
                                            bounds=bounds_ps,
                                            init_pos=initial_population,
                                            **kwargs)
        cost, pos = optimizer.optimize(fitness_func, iters=iters,
                                       verbose=verbose)
        res = optimizer

        if pbar is not None:
            pbar.close()

        if plot_pyswarms:
            from pyswarms.utils.plotters import plot_cost_history
            import matplotlib.pyplot as plt
            plot_cost_history(cost_history=optimizer.cost_history)
            plt.show()

        popt = pos * scale
        perror = None

    elif callable(algo):
        sig = inspect.signature(algo)
        valid_params = sig.parameters

        call_kwargs = {}
        # Map parameters dynamically based on the callable's signature
        if 'initial_guess' in valid_params:
            call_kwargs['initial_guess'] = scaled_initial
        elif 'x0' in valid_params:  # Standard scipy optimize param
            call_kwargs['x0'] = scaled_initial
        else:
            warn('Initial guess was ignored')
        if 'bounds' in valid_params:
            call_kwargs['bounds'] = bounds
        else:
            warn('bounds was ignored')
        if 'fun' in valid_params:
            call_kwargs['fun'] = obj_fn
        else:
            warn('objective function was ignored (as callable does not '
                 'accept "fun" parameter)')

        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in valid_params.values())
        ignored = []
        for k, v in kwargs.items():
            if k in valid_params or accepts_kwargs:
                call_kwargs[k] = v
            else:
                ignored.append(k)
        if use_jac:
            call_kwargs["jac"] = jac
        if ignored != []:
            warn('Ignored ' + ','.join(ignored))

        res = algo(**call_kwargs)
        popt = (res.x if hasattr(res, 'x') else res) * scale
        perror = calc_perror(res, len(target_Z) - len(popt), scale,
                             name=algo.__name__ if callable(algo)
                             else str(algo))
    else:
        raise ValueError(f"Unknown optimization algorithm: {opt['algorithm']}")
    if len(optimizations) > 0:
        ret = circuit_fit(frequencies, impedances, circuit,
                          initial_guess=popt, constants=constants,
                          bounds=bounds_, weight_by_modulus=weight_by_modulus,
                          global_opt=False, optimizations=optimizations,
                          scale=scale, use_jac=use_jac, **kwargs_org)
        if ret_obj:
            if len(ret) < 3:
                ret.append((res, scale))
            else:
                ret[2] = [(res, scale)] + ret[2] if isinstance(ret[2], list) \
                    else [(res, scale), ret[2]]
        return ret
    else:
        return [popt, perror, (res, scale)] if ret_obj else [popt, perror]


def wrapCircuit(circuit, constants, addn=0, retun_jac=False):
    """ wraps function so we can pass the circuit string """
    buildCircuit_text = buildCircuit(circuit, constants=constants,
                                     eval_string='', index=0)[0]
    builtCircuit = eval('lambda frequencies,parameters : ' +
                        buildCircuit_text, circuit_elements)

    def wrappedCircuit(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        circuit : string
        constants : dict
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """

        x = builtCircuit(frequencies, parameters)
        y_real = np.real(x)
        y_imag = np.imag(x)

        return np.hstack([y_real, y_imag, [0]*addn])

    if not retun_jac:
        return wrappedCircuit
    elif retun_jac == 2:
        return wrappedCircuit, None
   
    buildJac_text = buildCircuit(circuit, constants=constants, jac=True,
                                     eval_string='', index=0)[0]
    builtJac = eval('lambda frequencies,parameters,dzdp : ' +
                        buildJac_text, circuit_elements)
    def wrappedJac(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance jac
        components

        Parameters
        ----------
        circuit : string
        constants : dict
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """
        dzdp = np.zeros((len(frequencies), len(parameters)), dtype=complex)
        dconstdp = np.zeros((addn, len(parameters)))
        x = builtJac(frequencies, parameters, dzdp)
        dzdp_real = np.real(dzdp)
        dzdp_imag = np.imag(dzdp)

        return np.vstack([dzdp_real, dzdp_imag, dconstdp])

    return wrappedCircuit, wrappedJac


def buildCircuit(circuit, constants=None, jac=False, eval_string='', index=0):
    """ recursive function that transforms a circuit, parameters, and
    frequencies into a string that can be evaluated

    Parameters
    ----------
    circuit: str
    frequencies: list/tuple/array of floats
    parameters: list/tuple/array of floats
    constants: dict
    jac: boolean, set true to return jacobian

    Returns
    -------
    eval_string: str
        Python expression for calculating the resulting fit
        This would a string that can be used to construct a lambda function.
        For example if circuit=R1,CPE1 with CPE1_1 = const1, eval_string =
        "p(R(frequencies,[parameters[0]), CPE(frequencies,[p[1],const1]))";
        We can then construct lambda frequencies,parameters : <eval_string>
        When `jac=True`, the eval_string also evaluates jacobian. The 
        eval_string = "p(R(frequencies,[parameters[0],dzdp[:,0]), 
        CPE(frequencies,[p[1],const1],dzdp[:,1]))";
        We can then create 
        dzdp = np.zeros((len(freqs), len(parameters)), dtype=complex)
        construct lambda frequencies,parameters,dzdp : <eval_string>, 
        and return dzdp in a wrapper.
    index: int
        Tracks parameter index through recursive calling of the function
    """

    circuit = circuit.replace(' ', '')

    def parse_circuit(circuit, parallel=False, series=False):
        """ Splits a circuit string by either dashes (series) or commas
            (parallel) outside of any paranthesis. Removes any leading 'p('
            or trailing ')' when in parallel mode """

        assert parallel != series, \
            'Exactly one of parallel or series must be True'

        def count_parens(string):
            return string.count('('), string.count(')')

        if parallel:
            special = ','
            if circuit.endswith(')') and circuit.startswith('p('):
                circuit = circuit[2:-1]
        if series:
            special = '-'

        split = circuit.split(special)

        result = []
        skipped = []
        for i, sub_str in enumerate(split):
            if i not in skipped:
                if '(' not in sub_str and ')' not in sub_str:
                    result.append(sub_str)
                else:
                    open_parens, closed_parens = count_parens(sub_str)
                    if open_parens == closed_parens:
                        result.append(sub_str)
                    else:
                        uneven = True
                        while i < len(split) - 1 and uneven:
                            sub_str += special + split[i+1]

                            open_parens, closed_parens = count_parens(sub_str)
                            uneven = open_parens != closed_parens

                            i += 1
                            skipped.append(i)
                        result.append(sub_str)
        return result

    parallel = parse_circuit(circuit, parallel=True)
    series = parse_circuit(circuit, series=True)

    if series is not None and len(series) > 1:
        eval_string += "s(["
        split = series
    elif parallel is not None and len(parallel) > 1:
        eval_string += "p(["
        split = parallel
    elif series == parallel:  # only single element
        split = series

    for i, elem in enumerate(split):
        if ',' in elem or '-' in elem:
            eval_string, index = buildCircuit(elem, constants=constants, 
                                              jac=jac, eval_string=\
                                              eval_string, index=index)
        else:
            # Return a string that can be used to construct a lambda function
            # lambda f,p : R(f,[p[0],const1,p[1]...])
            param_string = ""
            jac_string = ""
            raw_elem = get_element_from_name(elem)
            elem_number = check_and_eval(raw_elem).num_params
            param_list = []
            jac_list = []
            for j in range(elem_number):
                if elem_number > 1:
                    current_elem = elem + '_{}'.format(j)
                else:
                    current_elem = elem

                if current_elem in constants.keys():
                    param_list.append(str(constants[current_elem]))
                    jac_list.append(f'None')
                else:
                    param_list.append(f'parameters[{index}]')
                    jac_list.append(f'dzdp[:,{index}]') #dzdp[:, 0]
                    index += 1

            param_string = "[" + ','.join(param_list) + "]"
            jac_string = "[" + ','.join(jac_list) + "]"
            new = raw_elem + '(' + param_string + ', frequencies' + ( ')' if not jac else f', {jac_string})' )
            eval_string += new

        if i == len(split) - 1:
            if len(split) > 1:  # do not add closing brackets if single element
                eval_string += '])'
        else:
            eval_string += ','

    return eval_string, index


def extract_circuit_elements(circuit):
    """ Extracts circuit elements from a circuit string.

    Parameters
    ----------
    circuit : str
        Circuit string.

    Returns
    -------
    extracted_elements : list
        list of extracted elements.

    """
    p_string = [x for x in circuit if x not in 'p(),-']
    extracted_elements = []
    current_element = []
    length = len(p_string)
    for i, char in enumerate(p_string):
        if char not in ints:
            current_element.append(char)
        else:
            # min to prevent looking ahead past end of list
            if p_string[min(i+1, length-1)] not in ints:
                current_element.append(char)
                extracted_elements.append(''.join(current_element))
                current_element = []
            else:
                current_element.append(char)
    extracted_elements.append(''.join(current_element))
    return extracted_elements


def calculateCircuitLength(circuit):
    """ Calculates the number of elements in the circuit.

    Parameters
    ----------
    circuit : str
        Circuit string.

    Returns
    -------
    length : int
        Length of circuit.

    """
    length = 0
    if circuit:
        extracted_elements = extract_circuit_elements(circuit)
        for elem in extracted_elements:
            raw_element = get_element_from_name(elem)
            num_params = check_and_eval(raw_element).num_params
            length += num_params
    return length


def check_and_eval(element):
    """ Checks if an element is valid, then evaluates it.

    Parameters
    ----------
    element : str
        Circuit element.

    Raises
    ------
    ValueError
        Raised if an element is not in the list of allowed elements.

    Returns
    -------
    Evaluated element.

    """
    allowed_elements = circuit_elements.keys()
    if element not in allowed_elements:
        raise ValueError(f'{element} not in ' +
                         f'allowed elements ({allowed_elements})')
    else:
        return eval(element, circuit_elements)
