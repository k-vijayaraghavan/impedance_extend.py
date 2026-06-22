import numpy as np


class ElementError(Exception):
    ...


class OverwriteError(ElementError):
    ...


def element(num_params, units, overwrite=False):
    """decorator to store metadata for a circuit element

    Parameters
    ----------
    num_params : int
        number of parameters for an element
    units : list of str
        list of units for the element parameters
    overwrite : bool (default False)
        if true, overwrites any existing element; if false,
        raises OverwriteError if element name already exists.

    All funtions accept parameter, frequency and a list of (reference to)
    np.array of floats-vectors that stores gradient vector.
    Each element would calculate impedance and jacobian of the impedance at
    different frequencies and return a 2-tuple consisting of impedance and
    list of jacobian. If gradient vector is not provided, we do not calculate
    the gradient; we return only the impedance.
    In order to calculate the gradient (to be used for Jacobian)
        Create a real-matrix param of size len(parameters)
        Create a complex-matrix Js of size len(frequencies) × len(parameters)
        If parameters k, k+1,...+(k+np-1) correspond to this element
            Pass p=[param[k],param[k+1],...] or param[k:(k+np)], and
            Js=[Js[:,k],Js[:,k+1],...] (Note: you cannot pass numpy array)
            If the j-th parameter is constant, pass Js[j]=None
    """

    def decorator(func):
        def wrapper(p, f, J=None):
            typeChecker(p, f, func.__name__, num_params)
            if J is not None:
                jacChecker(J, f, func.__name__, num_params)
            return func(p, f, J)

        wrapper.num_params = num_params
        wrapper.units = units
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        if func.__name__ in ["s", "p"]:
            raise ElementError("cannot redefine elements 's' (series)" +
                               "or 'p' (parallel)")
        elif func.__name__ in circuit_elements and not overwrite:
            raise OverwriteError(
                f"element {func.__name__} already exists. " +
                "If you want to overwrite the existing element," +
                "use `overwrite=True`."
            )
        else:
            circuit_elements[func.__name__] = wrapper

        return wrapper

    return decorator


def s(Zs_and_Js):
    """sums elements in series

    Notes
    ---------
    .. math::
        Z = Z_1 + Z_2 + ... + Z_n

    The Jacobian would add up, however the parameters list also expands. Hence,
    .. math::
        \\frac{dZ}{d[p_1, p_2, ...]
        = [(\\frac{dZ}{dp})_1, (\\frac{dZ}{dp})_2, ...]

    """
    if isinstance(Zs_and_Js[0], (tuple)):  # Passed Zs and Js
        Zs = []
        Js = []
        for s_, dz_ in Zs_and_Js:
            Zs.append(s_)
            Js.append(dz_)
    else:
        Zs = Zs_and_Js
        Js = None

    Zout = len(Zs[0]) * [0 + 0 * 1j]
    for Z in Zs:
        Zout += Z
    if Js is None:
        return Zout
    Jout = []
    for dz in Js:
        Jout.extend(dz)
    return (Zout, Jout)


def p(Zs_and_Js):
    """adds elements in parallel

    Notes
    ---------
    .. math::
        Z = \\frac{1}{\\frac{1}{Z_1} + \\frac{1}{Z_2} + ... + \\frac{1}{Z_n}}

    If each element of Zs_and_Js is a tuple, we need to calcualte the Jacobian.
    Consider, ΔZ, the total change in out resistance is given as
    .. math::
        ΔZ = \\frac{Z^2}{Z_1^2}ΔZ_1 + \\frac{Z^2}{Z_2^2}ΔZ_2 + ...

    Now, if $p_{k-1}$, $p_{k-2}$, ... are the parameters of $Z_k$,
    .. math::
        ΔZ_k = \\frac{∂Z_k(f_i)}{∂p_{k-j}} × [Δp_{k-1}, Δp_{k-1}, ...]^T

    Hence the effective output Jacobian becomes
    .. math::
        J = [\\frac{Z^2}{Z_1^2}×\\frac{∂Z_1(f_i)}{∂p_{1-j}},...]
    """

    if isinstance(Zs_and_Js[0], (tuple)):  # Passed Zs and Js
        Zs = []
        Js = []
        for p_, dz_ in Zs_and_Js:
            Zs.append(p_)
            Js.append(dz_)
    else:
        Zs = Zs_and_Js
        Js = None

    z = len(Zs[0]) * [0 + 0 * 1j]
    for elem in Zs:
        z += 1 / elem
    Zout = 1 / z
    if Js is None:
        return Zout

    Jout = []
    for elem, Js in Zs_and_Js:
        for J in Js:
            J *= Zout**2/elem**2
            Jout += [J]
    return (Zout, Jout)


# manually add parallel and series operators to circuit elements w/o metadata
# populated by the element decorator -
# this maps ex. 'R' to the function R to always give us a list of
# active elements in any context
circuit_elements = {"s": s, "p": p}


@element(num_params=1, units=["Ohm"])
def R(p, f, Js):
    """defines a resistor

    Notes
    ---------
    .. math::

        Z = R

    """
    R = p[0]
    Z = np.array(len(f) * [R], dtype=complex)
    if Js is None:
        return Z
    if Js[0] is not None:
        Js[0][:] = np.ones(len(f), dtype=complex)
    return (Z, Js)


@element(num_params=1, units=["F"])
def C(p, f, Js):
    """defines a capacitor

    .. math::

        Z = \\frac{1}{C \\times j 2 \\pi f}

    """
    omega = 2 * np.pi * np.array(f)
    C = p[0]
    Z = 1.0 / (C * 1j * omega)
    if Js is None:
        return Z
    if Js[0] is not None:
        Js[0][:] = -1.0 / (C**2 * 1j * omega)
    return (Z, Js)


@element(num_params=1, units=["H"])
def L(p, f, Js):
    """defines an inductor

    .. math::

        Z = L \\times j 2 \\pi f

    """
    omega = 2 * np.pi * np.array(f)
    L = p[0]
    Z = L * 1j * omega
    if Js is None:
        return Z
    if Js[0] is not None:
        Js[0][:] = 1j * omega
    return (Z, Js)


@element(num_params=1, units=["Ohm sec^-1/2"])
def W(p, f, Js):
    """defines a semi-infinite Warburg element

    Notes
    -----
    .. math::

        Z = \\frac{A_W}{\\sqrt{ 2 \\pi f}} (1-j)
    """
    omega = 2 * np.pi * np.array(f)
    Aw = p[0]
    Z = Aw * (1 - 1j) / np.sqrt(omega)
    if Js is None:
        return Z
    if Js[0] is not None:
        Js[0][:] = (1 - 1j) / np.sqrt(omega)
    return (Z, Js)


@element(num_params=2, units=["Ohm", "sec"])
def Wo(p, f, Js):
    """defines an open (finite-space) Warburg element

    Notes
    ---------
    .. math::
        Z = \\frac{Z_0}{\\sqrt{ j \\omega \\tau }}
        \\coth{\\sqrt{j \\omega \\tau }}
          = \\frac{Z_0}{\\sqrt{ j \\omega \\tau }
                    \\tanh{\\sqrt{j \\omega \\tau }}}
    where :math:`Z_0` = p[0] (Ohms) and
    :math:`\\tau` = p[1] (sec) = :math:`\\frac{L^2}{D}`

    """
    omega = 2 * np.pi * np.array(f)
    Z0, tau = p[0], p[1]

    arg = np.sqrt(1j * omega * tau)
    tanh = np.ones_like(arg, dtype=complex)
    tanh[arg.real <= -100] = -1.0
    mask = np.abs(arg.real) < 100
    tanh[mask] = np.tanh(arg[mask])

    Z = Z0 / (arg * tanh)
    if Js is None:
        return Z

    # d/dx coth = -csch^2
    csch2 = np.zeros_like(arg, dtype=complex)
    csch2[mask] = 1.0 / (np.sinh(arg[mask])**2)
    # darg/dtau = 1/(2 tau^0.5) * (1j*omega)^0.5 = arg/(2*tau)
    # (d/dtau) z = -Z0/arg*coth * (arg^-1*arg') [1st part]
    #            = -Z0/arg*coth * (arg^-1) * arg/(2*tau)
    # (d/dtau) z = - Z0/arg*csch^2*arg'  [2nd part]
    #            = -Z0/arg*csch^2*(arg/(2*tau))
    if Js[0] is not None:
        Js[0][:] = Z / Z0
    if Js[1] is not None:
        Js[1][:] = - Z / (2 * tau) - Z0 * csch2 / (2 * tau)
    return (Z, Js)


@element(num_params=2, units=["Ohm", "sec"])
def Ws(p, f, Js):
    """defines a short (finite-length) Warburg element

    Notes
    ---------
    .. math::
        Z = \\frac{Z_0}{\\sqrt{ j \\omega \\tau }}
        \\tanh{\\sqrt{j \\omega \\tau }}

    where :math:`Z_0` = p[0] (Ohms) and
    :math:`\\tau` = p[1] (sec) = :math:`\\frac{L^2}{D}`

    """
    omega = 2 * np.pi * np.array(f)
    Z0, tau = p[0], p[1]

    arg = np.sqrt(1j * omega * tau)
    tanh = np.ones_like(arg, dtype=complex)
    tanh[arg.real <= -100] = -1.0
    mask = np.abs(arg.real) < 100
    tanh[mask] = np.tanh(arg[mask])

    Z = Z0 * tanh / arg
    if Js is None:
        return Z

    # d/dx tanh = sech^2
    sech2 = np.zeros_like(arg, dtype=complex)
    sech2[mask] = 1.0 / (np.cosh(arg[mask])**2)

    if Js[0] is not None:
        Js[0][:] = Z / Z0
    if Js[1] is not None:
        Js[1][:] = Z0 * sech2 / (2 * tau) - Z / (2 * tau)
    return (Z, Js)


@element(num_params=2, units=["Ohm^-1 sec^a", ""])
def CPE(p, f, Js):
    """defines a constant phase element

    Notes
    -----
    .. math::

        Z = \\frac{1}{Q \\times (j 2 \\pi f)^\\alpha}

    where :math:`Q` = p[0] and :math:`\\alpha` = p[1].
    """
    omega = 2 * np.pi * np.array(f)
    Q, alpha = p[0], p[1]
    Z = 1.0 / (Q * (1j * omega) ** alpha)
    if Js is None:
        return Z

    # da^x/dx = lna * a^x
    # dZ/dalpha = -z/(1j*omega)**alpha) * (1j*omega)**alpha*ln(1j*omega)
    if Js[0] is not None:
        Js[0][:] = - Z / Q
    if Js[1] is not None:
        Js[1][:] = - Z * np.log(1j * omega)
    return (Z, Js)


@element(num_params=2, units=["H sec", ""])
def La(p, f, Js):
    """defines a modified inductance element as represented in [1]

    Notes
    -----
    .. math::

        Z = L \\times (j 2 \\pi f)^\\alpha

    where :math:`L` = p[0] and :math:`\\alpha` = p[1]

    [1] `EC-Lab Application Note 42, BioLogic Instruments (2019)
    <https://www.biologic.net/documents/battery-eis-modified-inductance-element-electrochemsitry-application-note-42>`_.
    """
    omega = 2 * np.pi * np.array(f)
    L, alpha = p[0], p[1]
    Z = (L * 1j * omega) ** alpha
    if Js is None:
        return Z

    if Js[0] is not None:
        Js[0][:] = alpha * Z / L
    if Js[1] is not None:
        Js[1][:] = Z * np.log(L * 1j * omega)
    return (Z, Js)


@element(num_params=2, units=["Ohm", "sec"])
def G(p, f, Js):
    """defines a Gerischer Element as represented in [1]

    Notes
    ---------
    .. math::

        Z = \\frac{R_G}{\\sqrt{1 + j \\, 2 \\pi f \\, t_G}}

    where :math:`R_G` = p[0] and :math:`t_G` = p[1]

    Gerischer impedance is also commonly represented as [2]:

    .. math::

        Z = \\frac{Z_o}{\\sqrt{K+ j \\, 2 \\pi f}}

    where :math:`Z_o = \\frac{R_G}{\\sqrt{t_G}}`
    and :math:`K = \\frac{1}{t_G}`
    with units :math:`\\Omega sec^{1/2}` and
    :math:`sec^{-1}` , respectively.

    [1] Y. Lu, C. Kreller, and S.B. Adler,
    Journal of The Electrochemical Society, 156, B513-B525 (2009)
    `doi:10.1149/1.3079337
    <https://doi.org/10.1149/1.3079337>`_.

    [2] M. González-Cuenca, W. Zipprich, B.A. Boukamp,
    G. Pudmich, and F. Tietz, Fuel Cells, 1,
    256-264 (2001) `doi:10.1016/0013-4686(93)85083-B
    <https://doi.org/10.1016/0013-4686(93)85083-B>`_.
    """
    omega = 2 * np.pi * np.array(f)
    R_G, t_G = p[0], p[1]
    u = np.sqrt(1 + 1j * omega * t_G)
    Z = R_G / u
    if Js is None:
        return Z

    if Js[0] is not None:
        Js[0][:] = Z / R_G
    if Js[1] is not None:
        Js[1][:] = - Z * 1j * omega / (2 * u**2)
    return (Z, Js)


@element(num_params=3, units=["Ohm", "sec", ""])
def Gs(p, f, Js):
    """defines a finite-length Gerischer Element as represented in [1]

    Notes
    ---------
    .. math::

        Z = \\frac{R_G}{\\sqrt{1 + j \\, 2 \\pi f \\, t_G} \\,
        tanh(\\phi \\sqrt{1 + j \\, 2 \\pi f \\, t_G})}

    where :math:`R_G` = p[0], :math:`t_G` = p[1] and :math:`\\phi` = p[2]

    [1] R.D. Green, C.C Liu, and S.B. Adler,
    Solid State Ionics, 179, 647-660 (2008)
    `doi:10.1016/j.ssi.2008.04.024
    <https://doi.org/10.1016/j.ssi.2008.04.024>`_.
    """
    omega = 2 * np.pi * np.array(f)
    R_G, t_G, phi = p[0], p[1], p[2]

    arg = phi * np.sqrt(1 + 1j * omega * t_G)
    tanh = np.ones_like(arg, dtype=complex)
    tanh[arg.real <= -100] = -1.0
    mask = np.abs(arg.real) < 100
    tanh[mask] = np.tanh(arg[mask])

    Z = R_G / (np.sqrt(1 + 1j * omega * t_G) * tanh)
    if Js is None:
        return Z

    # u = np.sqrt(1 + 1j * omega * t_G) = arg/phi
    if Js[0] is not None:
        Js[0][:] = Z / R_G

    """
    Z=R_G/(np.sqrt(1+1j*omega*t_G)*tanh(arg))
    let dz/dtG = v1+v2
    let arg_phi = np.sqrt(1+1j*omega*t_G)
    v1 = -z/(arg_phi)*(1/(2*arg_phi) * 1j*omega = -z*1j*omega/(2*arg_phi^2)
    v2 = -R_G/(arg_phi*sinh^2(arg)) * 0.5*phi*1j*omega/arg_phi
       = -R_G/(2*arg_phi^2*sinh^2(arg)) * phi*1j*omega
    """
    argp = np.sqrt(1 + 1j * omega * t_G)
    csch2 = np.zeros_like(arg, dtype=complex)
    csch2[mask] = 1.0 / (np.sinh(arg[mask])**2)
    if Js[1] is not None:
        Js[1][:] = - Z * 1j * omega / (2 * argp ** 2) \
                 - R_G * 1j * omega * phi**3 * csch2 / (2 * arg**2)

    """
    diff(Z,phi) = -(2*R_G)/(cosh(2*phi*(1+1j*omega*t_G)^(1/2)) - 1)
    Since arg = phi * (1 + 1j * omega * t_G)^.5
    diff(Z,phi) = -(2*R_G)/(cosh(2*arg) - 1)
    Since cosh(2*arg) - 1 = cosh^2(arg) + sinh^2(arg) - 1 = 2 sinh^2(arg)
    diff(Z,phi) = -R_G/sinh^2(arg)
    """
    if Js[2] is not None:
        Js[2][:] = - R_G * csch2
    return (Z, Js)


@element(num_params=2, units=["Ohm", "sec"])
def K(p, f, Js):
    """An RC element for use in lin-KK model

    Notes
    -----
    .. math::

        Z = \\frac{R}{1 + j \\omega \\tau_k}

    """
    omega = 2 * np.pi * np.array(f)
    R, tau_k = p[0], p[1]
    Z = R / (1 + 1j * omega * tau_k)
    if Js is None:
        return Z

    if Js[0] is not None:
        Js[0][:] = Z / R
    if Js[1] is not None:
        Js[1][:] = - Z * 1j * omega / (1 + 1j * omega * tau_k)
    return (Z, Js)


@element(num_params=3, units=['Ohm', 'sec', ''])
def Zarc(p, f, Js):
    """ An RQ element rewritten with resistance and
    and time constant as paramenters. Equivalent to a
    Cole-Cole relaxation in dielectrics.

    Notes
    -----
    .. math::

        Z = \\frac{R}{1 + (j \\omega \\tau_k)^\\gamma }

    """
    omega = 2 * np.pi * np.array(f)
    R, tau_k, gamma = p[0], p[1], p[2]
    Z = R / (1 + ((1j * omega * tau_k) ** gamma))
    if Js is None:
        return Z

    if Js[0] is not None:
        Js[0][:] = Z / R
    if Js[1] is not None:
        Js[1][:] = - Z**2 / R * gamma * ((1j * omega * tau_k) ** gamma) \
                    / tau_k
    if Js[2] is not None:
        Js[2][:] = - Z**2 / R * ((1j * omega * tau_k) ** gamma) \
                     * np.log(1j * omega * tau_k)
    return (Z, Js)


@element(num_params=3, units=["Ohm", "F sec^(gamma - 1)", ""])
def TLMQ(p, f, Js):
    """Simplified transmission-line model as defined in Eq. 11 of [1]

    Notes
    -----
    .. math::

        Z = \\sqrt{R_{ion}Z_{S}} \\coth \\sqrt{\\frac{R_{ion}}{Z_{S}}}


    [1] J. Landesfeind et al.,
    Journal of The Electrochemical Society, 163 (7) A1373-A1387 (2016)
    `doi: 10.1016/10.1149/2.1141607jes
    <http://doi.org/10.1149/2.1141607jes>`_.
    """
    omega = 2 * np.pi * np.array(f)
    Rion, Qs, gamma = p[0], p[1], p[2]
    Zs = 1 / (Qs * (1j * omega) ** gamma)

    arg = np.sqrt(Rion / Zs)
    tanh = np.ones_like(arg, dtype=complex)
    tanh[arg.real <= -100] = -1.0
    mask = np.abs(arg.real) < 100
    tanh[mask] = np.tanh(arg[mask])

    Z = np.sqrt(Rion * Zs) / tanh
    if Js is None:
        return Z
    csch2 = np.zeros_like(arg, dtype=complex)
    csch2[mask] = 1.0 / (np.sinh(arg[mask])**2)
    if Js[0] is not None:
        Js[0][:] = Z / (2 * Rion) - 0.5 * csch2
    if Js[1] is not None:
        Js[1][:] = - Z / (2 * Qs) - Rion * csch2 / (2 * Qs)
    if Js[2] is not None:
        Js[2][:] = (- Z - Rion * csch2) * 0.5 * np.log(1j * omega)
    return (Z, Js)


@element(num_params=4, units=["Ohm-m^2", "Ohm-m^2", "", "sec"])
def T(p, f, Js):
    """A macrohomogeneous porous electrode model from Paasch et al. [1]

    Notes
    -----
    .. math::

        Z = A\\frac{\\coth{\\beta}}{\\beta} + B\\frac{1}{\\beta\\sinh{\\beta}}

    where

    .. math::

        A = d\\frac{\\rho_1^2 + \\rho_2^2}{\\rho_1 + \\rho_2} \\quad
        B = d\\frac{2 \\rho_1 \\rho_2}{\\rho_1 + \\rho_2}

    and

    .. math::
        \\beta = (a + j \\omega b)^{1/2} \\quad
        a = \\frac{k d^2}{K} \\quad b = \\frac{d^2}{K}


    [1] G. Paasch, K. Micka, and P. Gersdorf,
    Electrochimica Acta, 38, 2653–2662 (1993)
    `doi: 10.1016/0013-4686(93)85083-B
    <https://doi.org/10.1016/0013-4686(93)85083-B>`_.
    """

    omega = 2 * np.pi * np.array(f)
    A, B, a, b = p[0], p[1], p[2], p[3]
    beta = (a + 1j * omega * b) ** (1 / 2)

    mask = np.abs(beta.real) < 100

    sinh = np.ones_like(beta, dtype=complex) * 1e10
    sinh[beta.real <= -100] = -1e10
    sinh[mask] = np.sinh(beta[mask])

    tanh = np.ones_like(beta, dtype=complex)
    tanh[beta.real <= -100] = -1.0
    tanh[mask] = np.tanh(beta[mask])

    Z = A / (beta * tanh) + B / (beta * sinh)
    if Js is None:
        return Z

    coth = 1.0 / tanh
    csch = 1.0 / sinh
    dz_dbeta = \
        A * (- coth / beta**2 - csch**2 / beta) + \
        B * (- csch / beta**2 - csch * coth / beta)

    if Js[0] is not None:
        Js[0][:] = coth / beta
    if Js[1] is not None:
        Js[1][:] = csch / beta
    if Js[2] is not None:
        Js[2][:] = dz_dbeta / (2 * beta)
    if Js[3] is not None:
        Js[3][:] = dz_dbeta * 1j * omega / (2 * beta)
    return (Z, Js)


def get_element_from_name(name):
    excluded_chars = "0123456789_"
    return "".join(char for char in name if char not in excluded_chars)


def typeChecker(p, f, name, length):
    assert isinstance(p, list), \
        "in {}, input must be of type list".format(name)
    for i in p:
        assert isinstance(
            i, (float, int, np.int32, np.float64)
        ), "in {}, value {} in {} is not a number".format(name, i, p)
    for i in f:
        assert isinstance(
            i, (float, int, np.int32, np.float64)
        ), "in {}, value {} in {} is not a number".format(name, i, f)
    assert len(p) == length, "in {}, input list must be length {}".format(
        name, length
    )
    return


def jacChecker(Js, f, name, length):
    assert isinstance(Js, list), \
        "in {}, Js-input must be of type list".format(name)
    assert len(Js) == length, \
        "in {}, gradinet list must be length {}".format(name, length)
    Nf = len(f)
    for dz in Js:
        assert isinstance(dz, (np.ndarray)), \
            "in {}, value {} in {} must be a numpy array".format(name, dz, Js)
        assert len(dz) == Nf, \
            "in {}, len({}) must be equal to {}".format(name, dz, Nf)
    return
