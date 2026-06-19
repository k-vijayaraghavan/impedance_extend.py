import string
import warnings

import numpy as np
import pytest

from impedance_extend.models.circuits.elements import OverwriteError, \
                                                            circuit_elements, \
                                                            element, p, s, \
                                                            ElementError


def test_each_element():
    freqs = [0.001, 1.0, 1000]
    correct_vals = {
        "R": [0.1, 0.1, 0.1],
        "C": [
            -1591.5494309189532j,
            -1.5915494309189535j,
            -0.0015915494309189533j,
        ],
        "L": [0.000628319j, 0.628319j, 628.319j],
        "W": [
            (1.26156626 - 1.26156626j),
            (0.03989423 - 0.03989423j),
            (0.00126157 - 0.00126157j),
        ],
        "Wo": [
            (0.033333332999112786 - 79.57747433847442j),
            (0.03300437046673635 - 0.08232866785870396j),
            (0.0019947114020071634 - 0.0019947114020071634j),
        ],
        "Ws": [
            (0.09999998 - 4.18878913e-05j),
            (0.08327519 - 3.33838167e-02j),
            (0.00199471 - 1.99471140e-03j),
        ],
        "CPE": [
            (26.216236841407248 - 8.51817171087997j),
            (6.585220960717244 - 2.139667994182814j),
            (1.6541327179718126 - 0.537460300252313j),
        ],
        "La": [
            (0.21769191 + 0.07073239j),
            (0.86664712 + 0.28159072j),
            (3.45018434 + 1.12103285j),
        ],
        "G": [
            (0.09999994078244179 - 0.00006283179105931961j),
            (0.07107755021941357 - 0.03427465211788068j),
            (0.00199550459845528 - 0.0019939172581851707j),
        ],
        "Gs": [
            (0.3432733166533134 - 0.00041895248193532704j),
            (0.1391819314527732 - 0.16248466787637972j),
            (0.0019955029598887875 - 0.0019939170758457437j),
        ],
        "TLMQ": [
            (20.42237173 - 10.38874276j),
            (2.6000937 - 1.30789949j),
            (0.35594139 - 0.16491599j),
        ],
        "T": [
            (1.00041 - 0.00837309j),
            (0.0156037 - 0.114062j),
            (0.00141056 - 0.00141039j),
        ],
        "K": [
            (0.099999842086579 - 0.000125663507704j),
            (0.038772663673915 - 0.048723166143232j),
            (6.332569967499333e-08 - 7.957742115295703e-05j),
        ],
        "Zarc": [
            (0.08900974 - 0.00486384j),
            (0.04818879 - 0.01198904j),
            (0.00969182 - 0.00436265j),
        ],
    }
    input_vals = [0.1, 0.2, 0.3, 0.4]
    for key, f in circuit_elements.items():
        # don't test the outputs of series and parallel functions
        if key not in ["s", "p", "np"]:
            num_inputs = f.num_params
            val = f(input_vals[:num_inputs], freqs)
            assert np.isclose(val, correct_vals[key]).all()

        # check for typing:
        with pytest.raises(AssertionError):
            f = circuit_elements["R"]
            f(1, 2)

        # test for handling more wrong inputs
        with pytest.raises(AssertionError):
            f(["hi"], ["yes", "hello"])

    # Test no overflow in T at high frequencies
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        circuit_elements["T"]([1, 2, 50, 100], [10000])


def test_each_element_deravative():
    freqs = [0.001, 1.0, 1000]
    input_vals = [0.1, 0.2, 0.3, 0.4]
    for key, f in circuit_elements.items():
        # don't test the outputs of series and parallel functions
        if key not in ["s", "p", "np"]:
            num_inputs = f.num_params
            input = input_vals[:num_inputs]
            dzdp = np.zeros((len(freqs), num_inputs), dtype=complex)
            dzdp_ = [dzdp[:, i] for i in range(num_inputs)]
            val = f(input, freqs, dzdp_)[0]
            # Loop small varn
            input_ = input.copy()
            dzdp_num = np.zeros((len(freqs), num_inputs), dtype=complex)
            for i in range(num_inputs):
                input_[i] = 1.01*input[i]
                val_ = f(input_, freqs)
                dzdp_num[:, i] = (val_-val)/(0.01*input[i])
                input_[i] = input[i]
            assert np.isclose(dzdp, dzdp_num, rtol=0.025).all()


def test_s():
    a = np.array([5 + 6 * 1j, 2 + 3 * 1j])
    b = np.array([5 + 6 * 1j, 2 + 3 * 1j])

    answer = np.array([10 + 12 * 1j, 4 + 6 * 1j])
    assert np.isclose(s([a, b]), answer).all()


def test_p():
    a = np.array([5 + 6 * 1j, 2 + 3 * 1j])
    b = np.array([5 + 6 * 1j, 2 + 3 * 1j])

    answer = np.array([2.5 + 3 * 1j, 1 + 1.5 * 1j])
    assert np.isclose(p([a, b]), answer).all()


def test_s_deravative():
    freqs = [0.001, 1.0, 1000]
    input2_vals = [0.1, 0.2, 0.3, 0.4]
    input1_vals = [0.5]
    # Try R-ele
    comb = s
    f1 = circuit_elements["R"]
    n_ins1 = 1
    for key, f2 in circuit_elements.items():
        if key not in ["s", "p", "np"]:
            n_ins2 = f2.num_params
            input1 = input1_vals[0:n_ins1]
            dzdp = np.zeros((len(freqs), n_ins1+n_ins2), dtype=complex)
            dzdp1_ = [dzdp[:, i] for i in range(n_ins1)]
            val1 = f1(input1, freqs, dzdp1_)
            input2 = input2_vals[:n_ins2]
            dzdp_ = [dzdp[:, i] for i in range(n_ins1,n_ins1+n_ins2)]
            val2 = f2(input2, freqs, dzdp_)
            val = comb([val1, val2])
            # Loop small varn
            input_ = input1.copy()
            dzdp_num = np.zeros((len(freqs), n_ins1+n_ins2), dtype=complex)
            for i in range(n_ins1):
                input_[i] = 1.01*input1[i]
                val1_ = f1(input_, freqs)
                val_ = comb([val1_, val2[0]])
                dzdp_num[:, i] = (val_-val[0])/(0.01*input1[i])
                input_[i] = input1[i]
            input_ = input2.copy()
            for i in range(n_ins2):
                input_[i] = 1.01*input2[i]
                val2_ = f2(input_, freqs)
                val_ = comb([val1[0], val2_])
                dzdp_num[:, n_ins1+i] = (val_-val[0])/(0.01*input2[i])
                input_[i] = input2[i]
            assert np.isclose(dzdp, dzdp_num, rtol=0.025).all()


def test_p_deravative():
    freqs = [0.001, 1.0, 1000]
    input2_vals = [0.1, 0.2, 0.3, 0.4]
    input1_vals = [0.5]
    # Try R-ele
    comb = p
    f1 = circuit_elements["R"]
    n_ins1 = 1
    for key, f2 in circuit_elements.items():
        if key not in ["s", "p", "np"]:
            n_ins2 = f2.num_params
            input1 = input1_vals[0:n_ins1]
            dzdp = np.zeros((len(freqs), n_ins1+n_ins2), dtype=complex)
            dzdp1_ = [dzdp[:, i] for i in range(n_ins1)]
            val1 = f1(input1, freqs, dzdp1_)
            input2 = input2_vals[:n_ins2]
            dzdp_ = [dzdp[:, i] for i in range(n_ins1,n_ins1+n_ins2)]
            val2 = f2(input2, freqs, dzdp_)
            val = comb([val1, val2])
            # Loop small varn
            input_ = input1.copy()
            dzdp_num = np.zeros((len(freqs), n_ins1+n_ins2), dtype=complex)
            for i in range(n_ins1):
                input_[i] = 1.01*input1[i]
                val1_ = f1(input_, freqs)
                val_ = comb([val1_, val2[0]])
                dzdp_num[:, i] = (val_-val[0])/(0.01*input1[i])
                input_[i] = input1[i]
            input_ = input2.copy()
            for i in range(n_ins2):
                input_[i] = 1.01*input2[i]
                val2_ = f2(input_, freqs)
                val_ = comb([val1[0], val2_])
                dzdp_num[:, n_ins1+i] = (val_-val[0])/(0.01*input2[i])
                input_[i] = input2[i]
            assert np.isclose(dzdp, dzdp_num, rtol=0.025).all()


def test_element_function_names():
    # run a simple check to ensure there are no integers
    # in the function names
    letters = string.ascii_uppercase + string.ascii_lowercase

    for elem in circuit_elements.keys():
        for char in elem:
            assert (
                char in letters
            ), f"{char} in {elem} is not in the allowed set of {letters}"


def test_changing_base_functions_fails():
    with pytest.raises(ElementError):
        @element(num_params=1, units=["Ohm"])
        def s(p, f):
            # try redefining the series
            return np.nan


def test_add_element():
    # checks if you can add your own custom element
    assert "NE" not in circuit_elements

    @element(num_params=1, units=["Ohm"])
    def NE(p, f, dzdp):
        """definitely a new circuit element no one has seen before

        Notes
        ---------
        .. math::

            Z = R

        """
        R = p[0]
        Z = np.array(len(f) * [R])
        return Z

    assert "NE" in circuit_elements


def test_add_element_overwrite_fails():
    # checks if you can add your own custom element
    # and then overwriting it raises an OverwriteError
    assert "NE2" not in circuit_elements

    @element(num_params=1, units=["Ohm"])
    def NE2(p, f, dzdp):
        """definitely a new circuit element no one has seen before

        Notes
        ---------
        .. math::

            Z = R

        """
        R = p[0]
        Z = np.array(len(f) * [R])
        return Z

    assert "NE2" in circuit_elements
    with pytest.raises(OverwriteError):
        # try to create the same element again without overwrite
        @element(num_params=1, units=["Ohm"])
        def NE2(p, f, dzdp):  # noqa: F811
            """definitely a new circuit element no one has seen before

            Notes
            ---------
            .. math::

                Z = R

            """
            R = p[0]
            Z = np.array(len(f) * [R])
            return Z


def test_add_element_overwrite():
    # checks if you can add your own custom element
    # and then overwriting it is allowed with correct kwarg
    assert "NE3" not in circuit_elements

    @element(num_params=1, units=["Ohm"])
    def NE3(p, f, dzdp):
        return [p * ff for ff in f]

    assert "NE3" in circuit_elements
    assert circuit_elements["NE3"]([1], [1]) == [[1]]
    # try to create the same element again with overwrite

    @element(num_params=1, units=["Ohm"], overwrite=True)
    def NE3(p, f, dzdp):  # noqa: F811
        # feel free to change to a better test
        return [p * ff * 2 for ff in f]

    assert "NE3" in circuit_elements
    assert circuit_elements["NE3"]([1], [1]) == [[1, 1]]
