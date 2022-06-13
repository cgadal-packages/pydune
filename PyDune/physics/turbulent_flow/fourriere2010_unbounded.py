import numpy as np
from scipy.integrate import solve_ivp

# ##################### coefficient approximation


def _function_coeff(R, a):
    return a[0] + (a[1] + a[2]*R + a[3]*R**2 + a[4]*R**3)/(1 + a[5]*R**2 + a[6]*R**4)


def A0_approx(eta_0):
    R = np.log(2*np.pi/eta_0)
    a = [2, 1.0702, 0.093069, 0.10838, 0.024835, 0.041603, 0.0010625]
    return _function_coeff(R, a)


def B0_approx(eta_0):
    R = np.log(2*np.pi/eta_0)
    b = [0, 0.036989, 0.15765, 0.11518, 0.0020249, 0.0028725, 0.00053483]
    return _function_coeff(R, b)


# ##################### solving linear model

def mu(eta, eta_0, Kappa=0.4):
    """
    eta = k z, vertical coordinate [Adi.]
    eta_0 = k z0, hydrodynamic roughness [Adi.]
    Kappa, Von Karman constant (typically 0.4)
    """
    return (1/Kappa)*np.log(1 + eta/eta_0)


def mu_prime(eta, eta_0, Kappa=0.4):
    r""" derivative of the ratio :math:`U(z)/u_{*}` following the law of the wall:

    ..:math::

        \frac{1}{u_{*}}\frac{\textup{d}U(z)}{\textup{d}z} = \frac{1}{\kappa}\frac{1}{(z + z_{0}}.


    Parameters
    ----------
    eta : scalar, np.array
        height
    eta_0 : scalar, np.array
        hydrodyamic roughness
    Kappa : float, optional
        Von Karmàn constant (the default is 0.4).

    Returns
    -------
    scalar, np.array
        Array of the ratio defined above.


    """

    return (1/Kappa)*(1/(eta + eta_0))


def _P(eta, eta_0, Kappa):
    P1 = [0, -1j, mu_prime(eta, eta_0, Kappa)/2, 0]
    P2 = [-1j, 0, 0, 0]
    P3 = [1j*mu(eta, eta_0, Kappa) + 4/mu_prime(eta, eta_0, Kappa), mu_prime(eta, eta_0, Kappa), 0, 1j]
    P4 = [0, -1j*mu(eta, eta_0, Kappa), 1j, 0]
    #
    P = np.array([P1, P2, P3, P4])
    return P


def _S(eta, eta_0, Kappa):
    return np.array([Kappa*mu_prime(eta, eta_0, Kappa)**2, 0, 0, 0])


def _func(eta, X, eta_0, Kappa):
    return np.dot(_P(eta, eta_0, Kappa), X)


def _func1(eta, X, eta_0, Kappa):
    return np.dot(_P(eta, eta_0, Kappa), X) + _S(eta, eta_0, Kappa)


def _solve_system(eta_0, eta_H, Kappa=0.4, max_z=None, dense_output=True, **kwargs):
    eta_span = [0, max_z]
    X0_vec = [np.array([-mu_prime(0, eta_0, Kappa), 0*1j, 0, 0], dtype='complex_'),
              np.array([0, 0*1j, 1, 0], dtype='complex_'),
              np.array([0, 0*1j, 0, 1], dtype='complex_')]
    Results = []
    for i, X0 in enumerate(X0_vec):
        if i == 0:
            test = solve_ivp(_func1, eta_span, X0, args=(eta_0, Kappa),
                             dense_output=dense_output, **kwargs)
        else:
            test = solve_ivp(_func, eta_span, X0, args=(eta_0, Kappa),
                             dense_output=dense_output, **kwargs)
        Results.append(test)
    return Results


def calculate_solution(eta_0, eta_H, max_z=None, Kappa=0.4, atol=1e-10,
                       rtol=1e-10, method='DOP853', **kwargs):
    if max_z is None:
        max_z = 0.9999*eta_H
    Results = _solve_system(eta_0, eta_H, Kappa=0.4,
                            max_z=max_z, atol=atol, rtol=rtol, method=method, **kwargs)
    # Defining boundary conditions
    To_apply = np.array([X.sol(max_z)[1:-1] for X in Results]).T  # calculating intermediate solutions in eta_H only for W and St [1:-1]
    #
    # ### Applying boundary conditions at the infinity (in eta_H very large)
    b = np.array([0,  # no vertical velocity at the lid in eta = eta_H
                  0])  # no order 1 stress (contstant at order 0) at the lid
    # Applying boundary condition
    pars = np.dot(np.linalg.inv(To_apply[:, 1:]), b - To_apply[:, 0])  # axz, ayz, an
    coeffs = np.array([1, pars[0], pars[1]])

    def interpolated_solution(eta):
        coeffs_expanded = np.expand_dims(coeffs, (1, ) + tuple(np.arange(len(np.array(eta).shape)) + 2))
        return np.sum(np.array([X.sol(eta) for X in Results])*coeffs_expanded, axis=0)
    return interpolated_solution
