import numpy as np
from pydune.math import cosd, sind
from scipy.integrate import solve_ivp
from pydune.physics.turbulent_flow.fourriere2010_unbounded import mu, mu_prime

# ##################### Solving linear system


def _P(eta, alpha, eta_0, Kappa):
    P1 = [0, 0, -1j*cosd(alpha), mu_prime(eta, eta_0, Kappa)/2, 0, 0]
    P2 = [0, 0, -1j*sind(alpha), 0, mu_prime(eta, eta_0, Kappa), 0]
    P3 = [-1j*cosd(alpha), -1j*sind(alpha), 0, 0, 0, 0]
    P4 = [(1 + 3*cosd(alpha)**2)/mu_prime(eta, eta_0, Kappa) + 1j*mu(eta, eta_0, Kappa)*cosd(alpha), 3*sind(alpha)*cosd(alpha)/mu_prime(eta, eta_0, Kappa), mu_prime(eta, eta_0, Kappa), 0, 0, 1j*cosd(alpha)]
    P5 = [3*sind(alpha)*cosd(alpha)/mu_prime(eta, eta_0, Kappa), (1 + 3*sind(alpha)**2)/mu_prime(eta, eta_0, Kappa) + 1j*mu(eta, eta_0, Kappa)*cosd(alpha), 0, 0, 0, 1j*sind(alpha)]
    P6 = [0, 0, -1j*mu(eta, eta_0, Kappa)*cosd(alpha), 1j*cosd(alpha), 1j*sind(alpha), 0]
    #
    P = np.array([P1, P2, P3, P4, P5, P6])
    return P


def _S(eta, eta_0, Kappa):
    return np.array([Kappa*mu_prime(eta, eta_0, Kappa)**2, 0, 0, 0, 0, 0])


def _func(eta, X, alpha, eta_0, Kappa):
    return np.dot(_P(eta, alpha, eta_0, Kappa), X)


def _func1(eta, X, alpha, eta_0, Kappa):
    return np.dot(_P(eta, alpha, eta_0, Kappa), X) + _S(eta, eta_0, Kappa)


def _solve_system(eta_0, eta_H, alpha, Kappa=0.4, max_z=None,
                  dense_output=True, **kwargs):
    eta_span = [0, max_z]
    X0_vec = [np.array([-mu_prime(0, eta_0, Kappa), 0*1j, 0, 0, 0, 0]),
              np.array([0, 0*1j, 0, 1, 0, 0]),
              np.array([0, 0*1j, 0, 0, 1, 0]),
              np.array([0, 0*1j, 0, 0, 0, 1])]
    Results = []
    for i, X0 in enumerate(X0_vec):
        if i == 0:
            test = solve_ivp(_func1, eta_span, X0, args=(alpha, eta_0, Kappa),
                             dense_output=True, **kwargs)
        else:
            test = solve_ivp(_func, eta_span, X0, args=(alpha, eta_0, Kappa),
                             dense_output=True, **kwargs)
        Results.append(test)
    return Results


def calculate_solution(eta_0, eta_H, alpha, max_z=None, Kappa=0.4,
                       atol=1e-10, rtol=1e-10, method='DOP853', **kwargs):
    if max_z is None:
        max_z = 0.9999*eta_H
    Results = _solve_system(eta_0, eta_H, alpha, Kappa=0.4,
                            max_z=max_z, atol=atol, rtol=rtol, method=method,  **kwargs)
    # Defining boundary conditions
    To_apply = np.array([X.sol(max_z)[2:-1] for X in Results]).T
    #
    # ### Applying boundary conditions at the infinity (in eta_H very large)
    b = np.array([0,
                  0,
                  0])
    # breakpoint()
    # Applying boundary condition
    pars = np.dot(np.linalg.inv(To_apply[:, 1:]), b - To_apply[:, 0])  # axz, ayz, an
    coeffs = np.array([1, pars[0], pars[1], pars[2]])
    #

    def interpolated_solution(eta):
        coeffs_expanded = np.expand_dims(coeffs, (1, ) + tuple(np.arange(len(np.array(eta).shape)) + 2))
        return np.sum(np.array([X.sol(eta) for X in Results])*coeffs_expanded, axis=0)
    return interpolated_solution
