import numpy as np
from PyDune.math import tand, arcsind
from scipy.integrate import solve_ivp
from PyDune.physics.turbulent_flow.fourriere2010_unbounded import mu, mu_prime


# ##################### Solving linear system

def _P(eta, eta_H, eta_0, Kappa):
    tp = (1 - eta/eta_H)
    #
    P1 = [0, -1j, mu_prime(eta, eta_0, Kappa)/(2*tp), 0]
    P2 = [-1j, 0, 0, 0]
    P3 = [1j*mu(eta, eta_0, Kappa) + 4*tp/mu_prime(eta, eta_0, Kappa), mu_prime(eta, eta_0, Kappa), 0, 1j]
    P4 = [0, -1j*mu(eta, eta_0, Kappa), 1j, 0]
    #
    P = np.array([P1, P2, P3, P4])
    return P


def _S(eta, eta_H, eta_0, Kappa):
    return np.array([Kappa*mu_prime(eta, eta_0, Kappa)**2 - mu_prime(eta, eta_0, Kappa)/(2*eta_H), 0, 0, 0])


def _S_delta(eta, eta_H, eta_0, Kappa):
    tp = (1 - eta/eta_H)
    return np.array([-eta*mu_prime(eta, eta_0, Kappa)/(2*eta_H**2*tp), 0, 0, 0])


def _func(eta, X, eta_H, eta_0, Kappa):
    return np.dot(_P(eta, eta_H, eta_0, Kappa), X)


def _func1(eta, X, eta_H, eta_0, Kappa):
    return np.dot(_P(eta, eta_H, eta_0, Kappa), X) + _S(eta, eta_H, eta_0, Kappa)


def _func_delta(eta, X, eta_H, eta_0, Kappa):
    return np.dot(_P(eta, eta_H, eta_0, Kappa), X) + _S_delta(eta, eta_H, eta_0, Kappa)


def _solve_system(eta_0, eta_H, Kappa=0.4, max_z=None, dense_output=True, **kwargs):
    eta_span_tp = [0, max_z]
    X0_vec = [np.array([-mu_prime(0, eta_0, Kappa), 0*1j, 0, 0], dtype='complex_'),
              np.array([0, 0*1j, 1, 0], dtype='complex_'),
              np.array([0, 0*1j, 0, 1], dtype='complex_'),
              np.array([0, 0, 0, 0], dtype='complex_')]
    Results = []
    for i, X0 in enumerate(X0_vec):
        if i == 0:
            test = solve_ivp(_func1, eta_span_tp, X0, args=(eta_H, eta_0, Kappa),
                             dense_output=dense_output, **kwargs)
        elif i == 4:
            test = solve_ivp(_func_delta, eta_span_tp, X0, args=(eta_H, eta_0, Kappa),
                             dense_output=dense_output, **kwargs)
        else:
            test = solve_ivp(_func, eta_span_tp, X0, args=(eta_H, eta_0, Kappa),
                             dense_output=dense_output, **kwargs)
        Results.append(test)
    return Results


def calculate_solution(eta_0, eta_H, Fr, max_z=None, Kappa=0.4,
                       atol=1e-10, rtol=1e-10, method='DOP853', **kwargs):
    if max_z is None:
        max_z = 0.9999*eta_H
    Results = _solve_system(eta_0, eta_H, Kappa=0.4,
                            max_z=max_z, atol=atol, rtol=rtol, method=method, **kwargs)
    # Defining boundary conditions
    To_apply = np.array([X.sol(max_z)[1:] for X in Results]).T  # calculating intermediate solutions in eta_H only for W and St [1:-1]
    #
    # ### Applying boundary conditions
    theta = arcsind((Kappa*Fr/np.log(1 + eta_H/eta_0)))**2
    b = np.array([1j*mu(max_z, eta_0, Kappa),
                  1/max_z,
                  1/(max_z*tand(theta))])
    # Applying boundary condition
    pars = np.dot(np.linalg.inv(To_apply[:, :-1]), b - To_apply[:, -1])
    coeffs = np.array([1, pars[1]/pars[0], pars[2]/pars[0], 1/pars[0]])

    def interpolated_solution(eta):
        coeffs_expanded = np.expand_dims(coeffs, (1, ) + tuple(np.arange(len(np.array(eta).shape)) + 2))
        return np.sum(np.array([X.sol(eta) for X in Results])*coeffs_expanded, axis=0)
    return interpolated_solution
