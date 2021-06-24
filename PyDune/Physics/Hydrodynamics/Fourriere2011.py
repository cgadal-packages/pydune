# @Author: gadal
# @Date:   2021-01-08T11:02:13+01:00
# @Email:  gadal@ipgp.fr
# @Last modified by:   gadal
# @Last modified time: 2021-02-11T16:03:32+01:00

import numpy as np
import matplotlib.pyplot as plt
from General.Math import cosd, sind, Rotation_matrix, tand
from scipy.integrate import solve_ivp

###############################################################################
############################### Without free Surface ##########################
###############################################################################

################################### Hydrodynamic coefficients
def function_coeff(R, a):
    return a[0] + (a[1] + a[2]*R + a[3]*R**2 + a[4]*R**3)/(1 + a[5]*R**2 + a[6]*R**4)

def coeffA(eta_0):
    R = np.log(2*np.pi/eta_0)
    a = [2, 1.0702, 0.093069, 0.10838, 0.024835, 0.041603, 0.0010625]
    return function_coeff(R, a)

def coeffB(eta_0):
    R = np.log(2*np.pi/eta_0)
    b = [0, 0.036989, 0.15765, 0.11518, 0.0020249, 0.0028725, 0.00053483]
    return function_coeff(R, b)


###################### Solving linear system
def mu(eta, eta_0, Kappa = 0.4):
    """
    eta = k z, vertical coordinate [Adi.]
    eta_0 = k z0, hydrodynamic roughness [Adi.]
    Kappa, Von Karman constant (typically 0.4)
    """
    return (1/Kappa)*np.log(1 + eta/eta_0)

def mu_prime(eta, eta_0, Kappa = 0.4):
    """
    eta = k z, vertical coordinate [Adi.]
    Kappa, Von Karman constant (typically 0.4)
    """
    return (1/Kappa)*(1/(eta + eta_0))

def P(eta, eta_0, Kappa):
    """P matrix of Fourrier et al. 2010:

    eta = k z, vertical coordinate [Adi]

    ## parameters
    eta_0 = k z0, hydrodynamic roughness [Adi.]
    Kappa, Von Karman constant (typically 0.4)
    """

    P1 = [0, -1j, mu_prime(eta, eta_0, Kappa)/2, 0]
    P2 = [-1j, 0, 0, 0]
    P3 = [1j*mu(eta, eta_0, Kappa) + 4/mu_prime(eta, eta_0, Kappa), mu_prime(eta, eta_0, Kappa), 0, 1j]
    P4 = [0, -1j*mu(eta, eta_0, Kappa), 1j, 0]
    #
    P = np.array([P1, P2, P3, P4])
    return P

def S(eta, eta_0, Kappa):
    return np.array([Kappa*mu_prime(eta, eta_0, Kappa)**2, 0, 0, 0])


def func(eta, X, eta_0, Kappa):
    return np.dot(P(eta, eta_0, Kappa), X)

def func1(eta, X, eta_0, Kappa):
    return np.dot(P(eta, eta_0, Kappa), X) + S(eta, eta_0, Kappa)


def Solve_lin(eta_0, eta_H, Kappa = 0.4):
    eta_val = np.linspace(0, eta_H, 100)
    eta_span = [eta_val.min(), eta_val.max()]
    X0_vec = [np.array([-mu_prime(0, eta_0, Kappa), 0*1j, 0, 0]),
              np.array([0, 0*1j, 1, 0]),
              np.array([0, 0*1j, 0, 1])]
    Res = []
    for i, X0 in enumerate(X0_vec):
        # print(i)
        if i == 0:
            test = solve_ivp(func1, eta_span, X0, args = (eta_0, Kappa), t_eval = eta_val, method = 'DOP853', dense_output = True, rtol = 1e-8 , atol = 1e-8)
        else:
            test = solve_ivp(func, eta_span, X0, args = (eta_0, Kappa), t_eval = eta_val, method = 'DOP853', dense_output = True, rtol = 1e-8 , atol = 1e-8)
        Res.append(test.y)
    return np.array(Res), test.t

if __name__ == '__main__':
    eta_H = 10 ### giant size, lambda ~ Hboundarylayer
    Kappa = 0.4
    # eta_0_vals = np.logspace(-10, -8, 1)
    eta_0_vals = np.logspace(-10, 0, 100)
    Coeffs = np.zeros((2, eta_0_vals.size))
    for i, eta_0 in enumerate(eta_0_vals):
        print(i)
        #
        test2, eta_val = Solve_lin(eta_0, eta_H, Kappa = 0.4)
        #### Applying boundary conditions at the infinity (in eta_H very large)
        b = np.array([0,  ## no vertical velocity at the lid in eta = eta_H
        0])               ## no order 1 stress (contstant at order 0) at the lid

        A = np.array(test2[1:, 1:-1, -1]).T
        pars = np.dot(np.linalg.inv(A), - test2[0, 1:-1, -1]) #axz, ayz, an
        coeffs = np.array([1, pars[0], pars[1]])
        X = np.sum(test2*coeffs[:, None, None], axis = 0)
        # # #
        Ax, Bx = np.real(X[2, 0]), np.imag(X[2, 0])
        Coeffs[:, i] = [Ax, Bx]
        #

        # print('Ax = ', Ax)
        # print('Bx =', Bx)

    plt.figure()
    plt.semilogx(eta_0_vals, Coeffs[0, :])
    plt.semilogx(eta_0_vals, Coeffs[1, :])
    plt.tight_layout()
    plt.ylim([0, 5])
    plt.show()
