r"""
Dune flat bed instability taking into account two spatial direction (x and y), under unidirectional, bidirectional and multidirectional wind regimes. Here, only the temporal instability is derived,
following Gadal et al. 2019.

This theory is developped assuming a quadratic transport law of the form

.. math::

    q_{\rm sat} = Q\left[1 - (u_{\rm th}/u_{*})^{2}\right] .

Here, all quantities are non dimensional:

- length scales by the saturation length :math:`L_{\rm sat}`.
- time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.


References
----------
[1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
[2] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.
"""


import numpy as np
from PyDune.General import cosd, sind


# def complex_pulsation(k, alpha, ax, bx, ay, by):
#     r"""Dispersion relation as the output of the temporal dune instability under a unidirectional wind:
#
#     .. math::
#
#         \omega = \frac{k^{2}}{1 + ik\cos\alpha}\left(\cos\alpha\left(\mathcal{a}_{x} + i\mathcal{b}_{x}\right) + \sin\alpha\left(\mathcal{a}_{y} + i\mathcal{b}_{y}\right)\right)
#
#     We use here the general expression without expliciting the form of the small hydrodynamic coefficients :math:`\mathcal{a}_{x}`, :math:`\mathcal{a}_{y}`, etc .. (Andreotti et al. 2012, Gadal et al. 2019).
#
#     Parameters
#     ----------
#     k : scalar, numpy array
#         Non dimensional wavenumber :math:`k`.
#     alpha : scalar, numpy array
#         Direction :math:`alpha` of the wavevector :math:`\boldsymbol{k}`. It is also the dune orientation measured with respect to the perpendicular to the wind direction .
#     ax : scalar, numpy array
#         Small hydrodynamic coefficient. Generally a function of :math:`k` and :math:`alpha`.
#     bx : scalar, numpy array
#         Small hydrodynamic coefficient. Generally a function of :math:`k` and :math:`alpha`.
#     ay : scalar, numpy array
#         Small hydrodynamic coefficient. Generally a function of :math:`k` and :math:`alpha`.
#     by : scalar, numpy array
#         Small hydrodynamic coefficient. Generally a function of :math:`k` and :math:`alpha`.
#
#     Returns
#     -------
#     scalar, numpy array
#         Complex pulsation calculated elementwise.
#
#     Notes
#     --------
#     Note that all quantities are made non dimensional:
#
#     - length scales by the saturation length :math:`L_{\rm sat}`.
#     - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.
#
#
#     Examples
#     --------
#     >>>
#
#     References
#     ----------
#     [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
#     [2] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.
#
#     """
#     return (k**2/(1 + 1j*k*cosd(alpha)))*(cosd(alpha)*(ax + 1j*bx) + sind(alpha)*(ay + 1j*by))


def temporal_growth_rate(k, alpha, Ax, Ay, Bx, By, r, mu, delta):
    r"""Dune instability temporal growth rate under a unidirectional wind - imaginary part of the complex pulsation.

    Parameters
    ----------
    k : scalar, numpy array
        Non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        Direction :math:`alpha` of the wavevector :math:`\boldsymbol{k}`. It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        Hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase). A vectorial function taking `k` as first argument and `alpha` as second one.
    Ay : function
        Hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase). A vectorial function taking `k` as first argument and `alpha` as second one.
    Bx : function
        Hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature). A vectorial function taking `k` as first argument and `alpha` as second one.
    By : function
        Hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature). A vectorial function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        Velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`
    mu : scalar, numpy array
        Friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        Cross-stream diffusion coefficient. Set to 0 if you want to recover the exact results of Andreotti et al. 2012, Gadal et al. 2019.

    Returns
    -------
    scalar, numpy array
        Temporal dune growth rate :math:`\sigma` calculated elementwise.

    Notes
    --------
    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.


    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0, 1, 500), k = np.linspace(0, 360, 600)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> # Chosing a form for the hydrodyanmic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> SIGMA = temporal_growth_rate(K, ALPHA, Ax, Ay, Bx, By, r, mu, delta)

    """
    ax = Ax(k, alpha)
    bx = Bx(k, alpha) - cosd(alpha)*(1/mu)*(1/r**2)
    ay = (1 - 1/r**2)*Ay(k, alpha) - delta*k*sind(alpha)*bx
    by = (1 - 1/r**2)*(By(k, alpha) - sind(alpha)*(1/mu)*(1/r)) - delta*k*sind(alpha)*ax
    return (k**2/(1 + (k*cosd(alpha))**2))*(bx*cosd(alpha) + by*sind(alpha) - k*cosd(alpha)*(ax*cosd(alpha) + ay*sind(alpha)))
    # return complex_pulsation(k, alpha, ax, bx, ay, by).imag


def temporal_pulsation(k, alpha, Ax, Ay, Bx, By, r, mu, delta):
    r"""Dune instability temporalpulsation under a unidirectional wind - real part of the complex pulsation.

    Parameters
    ----------
    k : scalar, numpy array
        Non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        Direction :math:`alpha` of the wavevector :math:`\boldsymbol{k}`. It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        Hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase). A vectorial function taking `k` as first argument and `alpha` as second one.
    Ay : function
        Hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase). A vectorial function taking `k` as first argument and `alpha` as second one.
    Bx : function
        Hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature). A vectorial function taking `k` as first argument and `alpha` as second one.
    By : function
        Hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature). A vectorial function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        Velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`
    mu : scalar, numpy array
        Friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        Cross-stream diffusion coefficient. Set to 0 if you want to recover the exact results of Andreotti et al. 2012, Gadal et al. 2019.

    Returns
    -------
    scalar, numpy array
        Temporal dune growth rate :math:`\sigma` calculated elementwise.

    Notes
    --------
    The dune migration velocity is then :math:`\omega/k`.

    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.


    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0, 1, 500), k = np.linspace(0, 360, 600)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> # Chosing a form for the hydrodyanmic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> PULSATION = temporal_pulsation(K, ALPHA, Ax, Ay, Bx, By, r, mu, delta)

    """
    ax = Ax(k, alpha)
    bx = Bx(k, alpha) - cosd(alpha)*(1/mu)*(1/r**2)
    ay = (1 - 1/r**2)*Ay(k, alpha) - delta*k*sind(alpha)*bx
    by = (1 - 1/r**2)*(By(k, alpha) - sind(alpha)*(1/mu)*(1/r)) - delta*k*sind(alpha)*ax
    # return complex_pulsation(k, alpha, ax, bx, ay, by).real
    return (k**2/(1 + (k*cosd(alpha))**2))*(ax*cosd(alpha) + ay*sind(alpha) + k*cosd(alpha)*(bx*cosd(alpha) + by*sind(alpha)))


def temporal_celerity(k, alpha, Ax, Ay, Bx, By, r, mu, delta):
    r"""Dune instability temporal celerity under a unidirectional wind - real part of the complex pulsation divided by the wavenumber.

    Parameters
    ----------
    k : scalar, numpy array
        Non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        Direction :math:`alpha` of the wavevector :math:`\boldsymbol{k}`. It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        Hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase). A vectorial function taking `k` as first argument and `alpha` as second one.
    Ay : function
        Hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase). A vectorial function taking `k` as first argument and `alpha` as second one.
    Bx : function
        Hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature). A vectorial function taking `k` as first argument and `alpha` as second one.
    By : function
        Hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature). A vectorial function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        Velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`
    mu : scalar, numpy array
        Friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        Cross-stream diffusion coefficient. Set to 0 if you want to recover the exact results of Andreotti et al. 2012, Gadal et al. 2019.

    Returns
    -------
    scalar, numpy array
        Temporal dune growth rate :math:`\sigma` calculated elementwise.

    Notes
    --------
    The dune migration velocity is then :math:`\omega/k`.

    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.


    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0, 1, 500), k = np.linspace(0, 360, 600)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> # Chosing a form for the hydrodyanmic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> CELERITY = temporal_celerity(K, ALPHA, Ax, Ay, Bx, By, r, mu, delta)

    """
    return temporal_pulsation(k, alpha, Ax, Ay, Bx, By, r, mu, delta)/k
    # return (k/(1 + (k*cosd(alpha))**2))*(ax*cosd(alpha) + ay*sind(alpha) + k*cosd(alpha)*(bx*cosd(alpha) + by*sind(alpha)))


################################################################################
"""
theta: angle betwen the two wind directions (deg.)
N: wind mass transport ratio
"""


def Growth_Rate_bidi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N):
    # changement de ref + prise en compte seulement de la direction et pas du sens de l'onde
    a1 = ((alpha + theta/2 + 90) % 180) - 90
    a2 = ((alpha - theta/2 + 90) % 180) - 90
    return (N/(N+1))*temporal_growth_rate(k, a1, Ax, Ay, Bx, By, r, mu, delta) + (1/(N+1))*temporal_growth_rate(k, a2, Ax, Ay, Bx, By, r, mu, delta)


def Celerity_bidi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N):
    # Has to be projected on the perpendicular to the most unstable orientation afterwards
    a1 = ((alpha + theta/2 + 90) % 180) - 90
    a2 = ((alpha - theta/2 + 90) % 180) - 90
    if theta == 90:
        return (N/(N+1))*temporal_celerity(k, a1, Ax, Ay, Bx, By, r, mu, delta) + np.sign(90-theta)*(1/(N+1))*temporal_celerity(k, a2, Ax, Ay, Bx, By, r, mu, delta)
    else:
        return (N/(N+1))*temporal_celerity(k, a1, Ax, Ay, Bx, By, r, mu, delta) + np.sign(90-theta)*(1/(N+1))*temporal_celerity(k, a2, Ax, Ay, Bx, By, r, mu, delta)


################################################################################
"""
theta: time series of wind direction (deg.)
N: time series of wind mass transport ratio
"""


def temporal_growth_rate_multi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N, axis=-1):
    amod = ((alpha - theta + 90) % 180) - 90
    Sigma = N*temporal_growth_rate(k, amod, Ax, Ay, Bx, By, r, mu, delta)
    return np.nansum(Sigma, axis=axis)


def temporal_celerity_multi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N, axis=-1):
    SIGN = np.sign(cosd(alpha)*cosd(theta) + sind(alpha)*sind(theta))
    amod = ((alpha - theta + 90) % 180) - 90
    Cel = SIGN*N*temporal_celerity(k, amod, Ax, Ay, Bx, By, r, mu, delta)
    return np.nansum(Cel, axis=axis)


def Get_most_unstable(Sigma, alpha, k):
    SigMax = np.amax(Sigma)
    Coord = np.argwhere(Sigma == SigMax)
    kmax = k[Coord[:, 1]]
    amax = alpha[Coord[:, 0]]
    return SigMax, kmax, amax
