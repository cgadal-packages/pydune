r"""
Dune flat bed instability taking into account two spatial direction (x and y), under unidirectional, bidirectional and multidirectional wind regimes. Here, only the temporal instability is derived,
following Gadal et al. 2019.

.. math::

    q_{\rm sat}/Q_{*} = Q\left[1 - (u_{\rm th}/u_{*})^{2}\right],

where :math:`Q_{*}` is a characteristic sand flux, :math:`u_{*}` the wind shear velocity and `u_{\rm th}` the threshold velocity for sediment transport.


In the following, all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q_{*}`.


References
----------
[1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
[2] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.
"""


import numpy as np
from PyDune.math import cosd, sind


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
#         direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`. It is also the dune orientation measured with respect to the perpendicular to the wind direction .
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
    r"""Temporal growth rate of sinusoidal periodic dunes of wavenumber :math:`k`
    and orientation :math:`\alpha` induced by a unidirectional constant wind through the linear dune instability.
    It is calculated elementwise.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`. It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Ay : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Bx : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    By : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        cross-stream diffusion coefficient. Set to 0 if you want to recover the
        exact results of Andreotti et al. 2012, Gadal et al. 2019. See the `PhD thesis "Dune emergence in multidirectional
        wind regimes" by Cyril Gadal <https://cgadal.github.io/files/ThesisCyrilGadal.pdf>`_, section 3.5.1, for additonal details.

    Returns
    -------
    sigma : scalar, numpy array
        temporal dune growth rate :math:`\sigma`.

    Notes
    --------
    All quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*}`.


    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0.001, 1, 300), np.linspace(0, 360, 300)
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

    References
    ----------
    [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
    [2] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.
    """

    ax = Ax(k, alpha)
    bx = Bx(k, alpha) - cosd(alpha)*(1/mu)*(1/r**2)
    ay = (1 - 1/r**2)*Ay(k, alpha) - delta*k*sind(alpha)*bx
    by = (1 - 1/r**2)*(By(k, alpha) - sind(alpha)*(1/mu)*(1/r)) - delta*k*sind(alpha)*ax
    return (k**2/(1 + (k*cosd(alpha))**2))*(bx*cosd(alpha) + by*sind(alpha) - k*cosd(alpha)*(ax*cosd(alpha) + ay*sind(alpha)))
    # return complex_pulsation(k, alpha, ax, bx, ay, by).imag


def temporal_pulsation(k, alpha, Ax, Ay, Bx, By, r, mu, delta):
    r"""Temporal pulsation :math:`\omega` of sinusoidal periodic dunes of wavenumber :math:`k`
    and orientation :math:`\alpha` induced by a unidirectional constant wind through the linear dune instability. It is calculated elementwise.

    Parameters
    ----------
    k : scalar, numpy array
        Non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`. It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Ay : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Bx : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    By : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        cross-stream diffusion coefficient. Set to 0 if you want to recover the
        exact results of Andreotti et al. 2012, Gadal et al. 2019. See the `PhD thesis "Dune emergence in multidirectional
        wind regimes" by Cyril Gadal <https://cgadal.github.io/files/ThesisCyrilGadal.pdf>`_, section 3.5.1, for additonal details.

    Returns
    -------
    omega : scalar, numpy array
        temporal dune growth pulsation :math:`\omega` calculated elementwise.

    Notes
    --------
    All quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*}`.


    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0.001, 1, 300), np.linspace(0, 360, 300)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> # Chosing a form for the hydrodynamic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> OMEGA = temporal_pulsation(K, ALPHA, Ax, Ay, Bx, By, r, mu, delta)

    References
    ----------
    [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
    [2] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.
    """
    ax = Ax(k, alpha)
    bx = Bx(k, alpha) - cosd(alpha)*(1/mu)*(1/r**2)
    ay = (1 - 1/r**2)*Ay(k, alpha) - delta*k*sind(alpha)*bx
    by = (1 - 1/r**2)*(By(k, alpha) - sind(alpha)*(1/mu)*(1/r)) - delta*k*sind(alpha)*ax
    # return complex_pulsation(k, alpha, ax, bx, ay, by).real
    return (k**2/(1 + (k*cosd(alpha))**2))*(ax*cosd(alpha) + ay*sind(alpha) + k*cosd(alpha)*(bx*cosd(alpha) + by*sind(alpha)))


def temporal_celerity(k, alpha, Ax, Ay, Bx, By, r, mu, delta):
    r"""Temporal celerity :math:`c = \omega/k` of sinusoidal periodic dunes of wavenumber :math:`k`
    and orientation :math:`\alpha` induced by a unidirectional constant wind through the linear dune instability.
    It is calculated elementwise.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`.
        It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Ay : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Bx : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    By : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        cross-stream diffusion coefficient. Set to 0 if you want to recover the
        exact results of Andreotti et al. 2012, Gadal et al. 2019. See the `PhD thesis "Dune emergence in multidirectional
        wind regimes" by Cyril Gadal <https://cgadal.github.io/files/ThesisCyrilGadal.pdf>`_, section 3.5.1, for additonal details.

    Returns
    -------
    c : scalar, numpy array
        temporal dune celerity :math:`c` calculated elementwise.

    Notes
    --------
    All quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*}`.


    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0.001, 1, 300), np.linspace(0, 360, 300)
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

    References
    ----------
    [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
    [2] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.
    """
    return temporal_pulsation(k, alpha, Ax, Ay, Bx, By, r, mu, delta)/k
    # return (k/(1 + (k*cosd(alpha))**2))*(ax*cosd(alpha) + ay*sind(alpha) + k*cosd(alpha)*(bx*cosd(alpha) + by*sind(alpha)))


################################################################################

def growth_rate_bidi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N):
    r"""Temporal growth rate :math:`\sigma` of sinusoidal periodic dunes of wavenumber :math:`k`
    and orientation :math:`\alpha` induced by a bidirectional wind whose orientation
    alternates regularly between :math:`-\theta/2` and :math:`+\theta/2`. It is calculated elementwise.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`.
        It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Ay : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Bx : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    By : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`. Here, it is the same for both winds.
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        cross-stream diffusion coefficient. Set to 0 if you want to recover the
        exact results of Andreotti et al. 2012, Gadal et al. 2019. See the `PhD thesis "Dune emergence in multidirectional
        wind regimes" by Cyril Gadal <https://cgadal.github.io/files/ThesisCyrilGadal.pdf>`_, section 3.5.1, for additonal details.
    theta : float
        angle between the two wind direction, in degree.
    N : float
        mass transport ratio between the two wind directions

    Returns
    -------
    sigma : scalar, numpy array
        temporal growth rate.

    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0.001, 1, 300), np.linspace(0, 360, 300)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> theta, N = 120, 3
    >>> # Chosing a form for the hydrodynamic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> SIGMA = growth_rate_bidi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N)

    References
    ----------
    [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
    """
    # changement de ref + prise en compte seulement de la direction et pas du sens de l'onde
    a1 = ((alpha + theta/2 + 90) % 180) - 90
    a2 = ((alpha - theta/2 + 90) % 180) - 90
    return (N/(N+1))*temporal_growth_rate(k, a1, Ax, Ay, Bx, By, r, mu, delta) + (1/(N+1))*temporal_growth_rate(k, a2, Ax, Ay, Bx, By, r, mu, delta)


def celerity_bidi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N):
    r"""Celerity :math:`\sigma` of sinusoidal periodic dunes of wavenumber :math:`k`
    and orientation :math:`\alpha` induced by a bidirectional wind whose orientation
    alternates regularly between :math:`-\theta/2` and :math:`+\theta/2`. It is calculated elementwise.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`.
        It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}` (in-phase).
        A vectorized function taking `k` as first argument and `alpha` as second one.
    Ay : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}` (in-phase).
        A vectorized function taking `k` as first argument and `alpha` as second one.
    Bx : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}` (in-qudrature).
        A vectorized function taking `k` as first argument and `alpha` as second one.
    By : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}` (in-qudrature).
        A vectorized function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`. Here, it is the same for both winds.
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        cross-stream diffusion coefficient. Set to 0 if you want to recover the
        exact results of Andreotti et al. 2012, Gadal et al. 2019. See the `PhD thesis "Dune emergence in multidirectional
        wind regimes" by Cyril Gadal <https://cgadal.github.io/files/ThesisCyrilGadal.pdf>`_, section 3.5.1, for additonal details.
    theta : float
        angle between the two wind direction, in degree.
    N : float
        mass transport ratio between the two wind directions

    Returns
    -------
    c : scalar, numpy array
        celerity

    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0.001, 1, 300), np.linspace(0, 360, 300)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> theta, N = 120, 3
    >>> # Chosing a form for the hydrodynamic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> CELERITY = celerity_bidi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N)

    References
    ----------
    [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
    """
    a1 = ((alpha + theta/2 + 90) % 180) - 90
    a2 = ((alpha - theta/2 + 90) % 180) - 90
    if theta == 90:
        return (N/(N+1))*temporal_celerity(k, a1, Ax, Ay, Bx, By, r, mu, delta) + np.sign(90-theta)*(1/(N+1))*temporal_celerity(k, a2, Ax, Ay, Bx, By, r, mu, delta)
    else:
        return (N/(N+1))*temporal_celerity(k, a1, Ax, Ay, Bx, By, r, mu, delta) + np.sign(90-theta)*(1/(N+1))*temporal_celerity(k, a2, Ax, Ay, Bx, By, r, mu, delta)


################################################################################

def temporal_growth_rate_multi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N, axis=-1):
    r"""Temporal growth rate :math:`\sigma` of sinusoidal periodic dunes of wavenumber :math:`k`
    and orientation :math:`\alpha` induced by succesion of winds of various orientations and strengths.
    It is calculated elementwise, and then averaged along `axis`, which should correspond to the time axis.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`.
        It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}`
        (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Ay : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}`
        (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Bx : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}`
        (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    By : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}`
        (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`. Here,
        it is the same for both winds.
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        cross-stream diffusion coefficient. Set to 0 if you want to recover the
        exact results of Andreotti et al. 2012, Gadal et al. 2019. See the `PhD thesis "Dune emergence in multidirectional
        wind regimes" by Cyril Gadal <https://cgadal.github.io/files/ThesisCyrilGadal.pdf>`_, section 3.5.1, for additonal details.
    theta : numpy array
        wind directions in degrees.
    N : numpy array
        mass transport ratios between the two wind directions.
    axis : int
        axis along which the sum is performed (the default is -1).

    Returns
    -------
    sigma : scalar, numpy array
        temporal growth rate.

    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0.001, 1, 300), np.linspace(0, 360, 300)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> theta, N = 360*np.random.random((100, )), np.random.random((100, ))
    >>> # Chosing a form for the hydrodynamic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> SIGMA = temporal_growth_rate_multi(K[..., None], ALPHA[..., None], Ax, Ay, Bx, By, r, mu, delta, theta[None, None, :], N[None, None, :])

    References
    ----------
    [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
    """
    amod = ((alpha - theta + 90) % 180) - 90
    Sigma = N*temporal_growth_rate(k, amod, Ax, Ay, Bx, By, r, mu, delta)
    return np.nansum(Sigma, axis=axis)


def temporal_celerity_multi(k, alpha, Ax, Ay, Bx, By, r, mu, delta, theta, N, axis=-1):
    r"""Temporal celerity :math:`c` of sinusoidal periodic dunes of wavenumber :math:`k`
    and orientation :math:`\alpha` induced by succesion of winds of various orientations and strengths.
    It is calculated elementwise, and then averaged along `axis`, which should correspond to the time axis.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    alpha : scalar, numpy array
        direction :math:`alpha` in degree of the wavevector :math:`\boldsymbol{k}`.
        It is also the dune orientation measured with respect to the perpendicular to the wind direction .
    Ax : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{A}_{x}`
        (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Ay : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{A}_{y}`
        (in-phase). A vectorized function taking `k` as first argument and `alpha` as second one.
    Bx : function
        hydrodynamic coefficient in the :math:`x`-direction :math:`\mathcal{B}_{x}`
        (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    By : function
        hydrodynamic coefficient in the :math:`y`-direction :math:`\mathcal{B}_{y}`
        (in-qudrature). A vectorized function taking `k` as first argument and `alpha` as second one.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`. Here,
        it is the same for both winds.
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    delta : scalar, numpy array
        cross-stream diffusion coefficient. Set to 0 if you want to recover the
        exact results of Andreotti et al. 2012, Gadal et al. 2019. See the `PhD thesis "Dune emergence in multidirectional
        wind regimes" by Cyril Gadal <https://cgadal.github.io/files/ThesisCyrilGadal.pdf>`_, section 3.5.1, for additonal details.
    theta : numpy array
        wind directions in degrees.
    N : numpy array
        mass transport ratios between the two wind directions.
    axis : int
        axis along which the sum is performed (the default is -1).

    Returns
    -------
    c : scalar, numpy array
        temporal growth rate.

    Examples
    --------
    >>> import numpy as np
    >>> # Range of parameter exploration
    >>> k, alpha = np.linspace(0.001, 1, 300), np.linspace(0, 360, 300)
    >>> K, ALPHA = np.meshgrid(k, alpha)
    >>> # Defining all parameters
    >>> A0, B0 = 3.5, 2
    >>> r, mu, delta = 1.5, 0.6, 0
    >>> theta, N = 360*np.random.random((100, )), np.random.random((100, ))
    >>> # Chosing a form for the hydrodynamic coefficients
    >>> Ax = lambda k, alpha: A0*np.cos(2*np.pi*alpha/180)**2
    >>> Bx = lambda k, alpha: B0*np.cos(2*np.pi*alpha/180)**2
    >>> Ay = lambda k, alpha: 0.5*A0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> By = lambda k, alpha: 0.5*B0*np.cos(2*np.pi*alpha/180)*np.sin(2*np.pi*alpha/180)
    >>> # Calculating growth function
    >>> C = temporal_celerity_multi(K[..., None], ALPHA[..., None], Ax, Ay, Bx, By, r, mu, delta, theta[None, None, :], N[None, None, :])

    References
    ----------
    [1] Gadal, C., Narteau, C., Du Pont, S. C., Rozier, O., & Claudin, P. (2019). Incipient bedforms in a bidirectional wind regime. Journal of Fluid Mechanics, 862, 490-516.
    """
    SIGN = np.sign(cosd(alpha)*cosd(theta) + sind(alpha)*sind(theta))
    amod = ((alpha - theta + 90) % 180) - 90
    Cel = SIGN*N*temporal_celerity(k, amod, Ax, Ay, Bx, By, r, mu, delta)
    return np.nansum(Cel, axis=axis)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
