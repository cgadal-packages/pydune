r"""
Dune growth mechanism theory developped by Courrech du Pont et al. 2014.



References
----------
.. line-block::
    [1] Courrech du Pont, S., Narteau, C., & Gao, X. (2014). Two modes for dune orientation. Geology, 42(9), 743-746.
"""

import numpy as np
from PyDune.math import vector_average, cosd, sind


def flux_at_crest(alpha, theta, Q0, gamma=1.6):
    r"""Compute the sand flux at the dune crest:

    .. math::

        Q_{\rm crest} = Q_{0}\left[1 + \gamma\sin\vert\theta-\alpha\vert\right]

    Parameters
    ----------
    alpha : scalar, numpy array
        dune orientation :math:`\alpha`.
    theta : scalar, numpy array
        flux orientation :math:`\theta` in degrees.
    Q0 : scalar, numpy array
        flux at the bottom of the dune :math:`Q_{0}`.
    gamma : scalar, numpy array
        flux-up ratio :math:`\gamma` (the default is 1.6).

    Returns
    -------
    scalar, numpy array
        flux at the dune crest :math:`Q_{\rm crest}`

    Examples
    --------
    >>> import numpy as np
    >>> alpha = 10
    >>> theta = np.random.random((1000,))*360
    >>> Q0 = np.random.random((1000,))*50
    >>> Qcrest = flux_at_crest(alpha, theta, Q0)

    """
    return theta, Q0*(1 + gamma*np.abs(sind(theta - alpha)))


def resultant_flux_at_crest(alpha, theta, Q0, gamma=1.6, **kwargs):
    r"""Compute the resultant flux (i.e vectorial average) of the sand flux at the dune crest.

    Parameters
    ----------
    alpha : scalar, numpy array
        dune orientation :math:`\alpha`.
    theta : scalar, numpy array
        flux orientation :math:`\theta` in degrees.
    Q0 : scalar, numpy array
        flux at the bottom of the dune :math:`Q_{0}`.
    gamma : scalar, numpy array
        flux-up ratio :math:`\gamma` (the default is 1.6).
    **kwargs :
        `kwargs` are passed to :func:`vector_average <PyDune.math.vector_average>`.

    Returns
    -------
    angle : array_like
        the counterclockwise angle of the resultant sand flux at the crest in the range [-180, 180], i.e RDD at the dune crest.
    norm : array_like
        norm of the resultant the resultant sand flux at the crest, i.e RDP at the dune crest.

    Examples
    --------
    >>> import numpy as np
    >>> alpha = 10
    >>> theta = np.random.random((1000,))*360
    >>> Q0 = np.random.random((1000,))*50
    >>> Qcrest = resultant_flux_at_crest(alpha, theta, Q0)

    """
    th_crest, N_crest = flux_at_crest(alpha, theta, Q0, gamma=gamma)
    return vector_average(th_crest, N_crest, **kwargs)


def resultant_flux_perp_crest_at_crest(alpha, theta, Q0, gamma=1.6, axis=-1):
    r"""Compute the component of the resultant flux (i.e vectorial average) at the crest perpendicular to the dune crest.

    Parameters
    ----------
    alpha : scalar, numpy array
        dune orientation :math:`\alpha`.
    theta : scalar, numpy array
        flux orientation :math:`\theta` in degrees.
    Q0 : scalar, numpy array
        flux at the bottom of the dune :math:`Q_{0}`.
    gamma : scalar, numpy array
        flux-up ratio :math:`\gamma` (the default is 1.6).
    axis : int
        axis over wich the average is done (the default is -1).

    Returns
    -------
    scalar, numpy array
        component of the resultant flux (i.e vectorial average) at the crest perpendicular to the dune crest.

    Examples
    --------
    >>> import numpy as np
    >>> alpha = 10
    >>> theta = np.random.random((1000,))*360
    >>> Q0 = np.random.random((1000,))*50
    >>> Qcrest_perp = resultant_flux_perp_crest_at_crest(alpha, theta, Q0)

    """
    RDD, RDP = resultant_flux_at_crest(alpha, theta, Q0, gamma=gamma, axis=axis)
    alpha_squeezed = np.squeeze(alpha, axis=axis)
    return RDP*(cosd(alpha_squeezed + 90)*cosd(RDD) + sind(alpha_squeezed + 90)*sind(RDD))


def resultant_flux_aligned_crest_at_crest(alpha, theta, Q0, gamma=1.6, axis=-1):
    r"""Compute the component of the resultant flux (i.e vectorial average) at the crest aligned with the dune crest.

    Parameters
    ----------
    alpha : scalar, numpy array
        dune orientation :math:`\alpha`.
    theta : scalar, numpy array
        flux orientation :math:`\theta` in degrees.
    Q0 : scalar, numpy array
        flux at the bottom of the dune :math:`Q_{0}`.
    gamma : scalar, numpy array
        flux-up ratio :math:`\gamma` (the default is 1.6).
    axis : int
        axis over wich the average is done (the default is -1).

    Returns
    -------
    scalar, numpy array
        component of the resultant flux (i.e vectorial average) at the crest perpendicular to the dune crest.

    Examples
    --------
    >>> import numpy as np
    >>> alpha = 10
    >>> theta = np.random.random((1000,))*360
    >>> Q0 = np.random.random((1000,))*50
    >>> Qcrest_perp = resultant_flux_perp_crest_at_crest(alpha, theta, Q0)

    """
    RDD, RDP = resultant_flux_at_crest(alpha, theta, Q0, gamma=gamma, axis=axis)
    alpha_squeezed = np.squeeze(alpha, axis=axis)
    return RDP*(cosd(alpha_squeezed)*cosd(RDD) + sind(alpha_squeezed)*sind(RDD))


def elongation_direction(theta, Q0, gamma=1.6, alpha_bins=np.linspace(0, 360, 361),
                         axis=-1, **kwargs):
    r"""Calculate the elongation direction following the model of Courrech du Pont et al. 2014.

    Parameters
    ----------
    theta : numpy array
        sand flux orientation :math:`\theta` in degrees.
    Q0 : numpy array
        sand flux at the bottom of the dune :math:`Q_{0}`.
    gamma : numpy array
        flux-up ratio :math:`\gamma` (the default is 1.6).
    alpha_bins : numpy array
        bins in dune orientation used to calculate the resultant flux at the crest (the default is np.linspace(0, 360, 361)).
    **kwargs :
        `kwargs` are optional parameters passed to :func:`resultant_flux_perp_crest_at_crest <PyDune.courrechdupont2014.resultant_flux_perp_crest_at_crest>`.

    Returns
    -------
    numpy array
        the elongation direction.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.random.random((1000,))*360
    >>> Q0 = np.random.random((1000,))*50
    >>> Alpha_F = elongation_direction(theta, Q0)

    References
    ----------
    [1] Courrech du Pont, S., Narteau, C., & Gao, X. (2014). Two modes for dune orientation. Geology, 42(9), 743-746.
    """

    # Matching dimensions
    alpha_expanded = np.expand_dims(alpha_bins, tuple(np.arange(1, len(theta.shape) + 1)))
    th_expanded, N_expanded, gamma_expended = np.expand_dims(theta, 0), np.expand_dims(Q0, 0), np.expand_dims(gamma, 0)
    #
    Alpha_F = alpha_bins[np.argmin(np.abs(
        resultant_flux_perp_crest_at_crest(alpha_expanded, th_expanded, N_expanded,
                                           gamma=gamma_expended, axis=axis, **kwargs)
                                           ), axis=0)]
    del alpha_expanded, th_expanded, N_expanded
    RDD, _ = vector_average(theta, Q0)  # wind resultant angle
    #
    prod = cosd(Alpha_F)*cosd(RDD) + sind(Alpha_F)*sind(RDD)  # check that the orientation goes in the right drirection
    del RDD
    return np.mod(np.where(prod > 0, Alpha_F, Alpha_F + 180), 360)


def growth_rate(alpha, theta, Q0, gamma=1.6, axis=-1, capture_rate=1):
    r"""Calculate the dune growth rate using the model of Courrech du Pont et al. 2014.

    Parameters
    ----------
    alpha : scalar, numpy array
        dune orientation :math:`\alpha`.
    theta : scalar, numpy array
        flux orientation :math:`\theta` in degrees.
    Q0 : scalar, numpy array
        flux at the bottom of the dune :math:`Q_{0}`.
    gamma : scalar, numpy array
        flux-up ratio :math:`\gamma` (the default is 1.6).
    axis : int
        axis over wich the average is done (the default is -1).
    capture_rate : function, scalar, numpy array
        capture rate of the avalanche slope. Can either be a scalar, a numpy array with dimensions corresponding to `alpha`, `theta` and `Q0`,
         or function taking as argument `alpha`, `theta` and `Q0`, in this order (the default is 1).

    Returns
    -------
    scalar, numpy array
        dune growth rate.

    Examples
    --------
    >>> import numpy as np
    >>> alpha = 10
    >>> theta = np.random.random((1000,))*360
    >>> Q0 = np.random.random((1000,))*50
    >>> Qcrest_perp = growth_rate(alpha, theta, Q0)

    References
    ----------
    [1] Courrech du Pont, S., Narteau, C., & Gao, X. (2014). Two modes for dune orientation. Geology, 42(9), 743-746.
    """

    if callable(capture_rate):
        CR = capture_rate(alpha, theta, Q0)
    else:
        CR = capture_rate

    return np.squeeze(np.sum(CR*Q0*(np.abs(sind(theta - alpha)) + gamma*sind(theta-alpha)**2), axis=axis))


def MGBNT_orientation(theta, Q0, gamma=1.6, alpha_bins=np.linspace(0, 360, 361), **kwargs):
    r"""Calculate the dune orientation growing from the 'maximum gross bedform normal-transport' rule
    following the model of Courrech du Pont et al. 2014, also called in the later 'bed instability'.

    Parameters
    ----------
    theta : scalar, numpy array
        flux orientation :math:`\theta` in degrees.
    Q0 : scalar, numpy array
        flux at the bottom of the dune :math:`Q_{0}`.
    gamma : scalar, numpy array
        flux-up ratio :math:`\gamma` (the default is 1.6).
    alpha_bins : numpy array
        bins in dune orientation used to calculate the resultant flux at the crest (the default is np.linspace(0, 360, 361)).
    **kwargs :
        `kwargs` are optional parameters passed to :func:`growth_rate <python_codes.courrechdupont2014.growth_rate>`.

    Returns
    -------
    scalar, numpy array
        dune orientation.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.random.random((1000,))*360
    >>> Q0 = np.random.random((1000,))*50
    >>> Alpha_F = bed_instability_orientation(theta, Q0)

    References
    ----------
    [1] Courrech du Pont, S., Narteau, C., & Gao, X. (2014). Two modes for dune orientation. Geology, 42(9), 743-746.
    """

    # Matching dimensions
    alpha_expanded = np.expand_dims(alpha_bins, tuple(np.arange(1, len(theta.shape) + 1)))
    th_expanded, N_expanded, gamma_expended = np.expand_dims(theta, 0), np.expand_dims(Q0, 0), np.expand_dims(gamma, 0)
    #
    G_rate = growth_rate(alpha_expanded, th_expanded, N_expanded, gamma=gamma_expended, **kwargs)
    return np.mod(alpha_bins[G_rate.argmax(0)], 180)
