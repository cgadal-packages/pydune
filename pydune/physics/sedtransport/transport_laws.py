r"""
Sediment transport laws. Here, sediment fluxes are made non dimensional
by the characteristic flux :math:`Q = \sqrt{\displaystyle\frac{(\rho_{\rm p} - \rho_{\rm f}) g d}{\rho_{\rm f}}}d`.

"""

import numpy as np


def quadratic_transport_law(theta, theta_d, omega):
    r"""Quadratic transport law :math:`q_{\rm sat}/Q = \Omega \sqrt{\theta_{\rm th}}(\theta - \theta_{\rm th})`, from Duràn et al. 2011.

    Parameters
    ----------
    theta : scalar, numpy array
        Shield number.
    theta_d : scalar, numpy array
        Threshold Shield number.
    omega : scalar, numpy array
        Prefactor of the transport law.

    Returns
    -------
    scalar, numpy array
        Sediment fluxes calculated elementwise using the quadratic transport law.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.random.random((2000, ))
    >>> theta_d, omega = 0.0053, 7.8
    >>> qsat = quadratic_transport_law(theta, theta_d, omega)

    References
    ----------
    .. line-block::
        [1] Durán, O., Claudin, P., & Andreotti, B. (2011). On aeolian transport: Grain-scale interactions,
    dynamical mechanisms and scaling laws. Aeolian Research, 3(3), 243-270.

    """
    return np.piecewise(theta, [theta > theta_d, theta <= theta_d],
                        [lambda theta: omega*np.sqrt(theta_d)*(theta - theta_d), lambda theta: 0])


def cubic_transport_law(theta, theta_d, omega):
    r"""Cubic transport law :math:`q_{\rm sat}/Q = \Omega \sqrt{\theta}(\theta - \theta_{\rm th})`, from Duràn et al. 2011.

    Parameters
    ----------
    theta : scalar, numpy array
        Shield number.
    theta_d : scalar, numpy array
        Threshold Shield number.
    omega : scalar, numpy array
        Prefactor of the transport law.

    Returns
    -------
    scalar, numpy array
        Sediment fluxes calculated elementwise using the cubic transport law.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.random.random((2000, ))
    >>> theta_d, omega = 0.0053, 7.8
    >>> qsat = cubic_transport_law(theta, theta_d, omega)

    References
    ----------
    .. line-block::
        [1] Durán, O., Claudin, P., & Andreotti, B. (2011). On aeolian transport: Grain-scale interactions,
    dynamical mechanisms and scaling laws. Aeolian Research, 3(3), 243-270.

    """
    return np.piecewise(theta, [theta > theta_d, theta <= theta_d],
                        [lambda theta: omega*np.sqrt(theta)*(theta - theta_d), lambda theta: 0])


def quartic_transport_law(theta, theta_d, Kappa=0.4, mu=0.63, cm=1.7):
    r"""Quartic transport law :math:`q_{\rm sat}/Q = \frac{2\sqrt{\theta_{\rm th}}}{\kappa\mu}(\theta - \theta_{\rm th})\left[1 + \frac{C_{\rm M}}{\mu}(\theta - \theta_{\rm th})\right]` from Pähtz et al. 2020.

    Parameters
    ----------
    theta : scalar, numpy array
        Shield number.
    theta_d : scalar, numpy array
        Threshold Shield number.
    Kappa : scalar, numpy array
        von Kármán constant (the default is 0.4).
    mu : scalar, numpy array
        Friction coefficient (the default is 0.63).
    cm : scalar, numpy array
        Transport law coefficient (the default is 1.7).

    Returns
    -------
    scalar, numpy array
        Sediment fluxes calculated elementwise using the quartic transport law.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.random.random((2000, ))
    >>> theta_d = 0.0035
    >>> qsat = quartic_transport_law(theta, theta_d)

    References
    ----------
    .. line-block::
        [1] Pähtz, T., & Durán, O. (2020). Unification of aeolian and fluvial sediment transport rate from granular physics. Physical review letters, 124(16), 168001.

    """
    return np.piecewise(theta, [theta > theta_d, theta <= theta_d],
                        [lambda theta: (2/(Kappa*mu))*np.sqrt(theta_d)*(theta - theta_d)*(1 + (cm/mu)*(theta - theta_d)), lambda theta: 0])
