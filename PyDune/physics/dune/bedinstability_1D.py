r"""
Dune flat bed instability under a unidirectional wind (1D -- no spanwise direction). Here, 'temporal' and 'spatial'
functions refer to the temporal and spatial version of the dune instability (see Gadal et al. 2020).

This theory is developped assuming a quadratic transport law of the form

.. math::

    q_{\rm sat}/Q_{*} = \omega \left[1 - (u_{\rm th}/u_{*})^{2}\right],

where :math:`Q_{*}` is a characteristic sand flux, :math:`\omega` a dimensional constant, :math:`u_{*}` the wind shear velocity and `u_{\rm th}` the threshold velocity for sediment transport.

In the following, all quantities are made non dimensional:

.. line-block::
    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q_{*}`.


References
----------
.. line-block::
    [1] Gadal, C., Narteau, C., Ewing, R. C., Gunn, A., Jerolmack, D., Andreotti, B., & Claudin, P. (2020).
Spatial and temporal development of incipient dunes. Geophysical Research Letters, 47(16), e2020GL088919.
"""

import numpy as np


# Temporal instability

def complex_pulsation(k, A, B):
    r"""Dispersion relation as the output of the temporal dune instability:

    .. math::

        \omega = k^{2}\frac{\mathcal{A} + i\mathcal{B}}{1 + i k}.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    A : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{A}` (in-phase).
    B : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{B}` (in-quadrature).

    Returns
    -------
    omega : scalar, numpy array
        complex pulsation.

    Notes
    -----
    Note that all quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*`.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(0.001, 1, 1000)
    >>> A, B = 3.5, 2
    >>> omega = complex_pulsation(k, A, B)

    """
    return k**2*(A + 1j*B)/(1 + 1j*k)


def temporal_growth_rate(k, A0, B0, mu, r):
    r""" Dune instability temporal growth rate - imaginary part of the complex pulsation where :math:`\mathcal{A} = \mathcal{A}_{0}`
    and :math:`\mathcal{B} = \mathcal{B}_{0} - 1/(r^{2}\mu)`, taking into account slope effects.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    A0 : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{A}_{0}` (in-phase).
    B0 : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{B}_{0}` (in-quadrature).
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`

    Returns
    -------
    sigma : scalar, numpy array
        temporal dune growth rate :math:`\sigma`.

    Notes
    -----
    Note that all quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*`.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(0.001, 1, 1000)
    >>> A0, B0, mu, r = 3.5, 2, 0.63, 2
    >>> sigma = temporal_growth_rate(k, A0, B0, mu, r)

    """
    A = A0
    B = B0 - (1/mu)*(1/r**2)
    return complex_pulsation(k, A, B).imag


def temporal_pulsation(k, A0, B0, mu, r):
    r""" Dune instability temporal pulsation - real part of the complex pulsation. Note
    that here, :math:`\mathcal{A} = \mathcal{A}_{0}` where :math:`\mathcal{B} = \mathcal{B}_{0} - 1/(r^{2}\mu)`,
    taking into account slope effects.

    Parameters
    ----------
    k : scalar, numpy array
        non dimensional wavenumber :math:`k`.
    A0 : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{A}_{0}` (in-phase).
    B0 : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{B}_{0}` (in-quadrature).
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`

    Returns
    -------
    omega_r : scalar, numpy array
        temporal dune pulsation :math:`\omega_{\rm r}`.

    Notes
    -----
    Note that all quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*`.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(0.001, 1, 1000)
    >>> A0, B0, mu, r = 3.5, 2, 0.63, 2
    >>> w = temporal_pulsation(k, A0, B0, mu, r)

    """
    A = A0
    B = B0 - (1/mu)*(1/r**2)
    return complex_pulsation(k, A, B).real


def temporal_velocity(k, A0, B0, mu, r):
    r""" Dune instability temporal velocity - real part of the complex pulsation divided by the wavenumber. Note
    that here, :math:`\mathcal{A} = \mathcal{A}_{0}` where :math:`\mathcal{B} = \mathcal{B}_{0} - 1/(r^{2}\mu)`,
    taking into account slope effects.

    Parameters
    ----------
    k : scalar, numpy array
        Non dimensional wavenumber :math:`k`.
    A0 : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{A}_{0}` (in-phase).
    B0 : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{B}_{0}` (in-quadrature).
    mu : scalar, numpy array
        Friction coefficient :math:`\mu`.
    r : scalar, numpy array
        Velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`

    Returns
    -------
    c : scalar, numpy array
        temporal dune elocity :math:`c`.

    Notes
    -----
    Note that all quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*`.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(0.001, 1, 1000)
    >>> A0, B0, mu, r = 3.5, 2, 0.63, 2
    >>> c = temporal_velocity(k, A0, B0, mu, r)

    """
    return temporal_pulsation(k, A0, B0, mu, r)/k

# Spatial instability


def complexe_wavenumer(w, A, B):
    r"""Dispersion relation as the output of the spatial dune instability:

    .. math:

        k_{\pm} = \frac{1}{2}\frac{i\omega \pm \sqrt{\Delta}}{\mathcal{A} + i\mathcal{B}},

    avec :math:`\Delta = \omega(4(\mathcal{A} + i\mathcal{B}) - \omega)`. Note that only the branch :math:`k_{+}` corresponds to spatially growing waves in the flow direction.

    Parameters
    ----------
    w : scalar, numpy array
        non dimensional pulsation :math:`\omega`.
    A : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{A}` (in-phase).
    B : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{B}` (in-quadrature).

    Returns
    -------
    kplus : scalar, numpy array
        :math:`k_{+}` branch.
    kminus : scalar, numpy array
        :math:`k_{-}` branch.

    Notes
    -----
    Note that all quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*}`.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.linspace(0.001, 1, 1000)
    >>> A, B = 3.5, 2
    >>> kplus, kminus = complexe_wavenumer(w, A, B)

    """
    delta = w*(4*(A+1j*B) - w)
    k_plus = 0.5*(1j*w + np.sqrt(delta))/(A + 1j*B)
    k_minus = 0.5*(1j*w - np.sqrt(delta))/(A + 1j*B)
    return k_plus, k_minus


def spatial_growth_rate(w, A0, B0, mu, r):
    r""" Dune instability spatial growth rate - imaginary part of the complex wavenumber :math:`k_{+}`. Note
    that here, :math:`\mathcal{A} = \mathcal{A}_{0}` where :math:`\mathcal{B} = \mathcal{B}_{0} - 1/(r^{2}\mu)`,
    taking into account slope effects.

    Parameters
    ----------
    w : scalar, numpy array
        non dimensional pulsation :math:`\omega`.
    A0 : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{A}_{0}` (in-phase).
    B0 : scalar, numpy array
        hydrodynamic coefficient :math:`\mathcal{B}_{0}` (in-quadrature).
    mu : scalar, numpy array
        friction coefficient :math:`\mu`.
    r : scalar, numpy array
        velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`

    Returns
    -------
    sigma_s : scalar, numpy array
        spatial dune growth rate :math:`\sigma_{\rm s}`.

    Notes
    -----
    Note that all quantities are made non dimensional:

        - length scales by the saturation length :math:`L_{\rm sat}`.
        - time scales by :math:`L_{\rm sat}^{2}/Q_{*}`.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.linspace(0.001, 1, 1000)
    >>> A0, B0, mu, r = 3.5, 2, 0.63, 2
    >>> sigma_s = spatial_growth_rate(w, A0, B0, mu, r)

    """
    A = A0
    B = B0 - (1/mu)*(1/r**2)
    return -complexe_wavenumer(w, A, B)[0].imag


def spatial_wavenumber(w, A0, B0, mu, r):
    r""" Dune instability spatial wavenumber - real part of the complex wavenumber :math:`k_{+}`. Note
    that here, :math:`\mathcal{A} = \mathcal{A}_{0}` where :math:`\mathcal{B} = \mathcal{B}_{0} - 1/(r^{2}\mu)`,
    taking into account slope effects.

    Parameters
    ----------
    w : scalar, numpy array
        Non dimensional pulsation :math:`\omega`.
    A0 : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{A}_{0}` (in-phase).
    B0 : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{B}_{0}` (in-quadrature).
    mu : scalar, numpy array
        Friction coefficient :math:`\mu`.
    r : scalar, numpy array
        Velocity ratio :math:`u_{*}/u_{\rm d} = \sqrt{\theta/\theta_{d}}`

    Returns
    -------
    scalar, numpy array
        Spatial dune wavenumber :math:`k` calculated elementwise.

    Notes
    -----
    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.linspace(0.001, 1, 1000)
    >>> A0, B0, mu, r = 3.5, 2, 0.63, 2
    >>> k = spatial_wavenumber(w, A0, B0, mu, r)

    """
    A = A0
    B = B0 - (1/mu)*(1/r**2)
    return complexe_wavenumer(w, A, B)[0].real


if __name__ == "__main__":
    import doctest
    doctest.testmod()
