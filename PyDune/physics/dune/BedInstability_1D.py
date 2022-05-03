# @Author: gadal
# @Date:   2021-02-11T15:32:13+01:00
# @Email:  gadal@ipgp.fr
# @Last modified by:   gadal
# @Last modified time: 2021-03-07T19:06:41+01:00

r"""
Dune flat bed instability under a unidirectional wind (1D). Here, 'Temporal' and 'Spatial'
functions refer to the temporal and spatial version of the dune instability (see Gadal et al. 2020).

This theory is developped assuming a quadratic transport law of the form

.. math::

    q_{\rm sat} = Q\left[1 - (u_{\rm th}/u_{*})^{2}\right] .

Here, all quantities are made non dimensional:

- length scales by the saturation length :math:`L_{\rm sat}`.
- time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.


References
----------
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
        Non dimensional wavenumber :math:`k`.
    A : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{A}` (in-phase).
    B : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{B}` (in-quadrature).

    Returns
    -------
    scalar, numpy array
        Complex pulsation calculated elementwise.

    Notes
    --------
    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(0, 1, 1000)
    >>> A, B = 3.5, 2
    >>> omega = complex_pulsation(k, A, B)

    """
    return k**2*(A + 1j*B)/(1 + 1j*k)


def temporal_growth_rate(k, A0, B0, mu, r):
    r""" Dune instability temporal growth rate - imaginary part of the complex pulsation. Note
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
    >>> k = np.linspace(0, 1, 1000)
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
    scalar, numpy array
        Temporal dune pulsation :math:`c` calculated elementwise.

    Notes
    --------
    The dune migration velocity is then :math:`\omega/k`.

    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(0, 1, 1000)
    >>> A0, B0, mu, r = 3.5, 2, 0.63, 2
    >>> w = temporal_pulsation(k, A0, B0, mu, r)
    >>> c = w/k  # Dune migration velocity

    """
    A = A0
    B = B0 - (1/mu)*(1/r**2)
    return complex_pulsation(k, A, B).real/k


# Spatial instability

def complexe_wavenumer(w, A, B):
    r"""Dispersion relation as the output of the spatial dune instability:

    .. math:

        k_{\pm} = \frac{1}{2}\frac{i\omega \pm \sqrt{\Delta}}{\mathcal{A} + i\mathcal{B}},

    avec :math:`\Delta = \omega(4(\mathcal{A} + i\mathcal{B}) - \omega) `. Note that only the branch :math:`k_{+}` corresponds
    to spatially growing waves in the flow direction.

    Parameters
    ----------
    w : scalar, numpy array
        Non dimensional pulsation :math:`\omega`.
    A : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{A}` (in-phase).
    B : scalar, numpy array
        Hydrodynamic coefficient :math:`\mathcal{B}` (in-quadrature).

    Returns
    -------
    list of scalar, numpy array
        The two branches :math:`k_{+}` and :math:`k_{-}` calculated elementwise.

    Notes
    --------
    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.linspace(0, 1, 1000)
    >>> A, B = 3.5, 2
    >>> kplus, kmoins = complexe_wavenumer(w, A, B)

    """
    delta = w*(4*(A+1j*B) - w)
    k_plus = 0.5*(1j*w + np.sqrt(delta))/(A + 1j*B)
    k_moins = 0.5*(1j*w - np.sqrt(delta))/(A + 1j*B)
    return k_plus, k_moins


def spatial_growth_rate(w, A0, B0, mu, r):
    r""" Dune instability spatial growth rate - imaginary part of the complex wavenumber :math:`k_{+}`. Note
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
        Spatial dune growth rate :math:`\sigma_{\rm s}` calculated elementwise.

    Notes
    --------
    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.linspace(0, 1, 1000)
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
    --------
    Note that all quantities are made non dimensional:

    - length scales by the saturation length :math:`L_{\rm sat}`.
    - time scales by :math:`L_{\rm sat}^{2}/Q`, where :math:`Q` is the characteristic flux.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.linspace(0, 1, 1000)
    >>> A0, B0, mu, r = 3.5, 2, 0.63, 2
    >>> k = spatial_wavenumber(w, A0, B0, mu, r)

    """
    A = A0
    B = B0 - (1/mu)*(1/r**2)
    return complexe_wavenumer(w, A, B)[0].real


if __name__ == "__main__":
    import doctest
    doctest.testmod()
