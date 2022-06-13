import numpy as np
from PyDune.math import cosd, sind

# %%
# Geometrical model
# -----------------
# Geometrical model for the scaling of the coefficient as the function of the orientation of the bottom perturbation


def Ax(alpha, A0):
    r"""Calculate the hydrodynamic coefficient :math:`\mathcal{A}_{x}` using the geometrical model:

    .. math::

        \mathcal{A}_{x} = \mathcal{A}_{0}\cos^{2}\alpha.

    Parameters
    ----------
    alpha : array, scalar
        Dune orientation with respect to the perpendicular to the flow direction (in degree).
    A0 : array, scalar
        value of the hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.

    Returns
    -------
    array, scalar
         the hydrodynamic coefficient.
    """
    return A0*cosd(alpha)**2


def Ay(alpha, A0):
    r"""Calculate the hydrodynamic coefficient :math:`\mathcal{A}_{y}` using the geometrical model:

    .. math::

        \mathcal{A}_{y} = 0.5\mathcal{A}_{0}\cos\alpha\sin\alpha.

    Parameters
    ----------
    alpha : array, scalar
        Dune orientation with respect to the perpendicular to the flow direction (in degree).
    A0 : array, scalar
        value of the hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.

    Returns
    -------
    array, scalar
         the hydrodynamic coefficient.
    """
    return A0*cosd(alpha)*sind(alpha)/2


def Bx(alpha, B0):
    r"""Calculate the hydrodynamic coefficient :math:`\mathcal{B}_{x}` using the geometrical model:

    .. math::

        \mathcal{B}_{x} = \mathcal{B}_{0}\cos^{2}\alpha.

    Parameters
    ----------
    alpha : array, scalar
        Dune orientation with respect to the perpendicular to the flow direction (in degree).
    B0 : array, scalar
        value of the hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.

    Returns
    -------
    array, scalar
         the hydrodynamic coefficient.
    """
    return B0*cosd(alpha)**2


def By(alpha, B0):
    r"""Calculate the hydrodynamic coefficient :math:`\mathcal{B}_{y}` using the geometrical model:

    .. math::

        \mathcal{B}_{y} = 0.5*\mathcal{B}_{0}\cos\alpha\sin\alpha

    Parameters
    ----------
    alpha : array, scalar
        Dune orientation with respect to the perpendicular to the flow direction (in degree).
    B0 : array, scalar
        value of the hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.

    Returns
    -------
    array, scalar
         the hydrodynamic coefficient.
    """
    return B0*cosd(alpha)*sind(alpha)/2


def _basal_shear_uni(x, y, alpha, A0, B0, AR):
    r"""Calculate the basal shear stress over a two dimensional sinusoidal topography for a wind from left to right (along the :math:`x`-direction):

        .. math::

            \Tau_{x} = \Re\left(1 + (\mathcal{A}_{x}(\alpha, \mathcal{A}_{0}) + i\mathcal{B}_{x}(\alpha, \mathcal{B}_{0}))k\xi\exp^{i\cos\alpha x + \sin\alpha y}\right)
            \Tau_{y} = \Re\left((\mathcal{A}_{y}(\alpha, \mathcal{A}_{0}) + i\mathcal{B}_{y}(\alpha, \mathcal{B}_{0}))k\xi\exp^{i\cos\alpha x + \sin\alpha y}\right)

    Parameters
    ----------
    x : array, scalar
        Streamwise coordinate, non-dimensional (:math:`kx`).
    y : array, scalar
        Spanwise coordinate, non-dimensional (:math:`ky`).
    alpha : array, scalar
        Dune orientation with respect to the perpendicular to the flow direction (in degree).
    A0 : array, scalar
        value of the in-phase hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.
    B0 : array, scalar
        value of the in-quadrature hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.
    AR : array, scalar
        dune aspect ratio, :math:`k\xi`.

    Returns
    -------
    Taux : array, scalar
        Streamwise component of the non-dimensional shear stress.
    Tauy : array, scalar
        Spanwise component of the non-dimensional shear stress

    """

    Taux = np.real(+ (1 + (Ax(alpha, A0) + 1j*Bx(alpha, B0))*AR*np.exp(1j*(cosd(alpha)*x + sind(alpha)*y))))
    Tauy = np.real(+ (Ay(alpha, A0) + 1j*By(alpha, B0))*AR*np.exp(1j*(cosd(alpha)*x + sind(alpha)*y)))
    return Taux, Tauy


def basal_shear(x, y, alpha, A0, B0, AR, theta):
    r"""Calculate the basal shear stress over a two dimensional sinusoidal topography for an arbitrary wind direction.

    Parameters
    ----------
    x : array, scalar
        Streamwise coordinate, non-dimensional (:math:`kx`).
    y : array, scalar
        Spanwise coordinate, non-dimensional (:math:`ky`).
    alpha : array, scalar
        Dune orientation with respect to the perpendicular to the flow direction (in degree).
    A0 : array, scalar
        value of the hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.
    B0 : array, scalar
        value of the hydrodynamic coefficient for :math:`\alpha = 0`, i.e. for a dune orientation perpendicular to the flow direction.
    AR : array, scalar
        dune aspect ratio, :math:`k\xi`.
    theta : array, scalar
        wind direction, in degree, in the trigonometric convention.

    Returns
    -------
    Taux : array, scalar
        streamwise component of the non-dimensional shear stress.
    Tauy : array, scalar
        spanwise component of the non-dimensional shear stress

    """
    # same but for an arbitrary wind direction oriented by theta
    xrot = x*cosd(theta) + y*sind(theta)
    yrot = y*cosd(theta) - x*sind(theta)
    alpha_rot = ((alpha - theta + 90) % 180) - 90
    # alpha_rot = alpha - theta
    Taux, Tauy = _basal_shear_uni(xrot, yrot, alpha_rot, A0, B0, AR)
    return cosd(theta)*Taux - sind(theta)*Tauy,  Taux*sind(theta) + Tauy*cosd(theta)
