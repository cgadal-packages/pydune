import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from xhistogram.core import histogram
# from PyDune.Physics.SedTransport import quadratic_law
# from .xhistogram_perso.core import histogram
# from matplotlib.patches import Circle


"""
Plot functions
"""


def plot_flux_rose(angles, distribution, ax, fig, nbins=20, withaxe=0, label=None,
                   props=dict(boxstyle='round', facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0),
                   **kwargs):
    """Short summary.

    Parameters
    ----------
    angles : array_like
        bin center in orientation of the flux distribution.
    distributions : array_like
        angular flux distribution.
    ax : matplotlib.axes
        ax of the figure on which the wind rose is plotted.
    fig : matplotlib.figure
        figure on which the wind rose is plotted.
    nbins : int
        number of angular bins for the plot (the default is 20).
    withaxe : 0 or 1
        Define if the polar axes are plotted or not (the default is 0).
    label : str
        If not None, sets a label at the bottom of the flux rose (the default is None).
    **kwargs :
        Optional parameters passed to :func:`windrose.WindroseAxes.bar <windrose.WindroseAxes>`.

    Returns
    -------
    WindroseAxes
        return the axe on which the wind rose is plotted. Can be used for further modifications.

    """

    PdfQ = distribution/np.nansum(distribution)  # normalization
    # creating the new pdf with the number of bins
    Lbin = 360/nbins
    Bins = np.arange(0, 360, Lbin)
    Qdat = []
    Qangle = []
    precision_flux = 0.001

    for n in range(len(Bins)):
        ind = np.argwhere((angles >= Bins[n] - Lbin/2) & (angles < Bins[n] + Lbin/2))
        integral = int(np.nansum(PdfQ[ind])/precision_flux)
        for i in range(integral):
            Qangle.append(Bins[n])
            Qdat.append(1)
    Qangle = np.array(Qangle)
    # #### making the plot
    ax_rose = WindroseAxes.from_ax(fig=fig)
    ax_rose.set_position(ax.get_position(), which='both')
    # bars = ax.bar(Angle, Intensity, normed=True, opening=1, edgecolor='k', nsector = Nsector, bins = Nbin, cmap = cmap)
    Qangle = (90 - Qangle) % 360
    if Qangle.size != 0:
        _ = ax_rose.bar(Qangle, Qdat, nsector=nbins, **kwargs)
        ax_rose.set_rmin(0)
        ax_rose.plot(0, 0, '.', color='w', zorder=100, markersize=3)
        # ax_rose.set_yticklabels(['{:.1f}'.format(float(i.get_text())*precision_flux) for i in ax.get_yticklabels()])
        if withaxe != 1:
            ax_rose.set_yticks([])
    if label is not None:
        fig.text(0.5, 0.05, label, ha='center', va='center', transform=ax.transAxes, bbox=props)
    ax.remove()
    return ax_rose


def plot_wind_rose(theta, U, bins, ax, fig, label=None,
                   props=dict(boxstyle='round', facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0),
                   legend=False, **kwargs):
    """Plot a wind rose from one dimensional time series.

    Parameters
    ----------
    theta : array_like
        Wind orientation in the trigonometric convention.
    U : array_like
        Wind velocity, same shape as `theta`.
    bins : list
        Velocity bin edges.
    ax : matplotlib.axes
        ax of the figure on which the wind rose is plotted.
    fig : matplotlib.figure
        figure on which the wind rose is plotted
    label : str or None
        if not None, label plotted below the wind rose (default is None).
    **kwargs :
        Optional parameters passed to :func:`windrose.WindroseAxes.bar <windrose.WindroseAxes>`.

    Returns
    -------
    WindroseAxes
        return the axe on which the wind rose is plotted. Can be used for further modifications.

    """
    ax_rose = WindroseAxes.from_ax(fig=fig)
    ax_rose.set_position(ax.get_position(), which='both')
    Angle = (90 - theta) % 360
    ax_rose.bar(Angle, U, bins=bins, normed=True, zorder=20, opening=1, edgecolor=None,
                linewidth=0.5, nsector=60, **kwargs)
    ax_rose.grid(True, linewidth=0.4, color='k', linestyle='--')
    ax_rose.patch.set_alpha(0.6)
    ax_rose.set_axisbelow(True)
    ax_rose.set_yticks([])
    ax_rose.set_xticklabels([])
    ax_rose.set_yticklabels([])
    if legend:
        ax_rose.set_legend()
    if label is not None:
        fig.text(0.5, 0.05, label, ha='center', va='center', transform=ax.transAxes, bbox=props)
    ax.remove()
    return ax_rose


"""
Sediment Fluxes calculation
"""

################################################################################
# Fluxes calculation


def Velocity_to_shear(U, z, z_0=1e-3, Kappa=0.4):
    return U*Kappa/np.log(z/z_0)


def Shear_to_velocity(Ustar, z, z_0=1e-3, Kappa=0.4):
    return Ustar*np.log(z/z_0)/Kappa


def Calculate_Fluxes(Ustar, transport_law, intermittency=False, **kwargs):
    if not intermittency:
        return transport_law(Ustar, **kwargs)
    else:
        print('intermittency not implemented yet. Using continuous transport loaw instead')
        return transport_law(Ustar, **kwargs)


def Make_angular_PDF(angles, weight, bin_edges=np.linspace(0, 360, 361), axis=-1):
    hist, _ = histogram(angles, bins=bin_edges, density=1, weights=weight, axis=axis)
    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2
    return hist, bin_centers


def Make_angular_average(angles, weight, bin_edges=np.linspace(0, 360, 361), axis=-1):
    hist, _ = histogram(angles, bins=bin_edges, weights=weight, axis=axis)
    counts, _ = histogram(angles, bins=bin_edges, axis=axis)
    bin_centers = np.array([np.mean(bin_edges[i:i+2]) for i in range(bin_edges.size - 1)])
    hist[counts == 0] = 1
    counts[counts == 0] = 1
    return hist/counts, bin_centers
