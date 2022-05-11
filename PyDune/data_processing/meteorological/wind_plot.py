"""
Module containing functions related to the processing and ploting of wind/sandflux data.
"""

import os
import numpy as np
from itertools import islice
from windrose import WindroseAxes
from xhistogram.core import histogram


def plot_flux_rose(angles, distribution, ax, fig, nbins=20, withaxe=False, label=None,
                   props=dict(boxstyle='round', facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0),
                   blowfrom=False, **kwargs):
    r"""Plot a sand flux angular distribution, or flux rose, on the given axe of the given figure.

    Parameters
    ----------
    angles : numpy array
        bin centers of the angular sand flux distribution, in degrees. Here, the angles show
        where the wind blows to, are anticlockwise and the 0 it a wind blowing to the East.
    distribution : numpy array
        sand flux angular distribution.
    ax : matplotlib.Axes
        axe of the figure that will be replaced by the flux rose.
    fig : matplotlib.figure
        figure on which the flux rose is plotted.
    nbins : int
        number of angular bins for the flux rose (the default is 20).
    withaxe : bool
        if true, labels the angular axis (the default is False).
    label : str, None
        if provided, labels the flux rose with the given string (the default is None).
    props : dict
        Bbox properties used around the label (the default is dict(boxstyle='round', facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0)).
    blowfrom : bool
        If blow from, the rose will be :math:`\pi`-rotated, to show where the fluxes come from (the default is False).
    **kwargs : other kwargs
        any other parameter supported by :func:`windrose.WindroseAxes.bar <windrose.WindroseAxes>`

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
        _ = ax_rose.bar(Qangle, Qdat, nsector=nbins, blowto=blowfrom, **kwargs)
        ax_rose.set_rmin(0)
        ax_rose.plot(0, 0, '.', color='w', zorder=100, markersize=3)
        # ax_rose.set_yticklabels(['{:.1f}'.format(float(i.get_text())*precision_flux) for i in ax.get_yticklabels()])
        if not withaxe:
            ax_rose.set_yticks([])
    if label is not None:
        fig.text(0.5, 0.05, label, ha='center', va='center', transform=ax.transAxes, bbox=props)
    ax.remove()
    return ax_rose


def plot_wind_rose(theta, U, bins, ax, fig, label=None,
                   props=dict(boxstyle='round', facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0),
                   blowfrom=False,
                   legend=False, **kwargs):
    r"""Plot a wind rose on the given axe of the given figure.

    Parameters
    ----------
    theta : numpy array
        time series of the wind angle, in degree. Here, the angles show
        where the wind blows to, are anticlockwise and the 0 it a wind blowing to the East.
    U : numpy array
        time series of the wind velocity.
    bins : list
        velocity bin edges.
    ax : matplotlib.Axes
        axe of the figure that will be replaced by the flux rose.
    fig : matplotlib.figure
        figure on which the flux rose is plotted.
    label : str, None
        if provided, labels the flux rose with the given string (the default is None).
    props : dict
        Bbox properties used around the label (the default is dict(boxstyle='round', facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0)).
    blowfrom : bool
        If blow from, the rose will be :math:`\pi`-rotated, to show where the fluxes come from (the default is False).
    legend : bool
        If True, display a legend for the velocity bins (the default is False).
    **kwargs : type
        any other optional parameters that can be passed to :func:`windrose.WindroseAxes.bar <windrose.WindroseAxes>`.

    Returns
    -------
    WindroseAxes
        return the axe on which the wind rose is plotted. Can be used for further modifications.

    """

    ax_rose = WindroseAxes.from_ax(fig=fig)
    ax_rose.set_position(ax.get_position(), which='both')
    Angle = (90 - theta) % 360
    # ax_rose.bar(Angle, U, bins=bins, normed=True,  blowto=blowfrom, zorder=20, opening=1, edgecolor=None,
    #             linewidth=0.5, nsector=60, **kwargs)
    ax_rose.bar(Angle, U, bins=bins, normed=True,  blowto=blowfrom, **kwargs)
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


################################################################################
# Google earth functions
################################################################################


def create_KMZ(name, coordinates):
    """ From a list of coordinates, create a .KMZ file that can be opened in
    GoogleEarth, and will display markers at the input coordinates.

    Parameters
    ----------
    name : str
        filename for the output KMZ file.
    coordinates : list, numpy array
        list of coordinates (latitude, longitude)

    Returns
    -------
    None
        nothing, just create the KMZ file.

    """
    loc_path = os.path.join(os.path.dirname(__file__), 'src')
    # Destination file
    with open(name + '.kml', 'w') as dest:
        # Writing the first Part
        with open(os.path.join(loc_path, 'En_tete_era5.kml'), 'r') as entete:
            for line in islice(entete, 10, None):
                if line == '	<name>Skeleton_Coast.kmz</name>' + '\n':  # Premiere occurence
                    line = ' 	<name>' + name + '.kmz</name>' + '\n'
                elif line == '		<name>Skeleton_Coast</name>' + '\n':  # Second occurence
                    line = ' 	<name>' + name + '</name>'+'\n'
                dest.write(line)
        #
        # Writing placemarks
        with open(os.path.join(loc_path, 'placemark.kml'), 'r') as placemark:
            for i, Coord in enumerate(coordinates):
                lat, lon = Coord
                lon = Coord[1]
                print('lat =', lat)
                print('lon =', lon)
                #
                for line in islice(placemark, 7, None):
                    if line == '			<name>1</name>' + '\n':
                        line = '			<name>' + str(i + 1) + '</name>' + '\n'
                    if line == '				<coordinates>11.25,-17.25,0</coordinates>' + '\n':
                        line = '				<coordinates>' + lon + ',' + lat + ',0</coordinates>' + '\n'
                    dest.write(line)
                placemark.seek(0, 0)

        # Wrtiting closure
        with open(os.path.join(loc_path, 'bottom_page.kml'), 'r') as bottom:
            dest.writelines(bottom.readlines()[7:])


"""
Sediment Fluxes calculation
"""

################################################################################
# Fluxes calculation


def velocity_to_shear(U, z, z_0=1e-3, Kappa=0.4):
    return U*Kappa/np.log(z/z_0)


def shear_to_velocity(Ustar, z, z_0=1e-3, Kappa=0.4):
    return Ustar*np.log(z/z_0)/Kappa


def make_angular_PDF(angles, weight, bin_edges=np.linspace(0, 360, 361), axis=-1):
    hist, _ = histogram(angles, bins=bin_edges, density=1, weights=weight, axis=axis)
    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2
    return hist, bin_centers


def make_angular_average(angles, weight, bin_edges=np.linspace(0, 360, 361), axis=-1):
    hist, _ = histogram(angles, bins=bin_edges, weights=weight, axis=axis)
    counts, _ = histogram(angles, bins=bin_edges, axis=axis)
    bin_centers = np.array([np.mean(bin_edges[i:i+2]) for i in range(bin_edges.size - 1)])
    hist[counts == 0] = 1
    counts[counts == 0] = 1
    return hist/counts, bin_centers

    # def Calculate_Fluxes(Ustar, transport_law, intermittency=False, **kwargs):
    #     if not intermittency:
    #         return transport_law(Ustar, **kwargs)
    #     else:
    #         print('intermittency not implemented yet. Using continuous transport loaw instead')
    #         return transport_law(Ustar, **kwargs)
