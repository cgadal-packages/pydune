"""
Module containing functions related to the processing and ploting of wind/sandflux data.
"""

import itertools as it
import os

import numpy as np
import windrose as wd

from pydune.math import make_angular_PDF


def plot_flux_rose(
    angles,
    distribution,
    ax,
    fig,
    nsector=20,
    label_flux=False,
    label_angle=False,
    label=None,
    props=dict(boxstyle="round", facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0),
    blowfrom=False,
    xlabel=0.5,
    ylabel=0.05,
    **kwargs,
):
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
    nsector : int
        number of angular bins for the flux rose (the default is 20).
    label_flux : bool
        if True, labels the radial axis (the default is False).
    label_angle : bool
        if True, label the angles (the default is False).
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

    PdfQ = distribution / np.nansum(distribution)  # normalization
    # creating the new pdf with the number of bins
    Lbin = 360 / nsector
    Bins = np.arange(0, 360, Lbin)
    Qdat = []
    Qangle = []
    precision_flux = 0.001

    for n in range(len(Bins)):
        ind = np.argwhere((angles >= Bins[n] - Lbin / 2) & (angles < Bins[n] + Lbin / 2))
        integral = int(np.nansum(PdfQ[ind]) / precision_flux)
        for i in range(integral):
            Qangle.append(Bins[n])
            Qdat.append(1)
    Qangle = np.array(Qangle)
    # #### making the plot
    ax_rose = wd.WindroseAxes.from_ax(fig=fig)
    # bars = ax.bar(Angle, Intensity, normed=True, opening=1, edgecolor='k', nsector = Nsector, bins = Nbin, cmap = cmap)
    Qangle = (90 - Qangle) % 360
    if Qangle.size != 0:
        _ = ax_rose.bar(Qangle, Qdat, nsector=nsector, blowto=blowfrom, **kwargs)
        ax_rose.set_rmin(0)
        ax_rose.plot(0, 0, ".", color="w", zorder=100, markersize=3)
        # ax_rose.set_yticklabels(['{:.1f}'.format(float(i.get_text())*precision_flux) for i in ax.get_yticklabels()])
        if not label_angle:
            ax_rose.set_xticklabels([])
        if not label_flux:
            ax_rose.set_yticks([])
    if label is not None:
        fig.text(xlabel, ylabel, label, ha="center", va="center", transform=ax.transAxes, bbox=props)
    ax_rose.set_position(ax.get_position(), which="both")
    ax.remove()
    return ax_rose


def plot_wind_rose(
    theta,
    U,
    bins,
    ax,
    fig,
    label_angle=False,
    label=None,
    props=dict(boxstyle="round", facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0),
    blowfrom=False,
    legend=False,
    xlabel=0.5,
    ylabel=0.05,
    **kwargs,
):
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
    label_angle : bool
        if True, label the angles (the default is False).
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

    ax_rose = wd.WindroseAxes.from_ax(fig=fig)
    ax_rose.set_position(ax.get_position(), which="both")
    Angle = (90 - theta) % 360
    # ax_rose.bar(Angle, U, bins=bins, normed=True,  blowto=blowfrom, zorder=20, opening=1, edgecolor=None,
    #             linewidth=0.5, nsector=60, **kwargs)
    ax_rose.bar(Angle, U, bins=bins, normed=True, blowto=blowfrom, **kwargs)
    ax_rose.grid(True, linewidth=0.4, color="k", linestyle="--")
    ax_rose.patch.set_alpha(0.6)
    ax_rose.set_axisbelow(True)
    ax_rose.set_yticks([])
    ax_rose.set_yticklabels([])
    if not label_angle:
        ax_rose.set_xticklabels([])
    if legend:
        ax_rose.set_legend()
    if label is not None:
        fig.text(xlabel, ylabel, label, ha="center", va="center", transform=ax.transAxes, bbox=props)
    ax.remove()
    return ax_rose


def netcdf_to_flux_rose(
    file,
    ax,
    fig,
    netcdflonlatinds=(0, 0),
    z=10,
    z_0=1e-3,
    rho_g=2.65e3,
    rho_f=1,
    g=9.81,
    d=180e-6,
    shield_th=0.0035,
    Kappa=0.4,
    mu=0.63,
    cm=1.7,
    bin_edges=np.linspace(0, 360, 361),
    nsector=20,
    label_flux=False,
    label_angle=False,
    label=None,
    props=dict(boxstyle="round", facecolor=(1, 1, 1, 0.9), edgecolor=(1, 1, 1, 1), pad=0),
    blowfrom=False,
    **kwargs,
):
    r"""This function loads and concatenate (along the time axis) several NETCDF
    files from a list of filenames, calcuates the sand flux from a location in the
    NETCDF wind data using the quartic_transport_law, and plots a sand flux angular
    distribution on the given axe of the given figure.

    Parameters
    ----------
    files_list : list, str
        file name or list of downloaded file names.
    ax : matplotlib.Axes
        axe of the figure that will be replaced by the flux rose.
    fig : matplotlib.figure
        figure on which the flux rose is plotted.
    netcdflonlatinds : tuple
        the longitude and latatitude indicies of the netcdf file the flux rose should be calculated for. (default is (0,0)).
    z : scalar, numpy array
        elevation of the wind velocity (the default is 10). units: m.
    z_0 : scalar, numpy array
        roughness length of surface (the default is 10^-3). units: m.
    rho_g : scalar, numpy array
        density of sediment (the default is 2650). units: kg/m^3.
    rho_f : scalar, numpy array
        density of fluid (the default is 1). units: kg/m^3.
    g : scalar, numpy array
        gravity acceleration (the default is 9.81). units: m/s^2.
    d : scalar, numpy array
        sediment grain diameter (the default is 180*10^-6). units: m.
    shield_th : scalar, numpy array
        threshold shields number for transport initiation (the default is 0.0035). units: dimensionless.
    Kappa : scalar, numpy array
        von Kármán constant (the default is 0.4). units: dimensionless.
    mu : scalar, numpy array
        friction coefficient (the default is 0.63). units: dimensionless.
    cm : scalar, numpy array
        transport law coefficient (the default is 1.7). units: dimensionless.
    bin_edges : numpy array
        edges of the bins for finding the angular distribution of flux (the default is np.linspace(0, 360, 361)). units: degrees.
    nsector : int
        number of angular bins for the flux rose (the default is 20).
    label_flux : bool
        if True, labels the radial axis (the default is False).
    label_angle : bool
        if True, label the angles (the default is False).
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

    from PyDune.data_processing.meteorological.downloadCDS import load_netcdf
    from PyDune.math import cartesian_to_polar
    from PyDune.physics.sedtransport.transport_laws import quartic_transport_law

    if type(file) is str:
        data = load_netcdf([file])
    elif type(file) is list:
        data = load_netcdf(file)
    velocity, orientation = cartesian_to_polar(
        data["u10"][:, netcdflonlatinds[0], netcdflonlatinds[1]],
        data["v10"][:, netcdflonlatinds[0], netcdflonlatinds[1]],
    )
    shear_velocity = velocity_to_shear(velocity, z, z_0, Kappa)
    Q = np.sqrt((rho_g - rho_f * g * d) / rho_f) * d
    shield = (rho_f / ((rho_g - rho_f) * g * d)) * shear_velocity**2
    sand_flux = Q * quartic_transport_law(shield, shield_th, Kappa, mu, cm)
    angular_PDF, angles = make_angular_PDF(orientation, sand_flux, bin_edges)
    ax_rose = plot_flux_rose(
        angles, angular_PDF, ax, fig, nsector, label_flux, label_angle, label, props, blowfrom, **kwargs
    )
    return ax_rose


################################################################################
# Google earth functions
################################################################################


def create_KMZ(name, coordinates):
    """From a list of coordinates, create a .KMZ file that can be opened in
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
    loc_path = os.path.join(os.path.dirname(__file__), "src")
    # Destination file
    with open(name + ".kml", "w") as dest:
        # Writing the first Part
        with open(os.path.join(loc_path, "En_tete_era5.kml")) as entete:
            for line in it.islice(entete, 10, None):
                if line == "	<name>Skeleton_Coast.kmz</name>" + "\n":  # Premiere occurence
                    line = " 	<name>" + name + ".kmz</name>" + "\n"
                elif line == "		<name>Skeleton_Coast</name>" + "\n":  # Second occurence
                    line = " 	<name>" + name + "</name>" + "\n"
                dest.write(line)
        #
        # Writing placemarks
        with open(os.path.join(loc_path, "placemark.kml")) as placemark:
            for i, Coord in enumerate(coordinates):
                lat, lon = Coord
                lon = Coord[1]
                print("lat =", lat)
                print("lon =", lon)
                #
                for line in islice(placemark, 7, None):
                    if line == "			<name>1</name>" + "\n":
                        line = "			<name>" + str(i + 1) + "</name>" + "\n"
                    if line == "				<coordinates>11.25,-17.25,0</coordinates>" + "\n":
                        line = "				<coordinates>" + lon + "," + lat + ",0</coordinates>" + "\n"
                    dest.write(line)
                placemark.seek(0, 0)

        # Wrtiting closure
        with open(os.path.join(loc_path, "bottom_page.kml")) as bottom:
            dest.writelines(bottom.readlines()[7:])


"""
Sediment Fluxes calculation
"""

################################################################################
# Fluxes calculation


def velocity_to_shear(U, z, z_0=1e-3, Kappa=0.4):
    return U * Kappa / np.log(z / z_0)


def shear_to_velocity(Ustar, z, z_0=1e-3, Kappa=0.4):
    return Ustar * np.log(z / z_0) / Kappa
