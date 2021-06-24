# @Author: gadal
# @Date:   2021-02-16T18:39:45+01:00
# @Email:  gadal@ipgp.fr
# @Last modified by:   gadal
# @Last modified time: 2021-03-02T17:01:18+01:00

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


def wind_rose(Angle, Intensity, place=None, fig=None, legend=False, coord=False, **kwargs):
    """Short summary.

    Parameters
    ----------
    Angle : type
        Description of parameter `Angle`.
    Intensity : type
        Description of parameter `Intensity`.
    place : type
        Description of parameter `place` (the default is None).
    fig : type
        Description of parameter `fig` (the default is None).
    legend : type
        Description of parameter `legend` (the default is False).
    coord : type
        Description of parameter `coord` (the default is False).
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    Examples
    --------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    # Angle : Orientation of the wind
    # Intensity : Intensity of tje wind
    # Nbin : nbins in terms of velocity
    # Nsector : n bins in terms of direction
    #
    # documentation https://windrose.readthedocs.io/en/latest/

    Angle = np.array(Angle)
    Intensity = np.array(Intensity)

    # removing nans
    inds = ~np.logical_or(np.isnan(Angle), np.isnan(Intensity))
    Angle = Angle[inds]
    Intensity = Intensity[inds]

    ax = WindroseAxes.from_ax(fig=fig)
    if place is not None:
        ax.set_position(place, which='both')
    # bars = ax.bar(Angle, Intensity, normed=True, opening=1, edgecolor='k', nsector = Nsector, bins = Nbin, cmap = cmap)
    Angle = (90 - Angle) % 360
    _ = ax.bar(Angle, Intensity,  **kwargs, zorder=20)
    ax.set_axisbelow(True)
    if legend:
        ax.set_legend()
    if not coord:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return ax


def flux_rose(Angle, PdfQ_tp, withaxe=0, place=None, fig=None, nsector=20, **kwargs):
    # pdfQ flux distribution
    # Corresponding angles in degree
    # N bin nuber of bins for the rose
    # withaxe : if 0, removes everything except the bars
    # place :: where on the figure

    PdfQ = PdfQ_tp/np.nansum(PdfQ_tp)  # normalization
    # creating the new pdf with the number of bins
    Lbin = 360/nsector
    Bins = np.arange(0, 360, Lbin)
    Qdat = []
    Qangle = []
    precision_flux = 0.001

    for n in range(len(Bins)):
        ind = np.argwhere((Angle >= Bins[n] - Lbin/2) & (Angle < Bins[n] + Lbin/2))
        integral = int(np.nansum(PdfQ[ind])/precision_flux)
        for i in range(integral):
            Qangle.append(Bins[n])
            Qdat.append(1)
    Qangle = np.array(Qangle)
    # ax = plt.subplot(111, projection='polar')
    ax = WindroseAxes.from_ax(fig=fig)
    if place is not None:
        ax.set_position(place, which='both')
    # bars = ax.bar(Angle, Intensity, normed=True, opening=1, edgecolor='k', nsector = Nsector, bins = Nbin, cmap = cmap)
    Qangle = (90 - Qangle) % 360
    if Qangle.size != 0:
        _ = ax.bar(Qangle, Qdat, nsector=nsector, **kwargs)
        ax.set_rmin(0)
        plt.plot(0, 0, '.', color='w', zorder=100, markersize=3)
        ax.set_yticklabels(['{:.1f}'.format(float(i.get_text())*precision_flux) for i in ax.get_yticklabels()])
        if withaxe != 1:
            ax.set_yticks([])
    return ax


# def Write_wind_rose(self, dir, ext='.pdf', **kwargs):
#     if os.path.isdir(dir) == False:
#         os.mkdir(dir)
#     i = 0
#     Npoints = self.Uwind.shape[0]*self.Uwind.shape[1]
#     format_string = '{:0' + str(int(np.log10(Npoints)) + 1) + '}'
#     for y in range(self.Uwind.shape[1]):
#         for x in range(self.Uwind.shape[0]):
#             print('Point number' + str(i))
#             plt.ioff()
#             fig = plt.figure()
#             wind_rose(self.Uorientation[x,y,:],self.Ustrength[x,y,:], fig = fig, **kwargs)
#             plt.savefig(dir + '/wind_rose_'+ format_string.format(i+1) + ext)
#             plt.close('all')
#             i = i + 1
#
# def Write_flux_rose(self, dir, ext = '.pdf', **kwargs):
#     if os.path.isdir(dir) == False:
#         os.mkdir(dir)
#     i = 0
#     Npoints = self.Uwind.shape[0]*self.Uwind.shape[1]
#     format_string = '{:0' + str(int(np.log10(Npoints)) + 1) + '}'
#     print('Printing flux roses ...')
#     for y in range(self.Uwind.shape[1]):
#         for x in range(self.Uwind.shape[0]):
#             print('Point number' + str(i))
#             pdfQ, Angle  = PDF_flux(self.Qorientation[x,y,:],self.Qstrength[x,y,:])
#             fig = plt.figure()
#             flux_rose(Angle,pdfQ, fig = fig, **kwargs)
#             plt.savefig(dir + '/flux_rose_'+ format_string.format(i+1) + ext)
#             plt.close('all')
#             i = i + 1

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
    hist = histogram(angles, bins=bin_edges, density=1, weights=weight, axis=axis)
    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2
    return hist, bin_centers


def Make_angular_average(angles, weight, bin_edges=np.linspace(0, 360, 361), axis=-1):
    hist = histogram(angles, bins=bin_edges, weights=weight, axis=axis)
    counts = histogram(angles, bins=bin_edges, axis=axis)
    bin_centers = np.array([np.mean(bin_edges[i:i+2]) for i in range(bin_edges.size - 1)])
    hist[counts == 0] = 1
    counts[counts == 0] = 1
    return hist/counts, bin_centers
