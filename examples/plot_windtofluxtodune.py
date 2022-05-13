r"""
==================================================
From wind data to sand fluxes and dune orientations
===================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from PyDune.data_processing.meteorological.downloadCDS import load_netcdf
from PyDune.data_processing.meteorological.wind_plot import velocity_to_shear, plot_flux_rose, plot_wind_rose, make_angular_PDF
from PyDune.math import cartesian_to_polar
from PyDune.physics.sedtransport.transport_laws import quartic_transport_law
from PyDune.physics.dune.courrechdupont2014 import elongation_direction, bed_instability_orientation

# %%
# We first load the data, and caculate the shear velocity using the law of the wall:
#
data = load_netcdf(['src/ERA5LAND_winddata.nc'])
z_ERA = 10  # height of wind data in the dataset, [m]
#
velocity, orientation = cartesian_to_polar(data['u10'][:, 0, 0], data['v10'][:, 0, 0])
shear_velocity = velocity_to_shear(velocity, 10)

# figure
bins_shear = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
bins = [0, 1.5, 3, 4.5, 6, 7.5]
bbox_label = dict(boxstyle='round', facecolor=(0, 0, 0, 0.15), edgecolor=(0, 0, 0, 1))

fig, axarr = plt.subplots(1, 2, constrained_layout=True)
a = plot_wind_rose(orientation, velocity, bins, axarr[0], fig, opening=1,
                   nsector=25, cmap=plt.cm.viridis, legend=True, label='velocity (10m) [m/s]',
                   props=bbox_label)
a.set_legend(bbox_to_anchor=(-0.15, -0.15))
a = plot_wind_rose(orientation, shear_velocity, bins_shear, axarr[1], fig, opening=1,
                   nsector=25, cmap=plt.cm.viridis, legend=True, label='shear velocity [m/s]',
                   props=bbox_label)
a.set_legend(bbox_to_anchor=(-0.15, -0.15))
plt.show()

# %%
# We then calculate sand fluxes using the quartic law:

# # Parameters
sectoday = 24*3600
rho_g = 2.65e3  # grain density
rho_f = 1   # fluid density
g = 9.81  # [m/s2]
grain_diameters = 180e-6  # grain size [m]
Q = np.sqrt((rho_g - rho_f*g*grain_diameters)/rho_f)*grain_diameters  # characteristic flux [m2/s]
shield_th_quartic = 0.0035    # threshold shield numbers for the quartic

shield = (rho_f/((rho_g - rho_f)*g*grain_diameters))*shear_velocity**2  # shield number
sand_flux = Q*quartic_transport_law(shield, shield_th_quartic)*sectoday  # dimensional sand flux, [m2/day]
angular_PDF, angles = make_angular_PDF(orientation, sand_flux)

# figure
bins_flux = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
fig, axarr = plt.subplots(1, 2, constrained_layout=True)
a = plot_wind_rose(orientation, sand_flux, bins, axarr[0], fig, opening=1,
                   nsector=25, cmap=plt.cm.viridis, legend=True, label='sand fluxes [m2/day]',
                   props=bbox_label)
a.set_legend(bbox_to_anchor=(-0.15, -0.15))
a = plot_flux_rose(angles, angular_PDF, axarr[1], fig, opening=1,
                   label='angular distribution', nsector=25,
                   props=bbox_label)
plt.show()


# %%
# We then compute the two possible dune orientations:
Alpha_E = elongation_direction(angles, angular_PDF)
Alpha_BI = bed_instability_orientation(angles, angular_PDF)

print(r'$\alpha_{{\textup{{E}}}}={: .0f}~ ^ {{\circ}}$, $\alpha_{{\textup{{BI}}}}={: .0f}~ ^ {{\circ}}$'.format(Alpha_E, Alpha_BI))
