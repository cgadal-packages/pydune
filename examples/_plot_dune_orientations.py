r"""
==================
Dune orientations
==================

Calculating the dune orientations from the two dune growth mechanisms
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../')
from PyDune.Physics.Dune import CourrechDuPont2014 as CDP


theta = np.array([0, 120])
Q0 = np.array([5, 1])
alpha = np.arange(0, 361)

r"""
## Elongation direction
"""

flux_perp = CDP.Resultant_flux_perp_crest_at_crest(alpha[:, None], theta[None, :], Q0[None, :])
flux_aligned = CDP.Resultant_flux_aligned_crest_at_crest(alpha[:, None], theta[None, :], Q0[None, :])
alpha_E = CDP.Elongation_direction(theta, Q0)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(alpha, flux_perp, label='Flux perp.')
ax.plot(alpha, flux_aligned, label='Flux aligned')
ax.axhline(0, color='k', ls='--', lw=1)
ax.axvline(alpha_E, color='k', ls='--', lw=1)
ax.set_xlim(0, 360)
ax.set_xlabel('Crest orientation [deg.]')
ax.set_ylabel('Fluxes [-]')
fig.show()


r"""
## MGBNT orientation
"""

alpha = np.arange(0, 181)

sigma = CDP.Growth_rate(alpha[:, None], theta[None, :], Q0[None, :])
alpha_MGBNT = CDP.Bed_Instability_Orientation(theta, Q0)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(alpha, sigma)
ax.axhline(sigma.max(), color='k', ls='--', lw=1)
ax.axvline(alpha_MGBNT, color='k', ls='--', lw=1)
ax.set_xlim(0, 180)
ax.set_xlabel('Crest orientation [deg.]')
ax.set_ylabel(r'Growth rate [-]')
fig.show()
