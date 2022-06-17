r"""
=============================
Selection of dune orientation
=============================

In this example, we show how the calculation of the elonagtion direction and MGBNT
orientation are selected in the model of COurrech du Pont et al 2014.

[1] Courrech du Pont, S., Narteau, C., & Gao, X. (2014). Two modes for dune orientation. Geology, 42(9), 743-746.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyDune.physics.dune import courrechdupont2014 as CDP


theta = np.array([0, 120])
Q0 = np.array([5, 1])
alpha = np.arange(0, 361)

# %%
# Elongation direction
# ====================


flux_perp = CDP.resultant_flux_perp_crest_at_crest(alpha[:, None], theta[None, :], Q0[None, :])
flux_aligned = CDP.resultant_flux_aligned_crest_at_crest(alpha[:, None], theta[None, :], Q0[None, :])
alpha_E = CDP.elongation_direction(theta, Q0)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(alpha, flux_perp, label='Flux perp. to crest')
ax.plot(alpha, flux_aligned, label='Flux aligned with crest')
ax.axhline(0, color='k', ls='--', lw=1)
ax.axvline(alpha_E, color='k', ls='--', lw=1)
ax.set_xlim(0, 360)
ax.set_xlabel('Crest orientation [deg.]')
ax.set_ylabel('Fluxes [-]')
ax.legend()
fig.show()


# %%
# MGBNT orientation
# =================

alpha = np.arange(0, 181)

sigma = CDP.growth_rate(alpha[:, None], theta[None, :], Q0[None, :])
alpha_MGBNT = CDP.MGBNT_orientation(theta, Q0)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(alpha, sigma)
ax.axhline(sigma.max(), color='k', ls='--', lw=1)
ax.axvline(alpha_MGBNT, color='k', ls='--', lw=1)
ax.set_xlim(0, 180)
ax.set_xlabel('Crest orientation [deg.]')
ax.set_ylabel(r'Growth rate [-]')
fig.show()
