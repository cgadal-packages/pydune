r"""
===========
Transport laws
===========

Create a plot comparing the different transport laws.
"""

import matplotlib.pyplot as plt
import numpy as np
from PyDune.physics.sedtransport import transport_laws as TL


theta = np.linspace(0, 0.8, 1000)
theta_d = 0.035
omega = 8


plt.figure()
plt.plot(theta, TL.quadratic_transport_law(theta, theta_d, omega), label='quadratic transport law')
plt.plot(theta, TL.cubic_transport_law(theta, theta_d, omega), label='cubic transport law')
plt.plot(theta, TL.quartic_transport_law(theta, theta_d), label='cubic transport law')
plt.xlabel('Shield number')
plt.ylabel('Non dimensional saturated flux')
plt.legend()
plt.tight_layout()
plt.show()
