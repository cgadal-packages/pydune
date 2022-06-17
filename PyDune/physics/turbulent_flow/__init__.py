# """
# .. autosummary::
#    # :template: custom-module-template.rst
#    # :toctree: _autosummary
#
#     PyDune.physics.turbulent_flow.flow_solver.solve_turbulent_flow
#     PyDune.physics.turbulent_flow.fourriere2010_unbounded.mu
#     PyDune.physics.turbulent_flow.geometrical_model
# """


from PyDune.physics.turbulent_flow.flow_solver import solve_turbulent_flow
from PyDune.physics.turbulent_flow.fourriere2010_unbounded import mu, mu_prime, A0_approx, B0_approx
from PyDune.physics.turbulent_flow.geometrical_model import *
