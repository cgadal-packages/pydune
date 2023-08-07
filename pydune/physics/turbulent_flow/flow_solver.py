r"""
In this module, the function :func:`solve_turbulent_flow <PyDune.physics.turbulent_flow.flow_solver.solve_turbulent_flow>`
solve  the turbulent flow on a sinusoidal bottom in different configurations:

- '1D_unbounded': 1D, unbounded turbulent flow (in practice capped by a rigid lid, that should be put far from the bed) [1, 2]
- '1D_freesurface': 1D, turbulent flow capped by a free surface (typically river configuration) [1, 2]
- '1D_freeatmosphere': 1D, turbulent flow capped by a stratified flow (typically atmopshere configuration) [3]
- '2D_unbounded': 2D, unbounded turbulent flow (in practice capped by a rigid lid, that should be put far from the bed) [1, 2, 3]

For details on the flow theoretical modelling, please refer to the references below.


References
----------

[1] Fourrière, A. (2009). Morphodynamique des rivières: Sélection de la largeur, rides et dunes (Doctoral dissertation, Université Paris-Diderot-Paris VII).

[2] Fourriere, A., Claudin, P., & Andreotti, B. (2010). Bedforms in a turbulent stream: formation of ripples by primary linear instability and of dunes by nonlinear pattern coarsening. Journal of Fluid Mechanics, 649, 287-328.

[3] Andreotti, B., Fourriere, A., Ould-Kaddour, F., Murray, B., & Claudin, P. (2009). Giant aeolian dune size determined by the average depth of the atmospheric boundary layer. Nature, 457(7233), 1120-1123.

[4] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.

"""

import pydune.physics.turbulent_flow.fourriere2010_unbounded as fourriere2010_unbounded
import pydune.physics.turbulent_flow.fourriere2010_freesurface as fourriere2010_freesurface
import pydune.physics.turbulent_flow.andreotti2009 as andreotti2009
import pydune.physics.turbulent_flow.andreotti2011_unbounded as andreotti2011_unbounded


def solve_turbulent_flow(model, parameters, Kappa=0.4, max_z=None,
                         method='DOP853', atol=1e-10, rtol=1e-10, **kwargs):
    r"""This function solves the perturbation of the flow induced by a sinusoidal
    bottom in various configurations. The description of each configuration and
    associated parameters is done in the description of the module.

    .. warning::
        The solver performs poorly for too large integration domains due to accumulation of numerical errors.
        In practice, the user should be carefull when using values of `eta_H` (:math:`\eta_{H} = k H`) larger than 10.

    Parameters
    ----------
    model : str
        Chosen configuration. It can be ``'1D_unbounded'``, ``'1D_freesurface'``, ``'1D_freeatmosphere'`` or ``'2D_unbounded'``.
    parameters : dict
        Dictionnary containing the physical parameters necessary for solving each models. For each model, the list of parameters the dictionnary must contain is:

        - 1D_unbounded: ``eta_0``, ``eta_H``
        - 1D_freesurface: ``eta_0``, ``eta_H``, ``Fr``
        - 1D_freeatmosphere: ``eta_0``, ``eta_H``, ``eta_B``, ``Fr``
        - 2D_unbounded: ``eta_0``, ``eta_H``, ``Fr``, ``alpha``
    Kappa : float, optional
        Von Karmàn constant (the default is 0.4).
    max_z : float, optional
        Maximum vertical position where the system is solved, and also where the boundary conditons are applied.
        Usually set to something slightly smaller than `eta_H` to avoid the very slow resolution close to the top of the boundary layer.
        Usefull when investigating the solution close to the bottom (the default is ``eta_H``).
    method: str, optional
        Numerical method used to solve the equations. It is passed to the solver
        :func:`solve_ivp <scipy.integrate.solve_ivp>` (default is ``DOP853``, which corresponds to an Explicit Runge-Kutta method of order 8 with an adaptative time-step).
    atol,rtol : float, optional
        Absolute tolerance. The solver keeps the local error estimate smaller
        than :math:`atol + rtol*abs(y)`, where :math:`y` is the solution at a given time step.
        More information can be found in the documentation of
        :func:`solve_ivp <scipy.integrate.solve_ivp>` (the default is 1e-10 for both).
    **kwargs : optional
        Any other optional parameters that can be passed to :func:`solve_ivp <scipy.integrate.solve_ivp>`.

    Returns
    -------
    solution: func, list of func
        function calculating the solution by taking as argument values of the vertical coordinate
        :math:`\eta = k z` (float, array_like). It is built from seleveral
        :class:`ODE_solution <scipy.integrate.OdeSolution>`. See corresponding
        documentation for more information about the interpolation algorithm. If
        ``model = '1D_freeatmosphere'``, return also a function calculating the
        streamfunction above eta_H. Its arguments are, in this order:
        - vertical coordinate :math:`\eta = k z`, numpy array
        - the horizontal coordinate :math:`\eta = k x`, numpy array,
        -   aspect ratio the bottom perturbation :math:`\eta = k \xi` (float)


    References
    ----------
    .. line-block::
        [1] Fourrière, A. (2009). Morphodynamique des rivières: Sélection de la largeur, rides et dunes (Doctoral dissertation, Université Paris-Diderot-Paris VII).
        [2] Fourriere, A., Claudin, P., & Andreotti, B. (2010). Bedforms in a turbulent stream: formation of ripples by primary linear instability and of dunes by nonlinear pattern coarsening. Journal of Fluid Mechanics, 649, 287-328.
        [3] Andreotti, B., Fourriere, A., Ould-Kaddour, F., Murray, B., & Claudin, P. (2009). Giant aeolian dune size determined by the average depth of the atmospheric boundary layer. Nature, 457(7233), 1120-1123.
        [4] Andreotti, B., Claudin, P., Devauchelle, O., Durán, O., & Fourrière, A. (2012). Bedforms in a turbulent stream: ripples, chevrons and antidunes. Journal of Fluid Mechanics, 690, 94-128.
    """

    if model == '1D_unbounded':
        solution = fourriere2010_unbounded.calculate_solution(parameters['eta_0'],
                                                              parameters['eta_H'],
                                                              max_z=max_z,
                                                              Kappa=Kappa,
                                                              atol=atol,
                                                              rtol=rtol,
                                                              **kwargs)
    elif model == '1D_freesurface':
        solution = fourriere2010_freesurface.calculate_solution(parameters['eta_0'],
                                                                parameters['eta_H'],
                                                                parameters['Fr'],
                                                                max_z=max_z,
                                                                Kappa=Kappa,
                                                                atol=atol,
                                                                rtol=rtol,
                                                                **kwargs)
    elif model == '2D_unbounded':
        solution = andreotti2011_unbounded.calculate_solution(parameters['eta_0'],
                                                              parameters['eta_H'],
                                                              parameters['alpha'],
                                                              max_z=max_z,
                                                              Kappa=Kappa,
                                                              atol=atol,
                                                              rtol=rtol,
                                                              **kwargs)
    elif model == '1D_freeatmosphere':
        solution = andreotti2009.calculate_solution(parameters['eta_0'],
                                                    parameters['eta_H'],
                                                    parameters['eta_B'],
                                                    parameters['Fr'],
                                                    max_z=max_z,
                                                    Kappa=Kappa,
                                                    atol=atol,
                                                    rtol=rtol,
                                                    **kwargs)
    return solution
