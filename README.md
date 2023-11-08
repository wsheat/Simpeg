# Simpeg

Notes
    -----
    Let :math:`\phi_d` represent the data misfit, :math:`\phi_m` represent the model
    objective function and :math:`\mathbf{m_0}` represent the starting model. The first
    model update is obtained by minimizing the a global objective function of the form:

        \phi (\mathbf{m_0}) = \phi_d (\mathbf{m_0}) + \beta_0 \phi_m (\mathbf{m_0})

    where :math:`\beta_0` represents the initial trade-off parameter (beta).
    Let :math:`\gamma` define the desired ratio between the data misfit and model
    objective functions at the initial beta iteration (defined by the 'beta0_ratio' input argument).
    Using the power iteration approach, our initial trade-off parameter is given by:

        \beta_0 = \gamma \frac{\lambda_d}{\lambda_m}

    where :math:`\lambda_d` as the largest eigenvalue of the Hessian of the data misfit, and
    :math:`\lambda_m` as the largest eigenvalue of the Hessian of the model objective function.
    For each Hessian, the largest eigenvalue is computed using power iteration. The input
    parameter 'n_pw_iter' sets the number of power iterations used in the estimate.

    For a description of the power iteration approach for estimating the larges eigenvalue,
    see :func:`SimPEG.utils.eigenvalue_by_power_iteration`.
