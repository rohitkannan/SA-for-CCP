# SA-for-CCP
Julia codes for the paper "A stochastic approximation method for chance-constrained nonlinear programs". Results in the paper were generated using Julia 0.6.2, JuMP 0.18.2, Gurobi 7.5.2, IPOPT 3.12.8 (with
MUMPS as the linear solver), and SCIP 6.0.0.

Analytical solution: Codes to numerically approximate the efficient frontier for "portfolio" and "portfolio_var" models

Stochastic approximation: Run stochapprox_template.jl after changing the "model" entry in this file. 
The folders for the examples "normopt_noniid" and "resource" contain files to obtain a better estimate of the risk levels at the final solutions returned by this method.

Scenario approximation: Run scenapprox_template.jl after changing the "modelName" entry in this file.
The folder for example "normopt_noniid" contains code to obtain a better estimate of the risk levels at the final solutions returned by this method.

Sigmoidal approximation: Run sigapprox_template.jl.

Note: Make sure to change the working directory in all of the above files.