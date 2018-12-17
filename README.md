# SA-for-CCP
Julia codes for the paper "A stochastic approximation method for chance-constrained nonlinear programs"

To run the stochastic approximation examples, change the "model" entry in stochapprox_template.jl and run this file. 
For examples "normopt_noniid" and "resource", the corresponding folders contain files to obtain a better estimate of the risk levels at the final solutions.

To run the scenario approximation examples, change the "modelName" entry in scenapprox_template.jl and run this file.
For example "normopt_noniid", the corresponding folder contains code to obtain a better estimate of the risk levels at the final solutions.

To run the sigmoidal approximation example, run sigapprox_template.jl.

Make sure to change the output directory in all of the above files.
