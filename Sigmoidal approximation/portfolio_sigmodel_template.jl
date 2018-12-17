using JuMP, Gurobi, Ipopt, Distributions, StatsFuns


const gurobi_env = Gurobi.Env()


# DIMENSIONS
const numAssets = 1000
const numRandomVariables = numAssets
const numVariables = numAssets+1

# use fixed gamma or adaptive gamma for sigmoidal approximation?
const fixedGamma = true


# CASES FOR SIGAPPROX
const riskLevels = [0.01]
const NumReplicates = 1
const NumSamplesFactor = [1,5,10,20,50,100]


#*============ OTHER MODEL PARAMETERS ================
const mean = Array{Float64}(numAssets)
const stdev = Array{Float64}(numAssets)
for i = 1:numAssets
	mean[i] = 1.05 + 0.3*(numAssets-i)/(numAssets-1)
	stdev[i] = (0.05 + 0.6*(numAssets-i)/(numAssets-1))/3
end
const sigma = zeros(Float64, numAssets, numAssets)
for i = 1:numAssets
	sigma[i,i] = stdev[i]
end
#*===============================================


# FILENAMES
const objFile = "objectiveValue.txt"
const iterObjFile = "iterObjective.txt"
const solnFile = "solution.txt"
const trueRiskFile = "computedRiskLevel.txt"
const iterRiskFile = "iterRiskLevel.txt"
const solnTimeFile = "solutionTime.txt"
const iterTimeFile = "iterTime.txt"
const CVaRTimeFile = "CVaRSolutionTime.txt"
const actualNumIterationsFile = "numIterations.txt"
const toptFile = "t_opt.txt"
const gammaFile = "gamma.txt"


# compute negative probability of all constraints being satisfied
# for given scenarios of demands d and a given decision vector x
function computeRiskLevel(y_opt::Array{Float64},objvar_opt::Float64)

	riskLevel::Float64 = 1.0 - normcdf((sum(mean[i]*y_opt[i] for i = 1:numAssets) - objvar_opt)/norm(sigma*y_opt))
	
	return riskLevel
end


# compute true alpha quantile of constraint at given solution
function getTrueQuantile(y::Array{Float64},objvar::Float64,riskLevel::Float64)

	numInitSamples::Int64 = 1000
	xi = generateRandomSamples(numInitSamples)
	con = zeros(Float64,numInitSamples)
	for scen = 1:numInitSamples
		con[scen] = objvar - sum(xi[i,scen]*y[i] for i = 1:numAssets)
	end
	var_low::Float64 = minimum(con)
	var_up::Float64 = maximum(con)
	var_tol::Float64 = 1E-06
	
	while(var_up - var_low > var_tol)
		var::Float64 = (var_low + var_up)/2.0
		estRisk::Float64 = computeRiskLevel(y,objvar-var)
		if(estRisk > riskLevel)
			var_low = var
		else
			var_up = var
		end
	end
	
	return var_up
end


# generate random samples of the demands
function generateRandomSamples(numSamples::Int64)

	xi = Array{Float64}(numAssets,numSamples)
	for i = 1:numAssets
		xi[i,:] = rand(Normal(mean[i],stdev[i]), numSamples)
	end

	return xi
end


# write solution to file
function initializeFiles(details_file::String)

	writeMode::String = "w"
	
	open(details_file, writeMode) do f
		write(f,"Model: $modelName \n")
		write(f,"numAssets: $numAssets \n")
		write(f,"riskLevels: $riskLevels \n")
		write(f,"NumReplicates: $NumReplicates \n")
		write(f,"NumSamplesFactor: $NumSamplesFactor \n")
		write(f,"Method: Sigmoidal approximations \n")
		write(f,"Fixed gamma?  $fixedGamma \n")
	end

end


# write solution to file
function writeSolutionToFile(dirName::String, objvar_opt::Float64, x_opt::Array{Float64}, trueRiskLevel::Float64, solutionTime::Float64, CVaRSolutionTime::Float64, actualNumIterations::Int64, t_opt::Float64, iterObjValue::Array{Float64}, iterRiskLevel::Array{Float64}, iterSolnTime::Array{Float64}, gamma::Float64)

	writeMode::String = "a"
	
	obj_file::String = dirName * objFile
	open(obj_file, writeMode) do f
		write(f,"$objvar_opt \n")
	end

	solution_file::String = dirName * solnFile
	open(solution_file, writeMode) do f
		for i = 1:numVariables
			write(f,"$(x_opt[i]) ")
		end
		write(f,"\n")
	end

	trueRisk_file::String = dirName * trueRiskFile
	open(trueRisk_file, writeMode) do f
		write(f,"$trueRiskLevel \n")
	end
	
	solntime_file::String = dirName * solnTimeFile
	open(solntime_file, writeMode) do f
		write(f,"$solutionTime \n")
	end
	
	cvartime_file::String = dirName * CVaRTimeFile
	open(cvartime_file, writeMode) do f
		write(f,"$CVaRSolutionTime \n")
	end
	
	iterations_file::String = dirName * actualNumIterationsFile
	open(iterations_file, writeMode) do f
		write(f,"$actualNumIterations \n")
	end
	
	topt_file::String = dirName * toptFile
	open(topt_file, writeMode) do f
		write(f,"$t_opt \n")
	end
	
	iter_obj_file::String = dirName * iterObjFile
	open(iter_obj_file, writeMode) do f
		write(f,"$iterObjValue \n")
	end
	
	iter_risk_file::String = dirName * iterRiskFile
	open(iter_risk_file, writeMode) do f
		write(f,"$iterRiskLevel \n")
	end
	
	iter_time_file::String = dirName * iterTimeFile
	open(iter_time_file, writeMode) do f
		write(f,"$iterSolnTime \n")
	end
	
	gamma_file::String = dirName * gammaFile
	open(gamma_file, writeMode) do f
		write(f,"$gamma \n")
	end

end


function solveCVaRApproximationModel(xi::Array{Float64},riskLevel::Float64)

	numScenarios::Int64 = size(xi,2)

	# construct the CVaR scenario model
	mod = Model(solver=GurobiSolver(gurobi_env,Presolve=0,OutputFlag=0))

	@variable(mod, objvar)
	@variable(mod, 0 <= y[1:numAssets] <= 1)
	@variable(mod, t)
	@variable(mod, z[1:numScenarios])
	@variable(mod, phi[1:numScenarios] >= 0)

	@objective(mod, Max, objvar)
	@constraint(mod, sum(y[i] for i = 1:numAssets) == 1)
	@constraint(mod, [scen=1:numScenarios], z[scen] == objvar - sum(xi[i,scen]*y[i] for i = 1:numAssets))
	@constraint(mod, [scen=1:numScenarios], phi[scen] >= z[scen] - t)
	@constraint(mod, sum(phi[scen] for scen = 1:numScenarios) <= -t*riskLevel*numScenarios)
	
	status = solve(mod)
	objvar_opt = getvalue(objvar)
	y_opt = getvalue(y)
	t_opt = getvalue(t)
	z_opt = getvalue(z)
	
	CVaR_true_risk::Float64 = computeRiskLevel(y_opt,objvar_opt)	
	CVaR_samp_risk::Float64 = sum(z_opt .> 0)*1.0/numScenarios

	println("CVAR RESULTS")
	println("CVaR obj: ",objvar_opt,"  sample risk level: ",CVaR_samp_risk,"  true risk level: ",CVaR_true_risk,"  max[z]: ",maximum(z_opt),"  t_opt: ",t_opt)
	
	return objvar_opt, y_opt, t_opt, CVaR_true_risk

end


# solve the scenario approximation model
function solveSigmoidalApproximationModel(riskLevel::Float64,numScenarios::Int64,dirName::String)

	tic()

	# sample the random variables
	xi = generateRandomSamples(numScenarios)
	
	
	const objectiveImprReq = 1E-04
	const maxNumSigmoidIterations = 100
	const lambda = 2.0
	const mu_bar = 2.506
	const maxIpoptIterations = 10000
	const IpoptTolerance = 1E-04
	const t_tolerance = 1E-06
	
	tic()
	
	objvar_opt::Float64, y_opt, t_opt::Float64, CVaR_true_risk::Float64 = solveCVaRApproximationModel(xi,riskLevel)

	if(fixedGamma)
		t_opt = -0.01
	else
		if(abs(t_opt) <= t_tolerance)
			t_opt = 1.0
			specRiskLevel::Float64 = riskLevel
			while(t_opt > -t_tolerance)
				specRiskLevel *= 2.0
				t_opt = getTrueQuantile(y_opt,objvar_opt,specRiskLevel)
			end
			println("t_opt tolerance activated!!!")
		end
	end
	
	CVaRSolutionTime::Float64 = toq()
	
	
	solve_status = []
	objectiveValues = Float64[]
	solutions = []
	sampleRisk = Float64[]
	trueRisk = Float64[]
	mu_values = Float64[]
	tau_values = Float64[]
	iterSolnTime = Float64[]

	
	mu::Float64 = mu_bar
	gamma::Float64 = -1/t_opt
	tau::Float64 = gamma*(mu+1)/2
	
	println("t_opt: ",t_opt)

	actualNumIterations::Int64 = 0
	status::Symbol = :Error
	
	ipopt_output_file_basic = dirName * "ipopt_out_"

	
	for iter = 1:maxNumSigmoidIterations
	
		tic()

		actualNumIterations += 1
		
		ipopt_output_file = ipopt_output_file_basic * string(iter) * ".txt"

		mod=Model(solver=IpoptSolver(max_iter=maxIpoptIterations,tol=IpoptTolerance,
				output_file=ipopt_output_file,file_print_level=5,print_frequency_iter=100,print_level=4,
				hessian_approximation="limited-memory",jac_c_constant="yes",max_cpu_time=3600.0))

		@variable(mod, objvar, start=objvar_opt)
		@variable(mod, 0 <= y[i=1:numAssets] <= 1, start=y_opt[i])
		@variable(mod, z[1:numScenarios])
		@variable(mod, phi[1:numScenarios] >= 0)
		@variable(mod, slack >= 0)

		@objective(mod, Max, objvar)
		@constraint(mod, sum(y[i] for i = 1:numAssets) == 1)
		@constraint(mod, [scen=1:numScenarios], z[scen] == objvar - sum(xi[i,scen]*y[i] for i = 1:numAssets))
		@NLconstraint(mod, [scen=1:numScenarios], phi[scen] >= 2*(1+mu)/(mu+exp(-tau*z[scen])) - 1)
		@constraint(mod, sum(phi[scen] for scen = 1:numScenarios) + slack == riskLevel*numScenarios)
		
		for scen = 1:numScenarios
			z_init::Float64 = objvar_opt - sum(xi[i,scen]*y_opt[i] for i = 1:numAssets)
			setvalue(z[scen], z_init)
			setvalue(phi[scen], 2*(1+mu)/(mu + exp(-tau*z_init)) - 1)
		end


		status = solve(mod)
		
		iterTime::Float64 = toq()
		
		push!(solve_status, status)
		push!(iterSolnTime, iterTime)
		push!(mu_values, mu)
		push!(tau_values, tau)
		
		
		if status != :Optimal
			println("Model not solved to optimality. Terminated!!!")
			if(iter == 1)
				push!(objectiveValues, NaN)
				push!(solutions, NaN*ones(numAssets))
				push!(sampleRisk, NaN)
				push!(trueRisk, NaN)
			end
			break
		end
		
		y_opt = getvalue(y)
		z_val = getvalue(z)
		objvar_tmp::Float64 = getvalue(objvar)
		
		SigVaR_samp_risk::Float64 = sum(z_val .> 0)*1.0/numScenarios
		SigVaR_true_risk::Float64 = computeRiskLevel(y_opt,objvar_tmp)
		

		push!(objectiveValues, objvar_tmp)
		push!(solutions, y_opt)
		push!(sampleRisk, SigVaR_samp_risk)
		push!(trueRisk, SigVaR_true_risk)
		
		
		mu = 2*mu
		tau = gamma*(mu+1)/2
	
	
		println("Iteration #",iter,":  SigVaR obj: ",objvar_tmp,"  sample risk level: ",SigVaR_samp_risk,"  true risk level: ",SigVaR_true_risk)

		
		new_objvar_opt = objvar_tmp
		if(iter > 1)
			if new_objvar_opt <= (1.0+objectiveImprReq)*objvar_opt
				println("Terminated because of insufficient progress")
				break
			end
		end
		objvar_opt = new_objvar_opt
	end
	
	# print results
	println("Results of sigmoidal approximation")
	println("solve status: ",solve_status)
	println("objective values: ",objectiveValues)
	println("sample risk: ",sampleRisk)
	println("computed risk level: ",trueRisk)
	println("specified risk level: ",riskLevel)
	println("mu values: ",mu_values)
	println("tau values: ",tau_values)
	println("iteration soln times: ",iterSolnTime)

	
		
	max_info = findmax(objectiveValues)
	objvar_opt = max_info[1]
	max_index::Int64 = max_info[2]
	y_opt = deepcopy(solutions[max_index])	
	x_opt = zeros(Float64,numAssets+1)
	for j = 1:numAssets
		x_opt[j] = y_opt[j]
	end
	x_opt[numAssets+1] = objvar_opt
	trueRiskLevel::Float64 = trueRisk[max_index]

	solutionTime::Float64 = toq()
	
	println("Final result of sigmoidal approximation")
	@printf "  Objective: %1.5f,  specified risk level: %.5f,  computed risk level: %.5f,  Time: %4.2f,  NumIterations: %d \n" objvar_opt riskLevel trueRiskLevel solutionTime actualNumIterations
	
	writeSolutionToFile(dirName, objvar_opt, x_opt, trueRiskLevel, solutionTime, CVaRSolutionTime, actualNumIterations, t_opt, objectiveValues, trueRisk, iterSolnTime, gamma)
	
	return objvar_opt, x_opt, trueRiskLevel, solutionTime, actualNumIterations
end
