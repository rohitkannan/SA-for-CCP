using JuMP, Gurobi, Distributions, StatsFuns, AmplNLWriter


include("resource_nonlinear_20_30.jl")


const gurobi_env = Gurobi.Env()


# CASES
# number of samples, replicates for scenario approximations
const NumSamples = ceil.(Int64,logspace(1,5,50))
const NumReplicates = 20



# FILENAMES
const objFile = "objectiveValue.txt"
const solnFile = "solution.txt"
const riskFile = "riskLevel.txt"
const solnTimeFile = "solutionTime.txt"
const checkingTimeFile = "checkingTime.txt"
const sortingTimeFile = "sortingTime.txt"
const riskTimeFile = "riskTime.txt"
const callbackFile = "numCallbacks.txt"
const numConstraintsFile = "numConstraintsEnforced.txt"



# compute conservative bound on the probability that
# constraints aren't satisfied
function boundRiskLevel(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64},reliabilityLevel::Float64=1E-06)

	const numSamples = size(lambda,2)
	simpleEst::Bool = false

	numViolated::Int64 = checkScenarioConstraints(x,lambda,rho)
	
	riskLevel::Float64 = Inf
	
	if(simpleEst)
		riskLevel = numViolated*1.0/numSamples
		return riskLevel
	else
		if(numViolated == 0)
			riskLevel = -log(reliabilityLevel)/numSamples
			return riskLevel
		end
	
		 # termination criterion for bisection
		gamma_tolerance::Float64 = 0.1/numSamples
		
		# use tail bounds on binomial distribution to estimate lower and upper bounds
		gamma_low::Float64 = (numViolated - sqrt(-numSamples*log(reliabilityLevel)/2.0))/numSamples
		if(gamma_low < 0.0)
			gamma_low = 0.0
		end
		gamma_up::Float64 = (numViolated + sqrt(-numSamples*log(reliabilityLevel)/2.0))/numSamples
		if(gamma_up > 1.0)
			gamma_up = 1.0
		end
		
		gamma::Float64 = 0.0
		func_value::Float64 = 0.0

		# use bisection to solve the nonlinear equation
		while (gamma_up - gamma_low > gamma_tolerance)

			gamma = (gamma_low + gamma_up)/2.0
			func_value = -reliabilityLevel
			for i = 1:numViolated
				tmp_func = i*log(gamma) + (numSamples-i)*log(1-gamma) + lfact(numSamples) - lfact(numSamples-i) - lfact(i)
				func_value += exp(tmp_func)
				if(func_value > 0)
					break
				end
			end
			if(func_value > 0)
				gamma_low = (gamma_low + gamma_up)/2.0
			else
				gamma_up = (gamma_low + gamma_up)/2.0
			end

		end
		
		riskLevel = gamma_up
		return riskLevel
	end

end


# determines if all constraints are satisfied for a given single scenario
# returns one if scenario constraints are all satisfied, zero otherwise
function checkScenarioConstraints(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64})

	numSamples::Int64 = size(lambda,2)

	numViolated::Int64 = 0
	
	mod = Model(solver=GurobiSolver(gurobi_env,Presolve=0,OutputFlag=0))

	@variable(mod, y[1:numResources,1:numCustomers] >= 0)

	@objective(mod, Min, 0)
	@constraint(mod, con1[i=1:numResources], sum(y[i,j] for j = 1:numCustomers) <= 0)
	@constraint(mod, con2[j=1:numCustomers], sum(nu[i,j]*y[i,j] for i = 1:numResources) >= 0)

	for iter = 1:numSamples
		for i = 1:numResources
			JuMP.setRHS(con1[i],rho[i,iter]*(x[i])^2)
		end
		
		for j = 1:numCustomers
			JuMP.setRHS(con2[j],lambda[j,iter])
		end

		status = solve(mod,suppress_warnings=true)

		if(status != Symbol("Optimal"))
			numViolated += 1
		end
	end
	
	# release memory
	mod = 0

	return numViolated
end


# generate random samples of the demands
function generateRandomSamples(numSamples::Int64)

	lambda = rand(MvNormal(lambda_mean[1,:],lambda_covariance),numSamples)
	rho = Array{Float64}(numResources,numSamples)
	for i = 1:numResources
		rho[i,:] = rand(Normal(rho_mean[i],rho_stdev),numSamples)
		for samp = 1:numSamples
			rho[i,samp] = min(rho[i,samp],rho_max)
		end
	end

	return lambda, rho
end



srand(1234)

# generate samples to estimate actual risk level of solution
const numAnalyticalSamples = 100000
const lambda_analytical, rho_analytical = generateRandomSamples(numAnalyticalSamples)

srand()



# determines if all constraints are satisfied for a given single scenario
# returns one if scenario constraints are all satisfied, zero otherwise
function getScenarioConstraintViolations(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64},pickFirstSetOfConstraints::Bool,maxNumConstraintsPerIteration::Int64)

	const numSamples = size(lambda,2)
	numViolated::Int64 = 0
	constraintViolations = Float64[]
	constraintViolationIndices = Int64[]
	constraintViolationTolerance::Float64 = 1E-06
	
	mod = Model(solver=GurobiSolver(gurobi_env,Presolve=0,OutputFlag=0))

	@variable(mod, y[1:numResources,1:numCustomers] >= 0)
	@variable(mod, slack)

	@objective(mod, Min, slack)
	@constraint(mod, con1[i=1:numResources], sum(y[i,j] for j = 1:numCustomers) <= 0)
	@constraint(mod, con2[j=1:numCustomers], sum(nu[i,j]*y[i,j] for i = 1:numResources) + slack >= 0)

	for iter = 1:numSamples
		for i = 1:numResources
			JuMP.setRHS(con1[i],rho[i,iter]*(x[i])^2)
		end
		
		for j = 1:numCustomers
			JuMP.setRHS(con2[j],lambda[j,iter])
		end

		status::Symbol = solve(mod)
		slack_opt::Float64 = getvalue(slack)

		if(slack_opt >= constraintViolationTolerance)
			numViolated += 1
			push!(constraintViolations,slack_opt)
			push!(constraintViolationIndices,iter)
			if(pickFirstSetOfConstraints && numViolated == numConstraintsPerIteration)
				break
			end
		end
	end
	
	# release memory
	mod = 0
	
	return numViolated, constraintViolations, constraintViolationIndices
end


# write solution to file
function initializeFiles(details_file::String)

	writeMode::String = "w"
	
	open(details_file, writeMode) do f
		write(f,"Model: $modelName \n")
		write(f,"numResources: $numResources \n")
		write(f,"numCustomers: $numCustomers \n")
		write(f,"NumSamples: $NumSamples \n")
		write(f,"NumReplicates: $NumReplicates \n")
		write(f,"Method: Scenario approximations \n")
	end

end


# write solution to file
function writeSolutionToFile(dirName::String, status::Symbol, objvar_opt::Float64, x_opt::Array{Float64}, riskLevel::Float64, solutionTime::Float64, checkingTime::Float64, sortingTime::Float64, riskTime::Float64, numCallbacks::Int64, numConstraintsAdded::Int64)

	if(status == Symbol("Optimal"))
		writeMode::String = "a"
		
		obj_file::String = dirName * objFile
		open(obj_file, writeMode) do f
			write(f,"$objvar_opt \n")
		end
	
		solution_file::String = dirName * solnFile
		open(solution_file, writeMode) do f
			for i = 1:numResources
				write(f,"$(x_opt[i]) ")
			end
			write(f,"\n")
		end
	
		risk_file::String = dirName * riskFile
		open(risk_file, writeMode) do f
			write(f,"$riskLevel \n")
		end
		
		solntime_file::String = dirName * solnTimeFile
		open(solntime_file, writeMode) do f
			write(f,"$solutionTime \n")
		end
		
		checktime_file::String = dirName * checkingTimeFile
		open(checktime_file, writeMode) do f
			write(f,"$checkingTime \n")
		end
		
		sorttime_file::String = dirName * sortingTimeFile
		open(sorttime_file, writeMode) do f
			write(f,"$sortingTime \n")
		end
		
		risktime_file::String = dirName * riskTimeFile
		open(risktime_file, writeMode) do f
			write(f,"$riskTime \n")
		end
		
		callback_file::String = dirName * callbackFile
		open(callback_file, writeMode) do f
			write(f,"$numCallbacks \n")
		end
		
		constraints_file::String = dirName * numConstraintsFile
		open(constraints_file, writeMode) do f
			write(f,"$numConstraintsAdded \n")
		end
	end

end


# solve the scenario approximation model
function solveScenarioApproximationModel(numScenarios::Int64,dirName::String)

	tic()

	# sample the random variables
	lambda, rho = generateRandomSamples(numScenarios)
	
	maxNumConstraintsPerIteration::Int64 = 5
	numConstraintsAdded::Int64 = 0
	
	pickFirstSetOfConstraints::Bool = false
	
	sortingTime::Float64 = 0.0
	checkingTime::Float64 = 0.0
	numCallbacks::Int64 = 0

	
	objvar_opt::Float64 = Inf
	x_opt = zeros(Float64,numResources)
	status::Symbol = Symbol("Error")
	x_initial_guess = zeros(Float64,numResources)
	
	
	numViolations::Int64 = numScenarios
	
	constraintIndices = Int64[]
	
	
	while(numViolations > 0)

		numCallbacks += 1
		numViolations = 0
		
		
		mod = Model(solver=AmplNLSolver("/usr/local/bin/scipampl", ["/home/rohitk/Research/chance_constraints/scenario_approximations/resource_nonlinear/scip.set"]))
		

		@variable(mod, objvar)
		@variable(mod, 0 <= x[i=1:numResources] <= 150,start=x_initial_guess[i])
		@variable(mod, 0 <= y[1:numResources,1:numCustomers,1:numConstraintsAdded] <= 150^2)

		@objective(mod, Min, objvar)
		@constraint(mod, sum(costVector[i]*x[i] for i = 1:numResources) <= objvar)
		@NLconstraint(mod, [i=1:numResources, scen=1:numConstraintsAdded], sum(y[i,j,scen] for j = 1:numCustomers) <= rho[i,constraintIndices[scen]]*(x[i])^2)
		@constraint(mod, [j=1:numCustomers, scen=1:numConstraintsAdded], sum(nu[i,j]*y[i,j,scen] for i = 1:numResources) >= lambda[j,constraintIndices[scen]])
	
	
		status = solve(mod)
		objvar_opt = getvalue(objvar)
		x_opt = getvalue(x)
		x_initial_guess = deepcopy(x_opt)
		
		
		# release memory
		mod = 0
		
		
		numViolatedConstraints::Int64 = 0
		constraintViolations = Float64[]
		constraintViolationIndices = Int64[]
		
		tic()
		
		numViolatedConstraints, constraintViolations, constraintViolationIndices = getScenarioConstraintViolations(x_opt,lambda,rho,pickFirstSetOfConstraints,maxNumConstraintsPerIteration)
		
		checkingTime += toq()
		
		
		numConstraintsEnforced::Int64 = min(size(constraintViolations,1),maxNumConstraintsPerIteration)
		
		tic()
		relevantIndices = 1:1:numConstraintsEnforced
		if(!pickFirstSetOfConstraints)
			relevantIndices = sortperm(constraintViolations, rev=true)
		end
		sortingTime += toq()
		
		for iter = 1:numConstraintsEnforced
			push!(constraintIndices,constraintViolationIndices[relevantIndices[iter]])
		end
		
		numConstraintsAdded += numConstraintsEnforced
		
		numViolations += numViolatedConstraints

	end

	tic()
	
	riskLevel::Float64 = boundRiskLevel(x_opt,lambda_analytical,rho_analytical)
	
	riskTime = toq()

	solutionTime::Float64 = toq()
		
	@printf "  Objective: %2.5f,  Risklevel: %.6f,  Time: %4.2f,  NumCallbacks: %d,  NumConstraints: %d \n" objvar_opt riskLevel solutionTime numCallbacks numConstraintsAdded
	
	writeSolutionToFile(dirName, status, objvar_opt, x_opt, riskLevel, solutionTime, checkingTime, sortingTime, riskTime, numCallbacks, numConstraintsAdded)
	
	return status, objvar_opt, x_opt, riskLevel, solutionTime, checkingTime, sortingTime, riskTime, numCallbacks, numConstraintsAdded
end
