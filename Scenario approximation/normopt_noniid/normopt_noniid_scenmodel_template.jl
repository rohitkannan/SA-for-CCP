using JuMP, Gurobi, Distributions, StatsFuns


const gurobi_env = Gurobi.Env()


# DIMENSIONS
const numVariables = 100
const numJCC = 100
const conRHS = numVariables^2


# CASES
# number of samples, replicates for scenario approximations
const NumSamples = ceil.(Int64,logspace(1,log10(50000.0),50))
const NumReplicates = 20


# FILENAMES
const objFile = "objectiveValue.txt"
const solnFile = "solution.txt"
const riskFile = "riskLevel.txt"
const riskTimeFile = "riskTime.txt"
const solnTimeFile = "solutionTime.txt"
const checkingTimeFile = "checkingTime.txt"
const sortingTimeFile = "sortingTime.txt"
const callbackFile = "numCallbacks.txt"
const numConstraintsFile = "numConstraintsEnforced.txt"



xi_mean = zeros(Float64,numVariables)
xi_covariance = 0.5*ones(Float64,numJCC,numJCC)

for j = 1:numVariables
	xi_mean[j] = j*1.0/numVariables
end

for i = 1:numJCC
	xi_covariance[i,i] = 1.0
end

xi_cov_chol_tmp = cholfact(xi_covariance)
xi_cov_chol = xi_cov_chol_tmp[:L]

xi_mean_mat = zeros(Float64,numJCC,numVariables)
for j = 1:numVariables
	xi_mean_mat[:,j] = xi_mean[j]*ones(numJCC)
end


# tailored implementation of multivariate normal distribution
function myMvNormal(numSamples::Int64)

	xi = zeros(Float64,numJCC,numVariables,numSamples)
	
	xi_tmp = rand(Normal(0.0,1.0),numJCC,numVariables,numSamples)
	for samp = 1:numSamples
		xi[:,:,samp] = xi_cov_chol*xi_tmp[:,:,samp] + xi_mean_mat
	end
	
	return xi
end


# generate random samples from multivariate normal distribution
function generateRandomSamples(numSamples::Int64)

	xi = myMvNormal(numSamples)
	
	return xi
end



srand(1234)

# generate samples to estimate actual risk level of solution
const numAnalyticalSamples = 20000
const xi_analytical = generateRandomSamples(numAnalyticalSamples)

srand()


# compute negative probability of all constraints being satisfied
# for given scenarios of demands d and a given decision vector x
function computeRiskLevel(x::Array{Float64},reliabilityLevel::Float64=1E-06)

	const numSamples = size(xi_analytical,3)
	simpleEst::Bool = false

	numViolated::Int64 = checkScenarioConstraints(x,xi_analytical)
	
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
function checkScenarioConstraints(x::Array{Float64},xi::Array{Float64})

	numSamples::Int64 = size(xi,3)

	numViolated::Int64 = 0
	for iter = 1:numSamples			
		for i = 1:numJCC
			con::Float64 = evaluateConstraint(i,x,xi[i,:,iter])
			if(con > 0)
				numViolated += 1
				break
			end
		end	
	end

	return numViolated
end


# evaluate the constraint values for given decision vector x,
# realization of the random variables xi, and constraint scalings
function evaluateConstraint(index::Int64,x::Array{Float64},xi::Array{Float64})

	con::Float64 = norm(xi.*x)^2 - conRHS

	return con
end


# determines if all constraints are satisfied for a given single scenario
# returns one if scenario constraints are all satisfied, zero otherwise
function getScenarioConstraintViolations(x::Array{Float64},xi::SubArray{Float64,2,Array{Float64,3},Tuple{Int64,Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true},pickFirstSetOfConstraints::Bool,maxNumConstraintsPerIteration::Int64)

	const numSamples = size(xi,2)
	numViolated::Int64 = 0
	constraintViolations = Float64[]
	constraintViolationIndices = Int64[]
	constraintViolationTolerance::Float64 = 1E-06

	for iter = 1:numSamples
		@views con::Float64 = norm(xi[:,iter].*x) - sqrt(conRHS)

		if(con >= constraintViolationTolerance)
			numViolated += 1
			push!(constraintViolations,con)
			push!(constraintViolationIndices,iter)
			if(pickFirstSetOfConstraints && numViolated == maxNumConstraintsPerIteration)
				break
			end
		end
	end
	
	return numViolated, constraintViolations, constraintViolationIndices
end


# write solution to file
function initializeFiles(details_file::String)

	writeMode::String = "w"
	
	open(details_file, writeMode) do f
		write(f,"Model: $modelName \n")
		write(f,"numVariables: $numVariables \n")
		write(f,"numJCC: $numJCC \n")
		write(f,"conRHS: $conRHS \n")
		write(f,"NumSamples: $NumSamples \n")
		write(f,"NumReplicates: $NumReplicates \n")
		write(f,"Method: Scenario approximations \n")
	end

end


# write solution to file
function writeSolutionToFile(dirName::String, status::Symbol, objvar_opt::Float64, x_opt::Array{Float64}, riskLevel::Float64, solutionTime::Float64, checkingTime::Float64, sortingTime::Float64, numCallbacks::Int64, numConstraintsAdded::Int64, riskTime::Float64)

	if(status == Symbol("Optimal"))
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
	
		risk_file::String = dirName * riskFile
		open(risk_file, writeMode) do f
			write(f,"$riskLevel \n")
		end
	
		risktime_file::String = dirName * riskTimeFile
		open(risktime_file, writeMode) do f
			write(f,"$riskTime \n")
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
	xi = generateRandomSamples(numScenarios)
	
	maxNumConstraintsPerIteration::Int64 = 10
	numConstraintsAdded::Int64 = 0
	
	pickFirstSetOfConstraints::Bool = false
	
	sortingTime::Float64 = 0.0
	checkingTime::Float64 = 0.0
	numCallbacks::Int64 = 0

	
	objvar_opt::Float64 = Inf
	y_opt = zeros(Float64,numVariables)
	status::Symbol = Symbol("Error")
	
	
	numViolations::Int64 = numScenarios
	
	constraintIndices = [Int64[] for i=1:numJCC]
	
	
	while(numViolations > 0)

		numCallbacks += 1
		numViolations = 0
		
		numConstraintsConsidered = zeros(Int64,numJCC)
		for i = 1:numJCC
			numConstraintsConsidered[i] = size(constraintIndices[i])[1]
		end
		
		
		mod = Model(solver=GurobiSolver(gurobi_env,Presolve=0,OutputFlag=0))

		@variable(mod, objvar)
		@variable(mod, 0 <= y[1:numVariables] <= 10^6)

		@objective(mod, Min, objvar)
		@constraint(mod, -sum(y[j] for j = 1:numVariables) <= objvar)
		@constraint(mod, [i=1:numJCC, scen=1:numConstraintsConsidered[i]], norm(xi[i,:,constraintIndices[i][scen]].*y) <= sqrt(conRHS))
	
		status = solve(mod)
		objvar_opt = getvalue(objvar)
		y_opt = getvalue(y)
		
		
		for i = 1:numJCC

			numViolatedConstraints::Int64 = 0
			constraintViolations = Float64[]
			constraintViolationIndices = Int64[]

			tic()
			
			@views numViolatedConstraints, constraintViolations, constraintViolationIndices = getScenarioConstraintViolations(y_opt,xi[i,:,:],pickFirstSetOfConstraints,maxNumConstraintsPerIteration)
			
			checkingTime += toq()
		
			numConstraintsEnforced::Int64 = min(size(constraintViolations,1),maxNumConstraintsPerIteration)
			
			tic()
			relevantIndices = 1:1:numConstraintsEnforced
			if(!pickFirstSetOfConstraints)
				relevantIndices = sortperm(constraintViolations, rev=true)
			end
			sortingTime += toq()
			
			for iter = 1:numConstraintsEnforced
				push!(constraintIndices[i],constraintViolationIndices[relevantIndices[iter]])
			end
			
			numConstraintsAdded += numConstraintsEnforced
			
			numViolations += numViolatedConstraints
		end

	end
	
	tic()

	riskLevel::Float64 = computeRiskLevel(y_opt)
	
	riskTime = toq()

	solutionTime::Float64 = toq()
		
	@printf "  Objective: %2.5f,  Risklevel: %.6f,  Time: %4.2f,  NumCallbacks: %d,  NumConstraints: %d \n" objvar_opt riskLevel solutionTime numCallbacks numConstraintsAdded
	
	writeSolutionToFile(dirName, status, objvar_opt, y_opt, riskLevel, solutionTime, checkingTime, sortingTime, numCallbacks, numConstraintsAdded, riskTime)
	
	return status, objvar_opt, y_opt, riskLevel, solutionTime, checkingTime, sortingTime, numCallbacks, numConstraintsAdded, riskTime
end
