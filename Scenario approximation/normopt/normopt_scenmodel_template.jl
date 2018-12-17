using JuMP, Gurobi, Distributions, StatsFuns


const gurobi_env = Gurobi.Env()


# DIMENSIONS
const numVariables = 100
const numJCC = 100
const conRHS = numVariables^2


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
const callbackFile = "numCallbacks.txt"
const numConstraintsFile = "numConstraintsEnforced.txt"


# compute negative probability of all constraints being satisfied
# for given scenarios of demands d and a given decision vector x
function computeRiskLevel(x::Array{Float64})

	x_sqr = Array{Float64}(0)
	for j = 1:numVariables
		if(x[j] > 0)
			push!(x_sqr,(x[j])^2)
		end
	end
	
	n::Int64 = size(x_sqr,1)
	x_sqr = sort(x_sqr)
	
	if(x_sqr[n] > x_sqr[1])
		baseFactor1::Float64 = 1.0
		for i = 1:n-1
			baseFactor1 *= sqrt(x_sqr[n]/x_sqr[i])
		end
		
		k::Int64 = 1
		error_tol::Float64 = 1E-09/numJCC
		error::Float64 = 1.0
		while(error > error_tol)
			k *= 2
			baseFactor2 = exp(lgamma(n/2.0 + k) - lgamma(n/2.0))
			baseFactor3 = exp(k*log(x_sqr[n]/x_sqr[1] - 1.0) - k*log(2) - lfact(k))
			baseFactor4 = chisqcdf(n+2*k,conRHS/x_sqr[n])
			error = baseFactor1*baseFactor2*baseFactor3*baseFactor4
		end
		
		p::Float64 = 2.0/(1.0/x_sqr[1] + 1.0/x_sqr[n])
		g_base = 1.0 - p./x_sqr
		g = zeros(k-1)
		for i = 1:k-1
			g[i] = sum(g_base.^i)
		end
		
		c = zeros(Float64,k)
		c[1] = 1.0
		for j = 1:n
			c[1] *= sqrt(p/x_sqr[j])
		end
		for j = 1:k-1
			for r = 1:j
				c[j+1] += g[j+1-r]*c[r]
			end
			c[j+1] /= (2.0*j)
		end
		
		prob = 0.0
		for j = 0:k-1
			prob += c[j+1]*chisqcdf(n+2*j,conRHS/p)
		end
		
	else
		prob = chisqcdf(n,conRHS/x_sqr[1])
	end
	
	riskLevel::Float64 = 1.0 - (prob)^(numJCC)
	
	return riskLevel
end


# generate random samples of the demands
function generateRandomSamples(numSamples::Int64)

	xi = rand(Normal(0.0,1.0), numJCC, numVariables, numSamples)
	return xi
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
function writeSolutionToFile(dirName::String, status::Symbol, objvar_opt::Float64, x_opt::Array{Float64}, riskLevel::Float64, solutionTime::Float64, checkingTime::Float64, sortingTime::Float64, numCallbacks::Int64, numConstraintsAdded::Int64)

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

	riskLevel::Float64 = computeRiskLevel(y_opt)

	solutionTime::Float64 = toq()
		
	@printf "  Objective: %2.5f,  Risklevel: %.6f,  Time: %4.2f,  NumCallbacks: %d,  NumConstraints: %d \n" objvar_opt riskLevel solutionTime numCallbacks numConstraintsAdded
	
	writeSolutionToFile(dirName, status, objvar_opt, y_opt, riskLevel, solutionTime, checkingTime, sortingTime, numCallbacks, numConstraintsAdded)
	
	return status, objvar_opt, y_opt, riskLevel, solutionTime, checkingTime, sortingTime, numCallbacks, numConstraintsAdded
end
