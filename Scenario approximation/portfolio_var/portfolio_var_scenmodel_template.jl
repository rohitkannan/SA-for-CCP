using JuMP, Gurobi, Distributions, StatsFuns


const gurobi_env = Gurobi.Env()


# DIMENSIONS
const numAssets = 1000
const numRandomVariables = numAssets
const numVariables = numAssets

const returnLBD = 1.2 #lower bound on the return


# CASES
# number of samples, replicates for scenario approximations
const NumSamples = ceil.(Int64,logspace(1,5,50))
const NumReplicates = 20


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
const solnFile = "solution.txt"
const riskFile = "riskLevel.txt"
const solnTimeFile = "solutionTime.txt"
const checkingTimeFile = "checkingTime.txt"
const sortingTimeFile = "sortingTime.txt"
const callbackFile = "numCallbacks.txt"
const numConstraintsFile = "numConstraintsEnforced.txt"


# compute negative probability of all constraints being satisfied
# for given scenarios of demands d and a given decision vector x
function computeRiskLevel(y_opt::Array{Float64})

	riskLevel::Float64 = 1.0 - normcdf((sum(mean[i]*y_opt[i] for i = 1:numAssets) - returnLBD)/norm(sigma*y_opt))
	
	return riskLevel
end


# generate random samples of the demands
function generateRandomSamples(numSamples::Int64)

	xi = Array{Float64}(numAssets,numSamples)
	for i = 1:numAssets
		xi[i,:] = rand(Normal(mean[i],stdev[i]), numSamples)
	end

	return xi
end


# determines if all constraints are satisfied for a given single scenario
# returns one if scenario constraints are all satisfied, zero otherwise
function getScenarioConstraintViolations(x::Array{Float64},xi::Array{Float64},pickFirstSetOfConstraints::Bool,maxNumConstraintsPerIteration::Int64)

	const numSamples = size(xi,2)
	numViolated::Int64 = 0
	constraintViolations = Float64[]
	constraintViolationIndices = Int64[]
	constraintViolationTolerance::Float64 = 1E-06

	for iter = 1:numSamples
		con::Float64 = returnLBD - xi[:,iter]'*x

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
		write(f,"numAssets: $numAssets \n")
		write(f,"returnLBD: $returnLBD \n")
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
	
	maxNumConstraintsPerIteration::Int64 = 100000
	numConstraintsAdded::Int64 = 0
	
	pickFirstSetOfConstraints::Bool = false
	
	sortingTime::Float64 = 0.0
	checkingTime::Float64 = 0.0
	numCallbacks::Int64 = 0

	
	objvar_opt::Float64 = Inf
	y_opt = zeros(Float64,numVariables)
	status::Symbol = Symbol("Error")
	
	
	numViolations::Int64 = numScenarios
	constraintIndices = Int64[]
	
	
	while(numViolations > 0)

		numCallbacks += 1
		numViolations = 0
		
		numConstraintsConsidered::Int64 = size(constraintIndices)[1]
	

		# construct the scenario model
		mod = Model(solver=GurobiSolver(gurobi_env,Presolve=0,OutputFlag=0))

		@variable(mod, y[1:numAssets] >= 0)

		@objective(mod, Min, sum(abs2(50.0*sigma[i,i])*y[i]*y[i] for i = 1:numAssets))
		@constraint(mod, sum(y[i] for i = 1:numAssets) == 1)
		@constraint(mod, [scen=1:numConstraintsConsidered], sum(xi[i,scen]*y[i] for i = 1:numAssets) >= returnLBD)
	
	
		status = solve(mod)
		y_opt = getvalue(y)
		objvar_opt = sum(abs2(sigma[i,i]*y_opt[i]) for i = 1:numAssets)


		numViolatedConstraints::Int64 = 0
		constraintViolations = Float64[]
		constraintViolationIndices = Int64[]

		tic()
		
		numViolatedConstraints, constraintViolations, constraintViolationIndices = getScenarioConstraintViolations(y_opt,xi,pickFirstSetOfConstraints,maxNumConstraintsPerIteration)
		
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

	riskLevel::Float64 = computeRiskLevel(y_opt)

	solutionTime::Float64 = toq()
		
	@printf "  Objective: %1.5f,  Risklevel: %.5f,  Time: %4.2f,  NumCallbacks: %d,  NumConstraints: %d \n" objvar_opt riskLevel solutionTime numCallbacks numConstraintsAdded
	
	writeSolutionToFile(dirName, status, objvar_opt, y_opt, riskLevel, solutionTime, checkingTime, sortingTime, numCallbacks, numConstraintsAdded)
	
	return status, objvar_opt, y_opt, riskLevel, solutionTime, checkingTime, sortingTime, numCallbacks, numConstraintsAdded
end