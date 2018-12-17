using JuMP, Gurobi, Distributions, StatsFuns, StatsBase

include("resource_nonlinear_20_30.jl")


const gurobi_env = Gurobi.Env()

const numVariables = numResources


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
lambda_analytical, rho_analytical = generateRandomSamples(numAnalyticalSamples)

srand()




# compute conservative bound on the probability that
# constraints aren't satisfied
function boundRiskLevel(x::Array{Float64},reliabilityLevel::Float64=1E-06)

	const numSamples = size(lambda_analytical,2)
	simpleEst::Bool = false

	numViolated::Int64 = checkScenarioConstraints(x,lambda_analytical,rho_analytical)
	
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

	const numSamples = size(lambda,2)
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
	
	mod = 0
	
	return numViolated
end


numTrials = 10


# solution file name
const baseDirName = "C:/Users/rkannan/Desktop/SA for CCP/Stochastic approximation/experiments/resource_stochapprox/"


for trial = 1:numTrials

	const dirName = baseDirName * "rep" * string(trial) * "/"

	const fileName = dirName * "solution.txt"

	const outputFile = dirName * "trueRiskLevels.txt"
	open(outputFile, "w") do f
	end

	const timeFile = dirName * "trueRiskTime.txt"
	open(timeFile, "w") do f
	end

	soln = readdlm(fileName)

	const numPoints = size(soln,1)

	for iter = 1:numPoints
		tic()
		trueRiskLevel::Float64 = boundRiskLevel(soln[iter,:])
		riskTime = toq()
		println("Risk level #",iter," : ",trueRiskLevel,"  time: ",riskTime)
		open(outputFile, "a") do f
			write(f,"$trueRiskLevel \n")
		end
		open(timeFile, "a") do f
			write(f,"$riskTime \n")
		end
		gc()
	end
end
