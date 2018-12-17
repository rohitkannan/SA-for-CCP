using JuMP, Gurobi, Distributions, StatsFuns, StatsBase


const gurobi_env = Gurobi.Env()


const numVariables = 100
const numJCC = 100
const conRHS = (numVariables*1.0)^2



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
const numAnalyticalSamples = 100000
const xi_analytical = generateRandomSamples(numAnalyticalSamples)

srand()




# compute conservative bound on the probability that
# constraints aren't satisfied
function boundRiskLevel(x::Array{Float64},reliabilityLevel::Float64=1E-06)

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
	for samp = 1:numSamples			
		for i = 1:numJCC
			con::Float64 = evaluateConstraint(i,x,xi[i,:,samp])
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


NumSamples = 50

# solution file name
const baseDirName = "C:/Users/rkannan/Desktop/SA for CCP/Scenario approximation/experiments/normopt_noniid_scenapprox/"


for samp = 1:NumSamples

	const dirName = baseDirName * string(samp) * "/"

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
	end

end