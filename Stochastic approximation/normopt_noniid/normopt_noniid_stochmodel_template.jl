using JuMP, Gurobi, Distributions, StatsFuns, StatsBase

include("normopt_noniid_initialization_template.jl")


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
const numAnalyticalSamples = 20000
const xi_analytical = generateRandomSamples(numAnalyticalSamples)

srand()



# compute negative probability of all constraints being satisfied
# for given scenarios of demands d and a given decision vector x
function computeRiskLevel(x::Array{Float64},reliabilityLevel::Float64=1E-06)

	const numSamples = size(xi_analytical,3)
	simpleEst::Bool = true

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
function evaluateConstraint(x::Array{Float64},xi::Array{Float64},scalings::Array{Float64})

	con = (sum((xi.*x').^2,2) - conRHS*ones(numJCC))./scalings

	return con
end


# evaluate the constraint values for given decision vector x,
# realization of the random variables xi, and constraint scalings
function evaluateConstraint(x::Array{Float64},xi::Array{Float64})

	con = evaluateConstraint(x,xi,ones(numJCC))

	return con
end


# evaluate the constraint values for given decision vector x,
# realization of the random variables xi, and constraint scalings
function evaluateConstraint(index::Int64,x::Array{Float64},xi::Array{Float64})

	con::Float64 = norm(xi.*x)^2 - conRHS

	return con
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},xi::Array{Float64})

	return evaluateConstraintGradients(x,xi,ones(numJCC))
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},xi::Array{Float64},scalings::Array{Float64})

	grad_con = zeros(Float64,numJCC,numVariables)
	
	for i = 1:numJCC
		grad_con[i,:] = (2*(xi[i,:]).^2.*x)/scalings[i]
	end

	return grad_con
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},xi::Array{Float64},scalings::Float64)

	grad_con = (2*(xi.^2).*x)/scalings

	return grad_con
end



# project the current iterate onto the feasible set
function projectOntoFeasibleSet(x_curr::Array{Float64},objectiveBound::Float64)

	mod = Model(solver=GurobiSolver(gurobi_env,Presolve=0,OutputFlag=0))

	@variable(mod, y[j=1:numVariables] >= 0)

	@objective(mod, Min, sum((y[i]-x_curr[i])^2 for i = 1:numVariables))
	@constraint(mod, -sum(y[j] for j=1:numVariables) <= objectiveBound)

	solve(mod)	
	x_proj = getvalue(y)
	
	return x_proj	
end


# get the stochastic gradient of the approximation
function getStochasticGradient(x::Array{Float64},numSamples::Int64,mu::Float64,tau::Float64,scalings::Array{Float64})

	xi = generateRandomSamples(numSamples)

	stochgrad = zeros(Float64,numVariables)
	
	for samp = 1:numSamples
		con = evaluateConstraint(x,xi[:,:,samp],scalings)
		
		max_info = findmax(con)
		max_con::Float64 = max_info[1]
		max_index::Int64 = max_info[2]

		baseFactor1::Float64 = tau/(sqrt(mu)*exp(0.5*tau*max_con) + sqrt(1.0/mu)*exp(-0.5*tau*max_con))^2
		
		grad_max_con = evaluateConstraintGradients(x,xi[max_index,:,samp],scalings[max_index])

		stochgrad += baseFactor1*grad_max_con/numSamples
	end
	
	return stochgrad

end


# get the stochastic gradient of the approximation for fixed random sample
function getStochasticGradientForFixedSample(x::Array{Float64},xi::SubArray{Float64,3,Array{Float64,3},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},mu::Float64,tau::Float64,scalings::Array{Float64})

	numSamples::Int64 = size(xi,3)

	stochgrad = zeros(Float64,numJCC,numVariables)
	
	for samp = 1:numSamples
	
		con = evaluateConstraint(x,xi[:,:,samp],scalings)
	
		grad_con = evaluateConstraintGradients(x,xi[:,:,samp],scalings)

		for i = 1:numJCC
			baseFactor1::Float64 = tau/(sqrt(mu)*exp(0.5*tau*con[i]) + sqrt(1.0/mu)*exp(-0.5*tau*con[i]))^2
			
			stochgrad[i,:] += baseFactor1*grad_con[i,:]/numSamples
		end
	end
	
	return stochgrad

end


# estimate constraint scalings
function estimateScalings(x::Array{Float64},numSamples::Int64)
	
	scaling_tol::Float64 = 1E-06
	
	xi = generateRandomSamples(numSamples)
	
	constraint_values = zeros(Float64,numJCC,numSamples)

	for samp = 1:numSamples
		constraint_values[:,samp] = evaluateConstraint(x,xi[:,:,samp],ones(numJCC))
	end

	scalings = zeros(Float64,numJCC)
	for index = 1:numJCC	
		scalings[index] = max(median(abs.(constraint_values[index,:])),scaling_tol)
	end
	
	return scalings
end




# estimate Lipschitz constant of the gradient of the approximation
function estimateCompositeLipschitzConstant(x_ref::Array{Float64},numSamplesForLipVarEst::Int64,numGradientSamples::Int64,batchSize::Int64,mu::Float64,tau::Float64,scalings::Array{Float64},objectiveBound::Float64)
	
	sample_radius::Float64 = norm(x_ref)/10.0
	
	xi = generateRandomSamples(batchSize)
	
	L_avg::Float64 = 0.0
	for batch = 1:batchSize
		xi_start::Int64 = batch
		xi_end::Int64 = batch
		
		L_max::Float64 = 0.0
		for samp = 1:numSamplesForLipVarEst
			x_1 = pickPointOnSphere(x_ref,sample_radius)
			x_1 = projectOntoFeasibleSet(x_1,objectiveBound)
			
			x_2 = pickPointOnSphere(x_ref,sample_radius)
			x_2 = projectOntoFeasibleSet(x_2,objectiveBound)	
		
			@views grad_1 = getStochasticGradientForFixedSample(x_1,xi[:,:,xi_start:xi_end],mu,tau,scalings)
			@views grad_2 = getStochasticGradientForFixedSample(x_2,xi[:,:,xi_start:xi_end],mu,tau,scalings)
			
			L_est::Float64 = norm(grad_2 - grad_1)/norm(x_2 - x_1)
			if(L_est > L_max)
				L_max = L_est
			end
		end
		
		L_avg += L_max/batchSize
	end
	
	return L_avg
end


# estimate step length factor
function estimateStepLength(x_ref::Array{Float64},numGradientSamples::Int64,
							numSamplesForLipVarEst::Int64,batchSize::Int64,maxNumIterations::Int64,
							minNumReplicates::Int64,mu::Float64,tau::Float64,
							scalings::Array{Float64},objectiveBound::Float64)
	
	sample_radius::Float64 = norm(x_ref)/10.0
	
	rho_est::Float64 = estimateCompositeLipschitzConstant(x_ref,numSamplesForLipVarEst,numGradientSamples,batchSize,mu,tau,scalings,objectiveBound)
	
	
	variance_est::Float64 = 0.0
	for samp = 1:numSamplesForLipVarEst
		x_1 = pickPointOnSphere(x_ref,sample_radius)
		x_1 = projectOntoFeasibleSet(x_1,objectiveBound)

		variance_1::Float64 = 0.0
		for batch = 1:batchSize
			grad_1 = getStochasticGradient(x_1,numGradientSamples,mu,tau,scalings)
			
			variance_1 += norm(grad_1)^2/batchSize
		end

		if(variance_1 > variance_est)
			variance_est = variance_1
		end
	end
	
	gamma::Float64 = 1/sqrt(rho_est*variance_est)
	
	stepLength::Float64 = gamma/sqrt((maxNumIterations+1.0)*minNumReplicates)

	return stepLength
end


# given a reference point, pick a random point within a given radius
function pickPointOnSphere(x_orig::Array{Float64},radius::Float64)

	if(radius < 1E-09)
		error("Sampling radius in pickPointOnSphere is almost zero! Set manually.")
	end

	n::Int64 = size(x_orig,1)
	unifRand = rand(1)
	factor1::Float64 = radius*(unifRand[1])^(1.0/n)
	normRand = rand(Normal(0,1), n)
	factor2::Float64 = norm(normRand)
	
	x_samp = x_orig + factor1*normRand/factor2

	return x_samp
end