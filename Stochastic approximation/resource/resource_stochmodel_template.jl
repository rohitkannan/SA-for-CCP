using JuMP, Gurobi, Distributions, StatsFuns, StatsBase

include("resource_nonlinear_20_30.jl")
include("resource_initialization_template.jl")


const gurobi_env = Gurobi.Env()

const numVariables = numResources


# generate samples of the random variables
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
const numAnalyticalSamples = 10000
const lambda_analytical, rho_analytical = generateRandomSamples(numAnalyticalSamples)

srand()


# compute conservative bound on the probability that
# constraints aren't satisfied
function computeRiskLevel(x::Array{Float64},reliabilityLevel::Float64=1E-06)

	const numSamples = size(lambda_analytical,2)
	simpleEst::Bool = true

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
# returns number of violated scenarios
function solveBasicModel(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64})

	const numSamples = size(lambda,2)
	obj = zeros(Float64,numSamples)
	dualmult = zeros(Float64,numResources,numSamples)
	
	
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

		status = solve(mod)

		obj[iter] = getvalue(slack)
		dualmult[:,iter] = getdual(con1)
	end
	
	mod = 0
	
	return obj, dualmult
end


# evaluate the constraint values for given decision vector x,
# realization of the random variables, and constraint scalings
function evaluateConstraint(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64},scalings::Float64)

	con, alpha_opt = solveBasicModel(x,lambda,rho)

	con /= scalings
	
	return con
end


# evaluate the constraint values for given decision vector x and
# realization of the random variables
function evaluateConstraint(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64})

	return evaluateConstraint(x,lambda,rho,1.0)
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64})

	return evaluateConstraintGradients(x,lambda,rho,1.0)
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64},scalings::Float64)

	const numSamples = size(lambda,2)
	grad_con = zeros(Float64,numResources,numSamples)
	
	con, alpha_opt = solveBasicModel(x,lambda,rho)
	
	for iter = 1:numSamples
		grad_con[:,iter] = 2*alpha_opt[:,iter].*rho[:,iter].*x/scalings
	end

	return grad_con
end


# evaluate constraint gradients
function evaluateConstraintAndGradient(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64},scalings::Float64)

	const numSamples = size(lambda,2)
	con = zeros(Float64,numSamples)
	grad_con = zeros(Float64,numResources,numSamples)
	
	con, alpha_opt = solveBasicModel(x,lambda,rho)
	con /= scalings
	
	for iter = 1:numSamples
		grad_con[:,iter] = 2*alpha_opt[:,iter].*rho[:,iter].*x/scalings
	end

	return con, grad_con
end


# determines if all constraints are satisfied for a given single scenario
# returns number of violated scenarios
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


# project the current iterate onto the feasible set
function projectOntoFeasibleSet(x_curr::Array{Float64},objectiveBound::Float64,proj_params::Array{Float64})

	mod = Model(solver=GurobiSolver(gurobi_env,Presolve=0,OutputFlag=0))

	@variable(mod, y[1:numResources] >= 0)

	@objective(mod, Min, sum((y[i]-x_curr[i])^2 for i = 1:numResources))
	@constraint(mod, sum(costVector[i]*y[i] for i=1:numResources) <= objectiveBound)

	solve(mod)	
	x_proj = getvalue(y)
	
	return x_proj, proj_params
end


# get the stochastic gradient of the approximation
function getStochasticGradient(x::Array{Float64},numSamples::Int64,mu::Float64,tau::Float64,scalings::Float64)

	stochgrad = zeros(numResources)
	
	lambda, rho = generateRandomSamples(numSamples)
	
	con, grad_con = evaluateConstraintAndGradient(x,lambda,rho,scalings)
	
	baseFactor1 = tau./((sqrt(mu)*exp.(0.5*tau*con) + sqrt(1.0/mu)*exp.(-0.5*tau*con)).^2)
	
	stochgrad = (grad_con*baseFactor1)/numSamples
	
	return stochgrad

end


# get the stochastic gradient of the approximation for fixed random sample
function getConstraintStochasticGradient(x::Array{Float64},lambda::Array{Float64},rho::Array{Float64},mu::Float64,tau::Float64,scalings::Float64)

	numSamples::Int64 = size(lambda,2)

	stochgrad = zeros(Float64,numResources)
	
	grad_con = evaluateConstraintGradients(x,lambda,rho,scalings)
		
	stochgrad = sum(grad_con,2)/numSamples
	
	return stochgrad

end


# estimate constraint scalings
function estimateScalings(x::Array{Float64},numSamples::Int64)

	scaling_tol::Float64 = 1E-06
	
	lambda, rho = generateRandomSamples(numSamples)
	
	constraint_values = evaluateConstraint(x,lambda,rho)

	abs_constraint_values = abs.(constraint_values)
	
	scalings::Float64 = max(median(abs_constraint_values),scaling_tol)

	return scalings
end


# estimate weak convexity parameter of the approximation
function estimateConstraintLipschitzConstant(x_ref::Array{Float64},numSamplesForLipVarEst::Int64,numGradientSamples::Int64,batchSize::Int64,mu::Float64,tau::Float64,scalings::Float64,objectiveBound::Float64)
	
	proj_params = zeros(Float64,3)
	
	lambda, rho = generateRandomSamples(batchSize)
	
	sample_radius::Float64 = norm(x_ref)/10.0
	
	L_max::Float64 = 0.0
	L_max_sqr::Float64 = 0.0
	
	for batch = 1:batchSize
		xi_start::Int64 = batch
		xi_end::Int64 = batch
		
		L_max_xi::Float64 = 0.0
		for iter = 1:numSamplesForLipVarEst
			x_1 = pickPointOnSphere(x_ref,sample_radius)
			x_1,~ = projectOntoFeasibleSet(x_1,objectiveBound,proj_params)
			
			x_2 = pickPointOnSphere(x_ref,sample_radius)
			x_2,~ = projectOntoFeasibleSet(x_2,objectiveBound,proj_params)
		
			grad_1 = getConstraintStochasticGradient(x_1,lambda[:,xi_start:xi_end],rho[:,xi_start:xi_end],mu,tau,scalings)
			grad_2 = getConstraintStochasticGradient(x_2,lambda[:,xi_start:xi_end],rho[:,xi_start:xi_end],mu,tau,scalings)
			
			L_est::Float64 = norm(grad_2 - grad_1)/norm(x_2 - x_1)
			if(L_est > L_max_xi)
				L_max_xi = L_est
			end
		end
		
		L_max += L_max_xi/batchSize
		L_max_sqr += (L_max_xi)^2/batchSize
	end
	
	return L_max, L_max_sqr
end


# estimate step length factor
function estimateStepLength(x_ref::Array{Float64},numGradientSamples::Int64,numSamplesForLipVarEst::Int64,
							batchSize::Int64,maxNumIterations::Int64,minNumReplicates::Int64,mu::Float64,tau::Float64,
							scalings::Float64,objectiveBound::Float64)
	
	proj_params = zeros(Float64,3)
	
	sample_radius::Float64 = norm(x_ref)/10.0

	grad_phi_up::Float64 = tau/4.0
	L_grad_phi::Float64 = 0.1*tau^2
	
	variance_phi = tau^2/16.0
	variance_avg::Float64 = 0.0
	for iter = 1:numSamplesForLipVarEst
		x_1 = pickPointOnSphere(x_ref,sample_radius)
		x_1,~ = projectOntoFeasibleSet(x_1,objectiveBound,proj_params)
	
		lambda, rho = generateRandomSamples(numGradientSamples*batchSize)

		variance_1::Float64 = 0.0
		
		for i = 1:batchSize
			xi_start::Int64 = (i-1)*numGradientSamples + 1
			xi_end::Int64 = i*numGradientSamples
			
			grad_1 = getConstraintStochasticGradient(x_1,lambda[:,xi_start:xi_end],rho[:,xi_start:xi_end],mu,tau,scalings)
			
			variance_1 += norm(grad_1)^2/batchSize
		end

		if(variance_1 > variance_avg)
			variance_avg = variance_1
		end
	end
	
	variance_est::Float64 = variance_avg*variance_phi
	
	
	L_max::Float64, L_max_sqr::Float64 = estimateConstraintLipschitzConstant(x_ref,numSamplesForLipVarEst,numGradientSamples,batchSize,mu,tau,scalings,objectiveBound)
	
	rho_est::Float64 = grad_phi_up*L_max + L_grad_phi*(2*sample_radius)^2*L_max_sqr/4.0 + L_grad_phi*L_max_sqr
	
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