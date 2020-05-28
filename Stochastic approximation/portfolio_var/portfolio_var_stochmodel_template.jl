using JuMP, Gurobi, Distributions, StatsFuns, StatsBase

include("portfolio_var_initialization_template.jl")


const gurobi_env = Gurobi.Env()

const numVariables = 1000
const numAssets = numVariables

const returnLBD = 1.2 #lower bound on the return

#*============ MODEL PARAMETERS ================
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


# compute probability that constraints aren't satisfied
function computeRiskLevel(y_opt::Array{Float64})

	riskLevel::Float64 = 1.0 - normcdf((sum(mean[i]*y_opt[i] for i = 1:numAssets) - returnLBD)/norm(sigma*y_opt))
	
	return riskLevel
end


# evaluate the constraint values for given decision vector x
# and realization of the random variables xi
function evaluateConstraint(x::Array{Float64},xi::Array{Float64})

	return evaluateConstraint(x,xi,1.0)
end


# evaluate the constraint values for given decision vector x,
# realization of the random variables xi, and constraint scalings
function evaluateConstraint(x::Array{Float64},xi::Array{Float64},scalings::Float64)

	con::Float64 = (returnLBD - xi'*x)/scalings

	return con
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},xi::Array{Float64})

	return evaluateConstraintGradients(x,xi,1.0)
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},xi::Array{Float64},scalings::Float64)

	grad_con = zeros(Float64,numVariables)
	for j = 1:numVariables
		grad_con[j] = -xi[j]
	end
	grad_con /= scalings

	return grad_con
end


# generate random samples of the stock returns
function generateRandomSamples(numSamples::Int64)

	xi = Array{Float64}(numAssets,numSamples)
	for i = 1:numAssets
		xi[i,:] = rand(Normal(mean[i],stdev[i]), numSamples)
	end

	return xi
end


# project on to the set {x[1:n] >= 0, sum(x[1:n]) = 1}
function projectOntoSimplex(y_ref::Array{Float64})
	
	y = deepcopy(y_ref)
	
	v = Array{Float64}(1)
	v[1] = y[1]
	w = Array{Float64}(0)
	rho::Float64 = y[1] - 1.0
	
	for j = 2:numVariables
		if(y[j] > rho)
			rho = rho + (y[j] - rho)/(size(v,1) + 1)
			if(rho > y[j] - 1.0)
				push!(v,y[j])
			else
				append!(w,v)
				v = [y[j]]
				rho = y[j] - 1.0
			end
		end
	end
	
	if(size(w,1) > 0)
		for j = 1:size(w,1)
			if(w[j] > rho)
				push!(v,w[j])
				rho = rho + (w[j] - rho)/size(v,1)
			end
		end
	end
	
	numDeleted::Int64 = 1
	
	while(numDeleted > 0)
		initialCard::Int64 = size(v,1)
		numDeleted = 0
		for j = 1:initialCard
			if(v[j-numDeleted] <= rho)
				rho = rho + (rho - v[j-numDeleted])/(size(v,1)-1)
				deleteat!(v,j-numDeleted)
				numDeleted += 1
			end
		end
	end
	

	y_proj = zeros(Float64,numVariables)
	for j = 1:numVariables
		y_proj[j] = max(0.0,y[j] - rho)
	end

	return y_proj
end


function solveInnerProblem(y_ref::Array{Float64,1},lambda::Float64,objectiveBound::Float64)
	
	y = deepcopy(y_ref)
	
	numerators = zeros(Float64,numVariables)
	denominators = zeros(Float64,numVariables)
	for j = 1:numVariables
		numerators[j] = y[j]/(1.0+2*abs2(stdev[j])*lambda)
		denominators[j] = 1.0/(1.0+2*abs2(stdev[j])*lambda)
	end

	v = Array{Float64}(1)
	den_v = Array{Int64}(1)
	v[1] = y[1]
	den_v[1] = 1
	w = Array{Float64}(0)
	den_w = Array{Float64}(0)
	nr::Float64 = numerators[1]
	dr::Float64 = denominators[1]
	rho::Float64 = (nr-1.0)/dr
	
	for j = 2:numVariables
		if(y[j] > rho)
			nr += numerators[j]
			dr += denominators[j]
			rho = (nr-1.0)/dr
			if(rho > (numerators[j]-1.0)/denominators[j])
				push!(v,y[j])
				push!(den_v,j)
			else
				append!(w,v)
				append!(den_w,den_v)
				v = [y[j]]
				den_v = [j]
				nr = numerators[j]
				dr = denominators[j]
				rho = (nr-1.0)/dr
			end
		end
	end
	
	if(size(w,1) > 0)
		for j = 1:size(w,1)
			if(w[j] > rho)
				push!(v,w[j])
				push!(den_v,den_w[j])
				nr += numerators[den_w[j]]
				dr += denominators[den_w[j]]
				rho = (nr-1.0)/dr
			end
		end
	end
	
	numDeleted::Int64 = 1
	
	while(numDeleted > 0)
		initialCard::Int64 = size(v,1)
		numDeleted = 0
		for j = 1:initialCard
			if(v[j-numDeleted] <= rho)
				nr -= numerators[den_v[j-numDeleted]]
				dr -= denominators[den_v[j-numDeleted]]
				rho = (nr-1.0)/dr
				deleteat!(v,j-numDeleted)
				deleteat!(den_v,j-numDeleted)
				numDeleted += 1
			end
		end
	end
	

	y_proj = zeros(numVariables)
	for j = 1:numVariables
		y_proj[j] = max(0.0, (y_ref[j] - rho)/(1.0 + 2*abs2(stdev[j])*lambda))
	end

	residual::Float64 = norm(sigma*y_proj) - sqrt(objectiveBound)

	return rho, residual
end


function projectOntoFeasibleSet(y_ref::Array{Float64},objectiveBound::Float64,proj_params::Array{Float64})

	y_test = projectOntoSimplex(y_ref)
	lambda::Float64 = 0.0
	residual_test::Float64 = norm(sigma*y_test) - sqrt(objectiveBound)
	if(residual_test <= 0.0)
		return y_test, proj_params
	end

	lambda_tol::Float64 = 1E-08
	lambda_low::Float64 = proj_params[1]
	lambda_up::Float64 = proj_params[2]
	residual_tol::Float64 = 1E-09
	~, residual_up::Float64 = solveInnerProblem(y_ref,lambda_up,objectiveBound)
	~, residual_low::Float64 = solveInnerProblem(y_ref,lambda_low,objectiveBound)
	while(residual_up > 0.0 || residual_low < 0.0)
		if(residual_up > 0.0)
			lambda_low = lambda_up
			lambda_up *= 2.0
			~, residual_up = solveInnerProblem(y_ref,lambda_up,objectiveBound)
		else
			lambda_up = lambda_low
			lambda_low /= 2.0
			~, residual_low = solveInnerProblem(y_ref,lambda_low,objectiveBound)
		end
	end
	
	while(lambda_up - lambda_low > lambda_tol*lambda_up)
		lambda = (lambda_low + lambda_up)/2.0
		~, residual::Float64 = solveInnerProblem(y_ref,lambda,objectiveBound)
		if(abs(residual) <= residual_tol)
			break
		elseif(residual > 0.0)
			lambda_low = lambda
		else
			lambda_up = lambda
		end
	end
	
	mu::Float64, ~ = solveInnerProblem(y_ref,lambda,objectiveBound)
	y_proj = zeros(Float64,numVariables)
	for j = 1:numVariables
		y_proj[j] = max(0.0, (y_ref[j]-mu)/(1.0+2*abs2(stdev[j])*lambda))
	end
	
	err_net::Float64 = max(maximum(-y_proj),abs(sum(y_proj)-1),norm(sigma*y_proj)-sqrt(objectiveBound))
	if(err_net > 1E-06)
		println("WARNING! PROJECTION STEP MAY NOT BE ACCURATE!!!")
	end
	
	if(proj_params[3] > 0.5)
		if(lambda > lambda_tol)
			lambda_factor::Float64 = sqrt(2.0)
			proj_params[1] = lambda/lambda_factor
			proj_params[2] = lambda*lambda_factor
		else
			proj_params[1] = 0.0
			proj_params[2] = 1.0
		end
	end
	
	return y_proj, proj_params
end


# get the stochastic gradient of the approximation
function getStochasticGradient(x::Array{Float64},numSamples::Int64,mu::Float64,tau::Float64,scalings::Float64)

	xi = generateRandomSamples(numSamples)

	stochgrad = zeros(numVariables)
	
	for samp = 1:numSamples
		con::Float64 = evaluateConstraint(x,xi[:,samp],scalings)
		
		grad_con = evaluateConstraintGradients(x,xi[:,samp],scalings)
		
		baseFactor1::Float64 = tau/((sqrt(mu)*exp(0.5*tau*con) + sqrt(1.0/mu)*exp(-0.5*tau*con))^2)
		
		stochgrad += baseFactor1*grad_con/numSamples
	end
	
	return stochgrad
end


# get the stochastic gradient of the approximation
function getStochasticGradientForFixedSample(x::Array{Float64},xi::SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},mu::Float64,tau::Float64,scalings::Float64)

	numSamples::Int64 = size(xi,2)

	stochgrad = zeros(numVariables)
	
	for samp = 1:numSamples
		con::Float64 = evaluateConstraint(x,xi[:,samp],scalings)
		
		grad_con = evaluateConstraintGradients(x,xi[:,samp],scalings)
		
		baseFactor1::Float64 = tau/((sqrt(mu)*exp(0.5*tau*con) + sqrt(1.0/mu)*exp(-0.5*tau*con))^2)
		
		stochgrad += baseFactor1*grad_con/numSamples
	end
	
	return stochgrad

end


# estimate constraint scalings
function estimateScalings(x::Array{Float64},numSamples::Int64)

	scaling_tol::Float64 = 1E-06
	
	xi = generateRandomSamples(numSamples)
	
	constraint_values = zeros(numSamples)
	for samp = 1:numSamples
		constraint_values[samp] = evaluateConstraint(x,xi[:,samp])
	end
	
	scalings::Float64 = median(abs.(constraint_values))
	
	scalings = max(scalings,scaling_tol)
	
	return scalings
end


# estimate Lipschitz constant of gradient of approximation
function estimateCompositeLipschitzConstant(x_ref::Array{Float64},numSamplesForLipVarEst::Int64,numGradientSamples::Int64,batchSize::Int64,mu::Float64,tau::Float64,scalings::Float64,objectiveBound::Float64,proj_params::Array{Float64})
	
	sample_radius::Float64 = norm(x_ref)/10.0
	
	xi = generateRandomSamples(batchSize)

	L_avg::Float64 = 0.0
	for batch = 1:batchSize	
		xi_start::Int64 = batch
		xi_end::Int64 = batch
		
		L_max::Float64 = 0.0
		for samp = 1:numSamplesForLipVarEst
			x_1 = pickPointOnSphere(x_ref,sample_radius)
			x_1,~ = projectOntoFeasibleSet(x_1,objectiveBound,proj_params)
			
			x_2 = pickPointOnSphere(x_ref,sample_radius)
			x_2,~ = projectOntoFeasibleSet(x_2,objectiveBound,proj_params)
		
			@views grad_1 = getStochasticGradientForFixedSample(x_1,xi[:,xi_start:xi_end],mu,tau,scalings)
			@views grad_2 = getStochasticGradientForFixedSample(x_2,xi[:,xi_start:xi_end],mu,tau,scalings)
			
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
							scalings::Float64,objectiveBound::Float64)
	
	proj_params = zeros(Float64,3)
	proj_params[2] = 1.0
	
	sample_radius::Float64 = norm(x_ref)/10.0
	
	rho_est::Float64 = estimateCompositeLipschitzConstant(x_ref,numSamplesForLipVarEst,numGradientSamples,batchSize,mu,tau,scalings,objectiveBound,proj_params)
	
	variance_est::Float64 = 0.0
	for samp = 1:numSamplesForLipVarEst
		x_1 = pickPointOnSphere(x_ref,sample_radius)
		x_1,~ = projectOntoFeasibleSet(x_1,objectiveBound,proj_params)
	
		xi = generateRandomSamples(numGradientSamples*batchSize)

		variance_1::Float64 = 0.0
		for batch = 1:batchSize
			xi_start::Int64 = (batch-1)*numGradientSamples + 1
			xi_end::Int64 = batch*numGradientSamples
		
			@views grad_1 = getStochasticGradientForFixedSample(x_1,xi[:,xi_start:xi_end],mu,tau,scalings)
			
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
