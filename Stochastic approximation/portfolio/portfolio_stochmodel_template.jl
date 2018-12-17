using Distributions, StatsFuns, StatsBase

include("portfolio_initialization_template.jl")

const numVariables = 1000 + 1
const numAssets = numVariables-1

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

	riskLevel::Float64 = 1.0 - normcdf((sum(mean[i]*y_opt[i] for i = 1:numAssets) - y_opt[numAssets+1])/norm(sigma*(y_opt[1:numAssets])))
	
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

	con::Float64 = (x[numVariables] - xi'*x[1:numVariables-1])/scalings

	return con
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},xi::Array{Float64})

	return evaluateConstraintGradients(x,xi,1.0)
end


# evaluate constraint gradients
function evaluateConstraintGradients(x::Array{Float64},xi::Array{Float64},scalings::Float64)

	grad_con = zeros(Float64,numVariables)
	for j = 1:numVariables-1
		grad_con[j] = -xi[j]
	end
	grad_con[numVariables] = 1
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


# project on to the set {x[1:n] >= 0, sum(x[1:n]) = 1, x[n+1] >= target}
function projectOntoFeasibleSet(z::Array{Float64},target::Float64)
	
	const n::Int64 = numVariables-1
	y = deepcopy(z)
	
	v = Array{Float64}(1)
	v[1] = y[1]
	w = Array{Float64}(0)
	rho::Float64 = y[1] - 1.0
	
	for j = 2:n
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
	

	x_proj = zeros(numVariables)
	for j = 1:n
		x_proj[j] = max(0,y[j] - rho)
	end
	x_proj[numVariables] = max(target,z[numVariables])

	return x_proj
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
function estimateCompositeLipschitzConstant(x_ref::Array{Float64},numSamplesForLipVarEst::Int64,numGradientSamples::Int64,batchSize::Int64,mu::Float64,tau::Float64,scalings::Float64,objectiveBound::Float64)
	
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
	
	sample_radius::Float64 = norm(x_ref)/10.0
	
	rho_est::Float64 = estimateCompositeLipschitzConstant(x_ref,numSamplesForLipVarEst,numGradientSamples,batchSize,mu,tau,scalings,objectiveBound)
	
	
	variance_est::Float64 = 0.0
	for samp = 1:numSamplesForLipVarEst
		x_1 = pickPointOnSphere(x_ref,sample_radius)
		x_1 = projectOntoFeasibleSet(x_1,objectiveBound)
	
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

	n::Int64 = size(x_orig,1)
	unifRand = rand(1)
	factor1::Float64 = radius*(unifRand[1])^(1.0/n)
	normRand = rand(Normal(0,1), n)
	factor2::Float64 = norm(normRand)
	
	x_samp = x_orig + factor1*normRand/factor2

	return x_samp
end
