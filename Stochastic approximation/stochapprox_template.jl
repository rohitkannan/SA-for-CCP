
#*========= MODEL INFORMATION ==========
const model = "portfolio"

# number of replicates of the overall algorithm
const numTrials = 10

const modelFile = model * "/" * model * "_stochmodel_template.jl"

include(modelFile) # parameters of the model
#*==============================================

# directory name for storing results
const workingDir = "C:/Users/rkannan/Desktop/SA for CCP/Stochastic approximation/"
const baseDirName = workingDir * "experiments/" * model * "_stochapprox/"
mkpath(baseDirName)

#*========= PRINT OPTIONS ==============
const storeResults = true
const printIterationStats = false
const riskLevelPrecision = 6
const printScalings = true

const objFile = "obj_bound.txt"
const solnFile = "solution.txt"
const riskFile = "riskLevel.txt"
const scalingsFile = "scalings.txt"
const timeFile = "solutionTime.txt"
const projTimeFile = "projectionTime.txt"
const riskTimeFile = "riskTime.txt"
const gradTimeFile = "gradTime.txt"
const totalTimeFile = "total_time.txt"
#*======================================

#*========= INITIALIZATION OPTIONS ==========
# use previous soln as initial guess in next run of opt loop?
const usePrevSolnInOptLoop = true 
#*===========================================

#*========== ALGORITHMIC PARAMETERS ==========
# mini-batch size (m)
const minibatchSize = 20 

# max. number of iterations per replicate (N)
const maxNumIterations = 1000 

# number of runs of the optimization loop (S)
const minNumReplicates = 10
const maxNumReplicates = 50 

# improvement requirement for considering another replicate
const impReqForTerm = 1E-04
const impReqForDecrStepSize = 1E-01
const termCheckPeriod = 5

# sigmoid parameters: sigmoid(z) = mu/(mu + exp(-tau*z))
# NOTE: above notation differs from the paper
const mu = 1.0
const tau = Float64[1.0,10.0,100.0]

# should we scale the chance constraints at each objective bound iteration?
const numSamplesForScaling = 10000
const scaleAtEachIter = true

# step size modification parameters
const stepLengthCheckIter = 3
const stepLengthIncrFactor = 10.0
const stepLengthDecrFactor = 10.0

#*===========================================

#*========== OTHER ALGORITHMIC CONSTANTS ============
const numSamplesForLipVarEst = 200
const batchSize = 20
#*===================================================


println("")
println("***** Starting the stochastic approximation algorithm for model ",model)
println("")



for trial = 1:numTrials


println("=======================")
println("=======================")
println("Trial ",trial,"/",numTrials)
println("=======================")
println("=======================")


# get total time
tic()

# get preprocessing time
tic()

# compute initial risk level
user_risk_level::Float64 = computeRiskLevel(user_initial_guess)
println("user_risk_level: ",round.(user_risk_level,riskLevelPrecision))

proj_params = ones(Float64,3)
proj_params[1] = 0.0

# project initial guess onto feasible set
initial_guess,~ = projectOntoFeasibleSet(user_initial_guess,initialObjValue,proj_params)

# initial estimate of constraint scalings
scalings = estimateScalings(initial_guess,numSamplesForScaling)
if(printScalings)
	if(size(scalings,1) == 1)
		println("scalings: ",round.(scalings,4))
	else
		print("max scaling: ",round.(maximum(scalings),4),"  min scaling: ",round.(minimum(scalings),4))
	end
end


#*============ ESTIMATE STEP LENGTH =====================
stepLength = zeros(Float64,size(tau,1))

for iter_t = 1:size(tau,1)

	println("")
	@printf "Tau iteration # %d/%d  with tau = %3.2f" iter_t size(tau,1) tau[iter_t]
	
	tic()

	if(iter_t == 1)
		stepLength[iter_t] = estimateStepLength(initial_guess,minibatchSize,numSamplesForLipVarEst,
										batchSize,maxNumIterations,minNumReplicates,mu,tau[iter_t],
										scalings,initialObjValue)
	else
		stepLength[iter_t] = stepLength[1]*((tau[1])^2)/((tau[iter_t])^2)
	end

	stepLengthTime = toq()

	@printf "  stepLength: %0.4e  in time: %3.2f \n" stepLength[iter_t] stepLengthTime
end

println("")

# store the computed step lengths separately
const initialStepLength = deepcopy(stepLength)

#*==============================================================================

preprocessingTime = toq()


#*========= STORE RESULTS ==========
if(storeResults)

	subDirName = baseDirName * "rep" * string(trial) * "/"
	mkpath(subDirName)
	
	# write details to text file
	details_file = subDirName * model * ".txt"
	open(details_file, "w") do f
		write(f,"Model: $model \n")
		write(f,"Mini-batch size (m): $minibatchSize \n")
		write(f,"Max. number of iterations (N): $maxNumIterations \n")
		write(f,"Max. number of replicates (S): $maxNumReplicates \n")
		write(f,"Min. number of replicates: $minNumReplicates \n")
		write(f,"risk level lower bound: $riskLowerBound \n")
		write(f,"mu: $mu \n")
		write(f,"tau: $tau \n")
		write(f,"warm start replicate solution?: $usePrevSolnInOptLoop \n")
		write(f,"improvement requirement for termination: $impReqForTerm \n")
		write(f,"non-improvement requirement for decreasing stepLength: $impReqForDecrStepSize \n")
		write(f,"number of samples for scaling: $numSamplesForScaling \n")
		write(f,"scale at each iteration?: $scaleAtEachIter \n")
		write(f,"initial step length: $initialStepLength")
		write(f,"check for increasing step length every $stepLengthCheckIter iterations \n")
		write(f,"step length increase factor: $stepLengthIncrFactor \n")
		write(f,"step length decrease factor: $stepLengthDecrFactor \n")
		write(f,"number of samples for estimating Lipschitz constant and variance: $numSamplesForLipVarEst \n")
		write(f,"batch size for Lipschitz and variance estimates: $batchSize \n")
		write(f,"preprocessing time: $preprocessingTime \n")
	end	
end
#*==================================


#*========= STORE RESULTS ==========
if(storeResults)

	# write data to text file
	scalings_file = subDirName * scalingsFile
	open(scalings_file, "w") do f
	end

	obj_file = subDirName * objFile
	open(obj_file, "w") do f
	end

	riskLevel_file = subDirName * riskFile
	open(riskLevel_file, "w") do f
	end

	soln_file = subDirName * solnFile
	open(soln_file, "w") do f
	end

	time_file = subDirName * timeFile
	open(time_file, "w") do f
	end

	grad_time_file = subDirName * gradTimeFile
	open(grad_time_file, "w") do f
	end

	proj_time_file = subDirName * projTimeFile
	open(proj_time_file, "w") do f
	end

	obj_time_file = subDirName * riskTimeFile
	open(obj_time_file, "w") do f
	end
	
end
#*==================================

# store the best solution for each objective bound
best_solution = zeros(Float64,numVariables) 
best_risk_level = Inf
for i = 1:numVariables
	best_solution[i] = user_initial_guess[i]
end


objIterCount::Int64 = 0

if(abs(objGap) < 1E-09)
	error("Objective spacing for EF is almost zero! Reset manually.")
end


while true

	objIterCount += 1
	currentObjBound::Float64 = initialObjValue + (objIterCount - 1)*objGap

	tic()

	println("")
	println("Objective bound iteration # ",objIterCount," with objectiveBound = ",currentObjBound)
	
	
	# reset best found objective value
	best_risk_level = Inf
	
	
	solutionTime::Float64 = 0
	gradTime::Float64 = 0
	projTime::Float64 = 0
	objTime::Float64 = 0
	otherTime::Float64 = 0

for iter_t = 1:size(tau,1) # loop over the smoothing parameter values

	println("")
	println("Tau iteration # ",iter_t, "/",size(tau,1)," with tau = ",tau[iter_t])
	println("")

	#*========= INITIAL GUESS ==========
	initial_guess, proj_params = projectOntoFeasibleSet(initial_guess,currentObjBound,proj_params)
	#*==================================

	
	# estimate initial risk level
	initial_risk_level::Float64 = computeRiskLevel(initial_guess)
	println("initial_risk_level: ",round.(initial_risk_level,riskLevelPrecision)) 

	if(initial_risk_level < best_risk_level)
		best_risk_level = initial_risk_level
		for i = 1:numVariables
			best_solution[i] = initial_guess[i]
		end
	end
	

	if(scaleAtEachIter && (iter_t == 1))
		tic()
		scalings = estimateScalings(initial_guess,numSamplesForScaling)
		scalingTime::Float64 = toq()
		if(printScalings)
			if(size(scalings,1) == 1)
				println("scalings: ",round.(scalings,4),"  time: ",round.(scalingTime,2))
			else
				print("max scaling: ",round.(maximum(scalings),4),"  min scaling: ",round.(minimum(scalings),4),"  time: ",round.(scalingTime,2))
			end
		end
	end


	#*========= STORE RESULTS ==========
	if(storeResults && (iter_t == 1))
		open(obj_file, "a") do f
			write(f,"$(currentObjBound) \n")
		end

		open(scalings_file, "a") do f
			write(f,"$scalings \n")
		end
	end
	#*==================================

	# store the final iterate and objective in each run
	solution = zeros(Float64,maxNumReplicates,numVariables) 
	riskLevel_soln = Inf*ones(Float64,maxNumReplicates)

	
	tic()
	
	numActualReplicates::Int64 = 0
	numStepSizeIncr::Int64 = 0
	numStepSizeDecr::Int64 = 0

	
	stepLength[iter_t] = initialStepLength[iter_t]
	
	local_best_risk_level::Float64 = best_risk_level
	local_best_risk_level2::Float64 = best_risk_level

	
	#*========= OPTIMIZATION LOOP ==========
	for iter_S = 1:maxNumReplicates
	
		tic()
		iterSolnTime::Float64 = Inf
		iterObjTime::Float64 = 0
		iterGradTime::Float64 = 0
		iterProjTime::Float64 = 0
		iterOtherTime::Float64 = 0
	
		if(printIterationStats)
			println("=====================================")
			println("Optimization iteration # ", iter_S)
		end
		
		# determine actual number of SGD iterations
		uniformRand = rand(1)
		numIterations::Int64 = ceil.(uniformRand[1]*maxNumIterations)
		
		if(printIterationStats)
			println("Number of iterations: ",numIterations)
		end
		
			
		#*========= SET INITIAL GUESS ==========
		x = zeros(Float64,numVariables)
		if(iter_S == 1 || usePrevSolnInOptLoop == false)
			for i = 1:numVariables
				x[i] = initial_guess[i]
			end
		else
			for i = 1:numVariables
				x[i] = solution[iter_S-1,i]
			end
		end
		#*======================================
	
		#*========= SGD LOOP ==========
		for t = 1:numIterations

			tic()
		
			#*==== COMPUTE STOCHASTIC GRADIENT =====
			grad = getStochasticGradient(x,minibatchSize,mu,tau[iter_t],scalings)
			#*======================================
			
			iterGradTime += toq()
			
			#*==== SGD STEP =====
			x = x - stepLength[iter_t]*grad
			#*===================
			
			tic()
			
			#*==== PROJECTION STEP =====
			x, proj_params = projectOntoFeasibleSet(x,currentObjBound,proj_params)
			#*==========================
		
			iterProjTime += toq()

		end # SGD iterations
		#*========= END SGD LOOP ==========
			
		#*==== ESTIMATE RISK LEVEL =====
		tic()
		
		# evaluate true objective value
		est_risk_level::Float64 = computeRiskLevel(x)
		
		iterObjTime += toq()
		#*====================================
			
		# store the final iterate and objective
		riskLevel_soln[iter_S] = est_risk_level
		for j = 1:numVariables
			solution[iter_S,j] = x[j]
		end

		iterSolnTime = toq()
		iterOtherTime = iterSolnTime - iterObjTime - iterGradTime - iterProjTime

		#*========= STORE RESULTS ==========
		if(storeResults)
			# create folder for storing results
			dirName = subDirName * string(objIterCount) * "/" * string(iter_t) * "/"
			mkpath(dirName)
			
			# write true final objective to text file
			est_risk_level_file = dirName * "riskLevels.txt"
			open(est_risk_level_file, "a") do f
				write(f,"$est_risk_level \n")
			end
			# write solution times to text file
			solution_time_file = dirName * "solution_time.txt"
			open(solution_time_file, "a") do f
				write(f,"$iterSolnTime \n")
			end
			gradient_time_file = dirName * "gradient_time.txt"
			open(gradient_time_file, "a") do f
				write(f,"$iterGradTime \n")
			end
			projection_time_file = dirName * "projection_time.txt"
			open(projection_time_file, "a") do f
				write(f,"$iterProjTime \n")
			end
			risk_time_file = dirName * "risk_time.txt"
			open(risk_time_file, "a") do f
				write(f,"$iterObjTime \n")
			end
		end
		#*==================================

		if(printIterationStats)
			println("FINAL RISK LEVEL: ",round.(est_risk_level,riskLevelPrecision))
			println("SOLN TIME: ",round.(iterSolnTime,2)," GRAD TIME: ",round.(iterGradTime,2)," PROJ TIME: ",round.(iterProjTime,2)," OBJ TIME: ", round.(iterObjTime,2)," OTHER TIME: ", round.(iterOtherTime,2))
		end

		# update overall times
		solutionTime += iterSolnTime
		gradTime += iterGradTime
		projTime += iterProjTime
		objTime += iterObjTime
		otherTime += iterOtherTime
		
		numActualReplicates = iter_S
		
		if(iter_S >= minNumReplicates)
			improvements = zeros(Float64,termCheckPeriod)
			for j = 1:termCheckPeriod
				improvements[j] = local_best_risk_level - riskLevel_soln[iter_S+1-j]
			end
			
			localImpReq::Float64 = impReqForTerm*local_best_risk_level
			max_improvement::Float64 = maximum(improvements)
			
			if(max_improvement < localImpReq)
				break
			end
		end
		
		
		if(rem(numActualReplicates,stepLengthCheckIter) == 0)
			improvements = zeros(Float64,stepLengthCheckIter)
			for j = 1:stepLengthCheckIter
				improvements[j] = local_best_risk_level2 - riskLevel_soln[iter_S+1-j]
			end
			
			localImpReq1::Float64 = impReqForTerm*local_best_risk_level2
			localImpReq2::Float64 = impReqForDecrStepSize*local_best_risk_level2
			
			max_improvement = maximum(improvements)
			
			if(max_improvement > -localImpReq1)
				stepLength[iter_t] *= stepLengthIncrFactor
				numStepSizeIncr += 1
			elseif(max_improvement < -localImpReq2)
				stepLength[iter_t] /= stepLengthDecrFactor
				numStepSizeDecr += 1
			end
		end
		
		if(iter_S > termCheckPeriod)
			if(riskLevel_soln[iter_S-termCheckPeriod] < local_best_risk_level)
				local_best_risk_level = riskLevel_soln[iter_S-termCheckPeriod]
			end
		end

		if(iter_S > stepLengthCheckIter)
			if(riskLevel_soln[iter_S-stepLengthCheckIter] < local_best_risk_level2)
				local_best_risk_level2 = riskLevel_soln[iter_S-stepLengthCheckIter]
			end
		end

	end # S
	#*========= END OPTIMIZATION LOOP ==========

	min_risk_level::Float64 = Inf
	best_obj_index::Int64 = -1

	#*========= LOOP OVER COMPUTED SOLUTIONS ==========
	for iter_S = 1:maxNumReplicates

		#*==== FIND MIN. OBJECTIVE =====
		if(riskLevel_soln[iter_S] < min_risk_level)
			min_risk_level = riskLevel_soln[iter_S]
			best_obj_index = iter_S
		end
		#*============================================

	end # S
	#*=================================================

	
	totalTime = toq()
	
	
	if(riskLevel_soln[best_obj_index] < best_risk_level)
		best_risk_level = riskLevel_soln[best_obj_index]
		for j = 1:numVariables
			best_solution[j] = solution[best_obj_index,j]
		end
	end

	println("")
	println("Actual number of replicates: ",numActualReplicates)
	println("Number of step size incr/decr: ",numStepSizeIncr,"/",numStepSizeDecr)
	println("Step length change factor: ",stepLength[iter_t]/initialStepLength[iter_t])
	println("Lowest two-phase risk level: ",round.(min_risk_level,riskLevelPrecision))
	@printf "TOTAL TIME: %3.1f \n" totalTime
	println("")
	
	#*========= END POST-OPTIMIZATION PHASE ==========
	
	for j = 1:numVariables
		initial_guess[j] = best_solution[j]
	end

end #tau

	#*========= STORE RESULTS ==========
	if(storeResults)
		# write risk level to file
		riskLevel_file = subDirName * riskFile
		open(riskLevel_file, "a") do f
			write(f,"$best_risk_level \n")
		end

		# write solution to text file
		soln_file = subDirName * solnFile
		open(soln_file, "a") do f
			for j = 1:numVariables
				write(f,"$(best_solution[j]) ")
			end
			write(f,"\n")
		end
	end
	#*==================================
	
	iterTime::Float64 = toq()
	
	println("")
	println("OBJECTIVE BOUND ITERATION # ",objIterCount," with objectiveBound = ",currentObjBound,"  best risk level: ",round.(best_risk_level,riskLevelPrecision),"  in time: ",iterTime)
	println("")

	#*========= STORE RESULTS ==========
	if(storeResults)
		open(time_file, "a") do f
			write(f,"$iterTime \n")
		end
		
		open(grad_time_file, "a") do f
			write(f,"$gradTime \n")
		end
		
		open(proj_time_file, "a") do f
			write(f,"$projTime \n")
		end
		
		open(obj_time_file, "a") do f
			write(f,"$objTime \n")
		end
	end
	#*==================================
	
	if(best_risk_level < riskLowerBound)
		break
	end
	
	gc()
	
end # objectiveBound

totalElapsedTime = toc()

#*========= STORE RESULTS ==========
if(storeResults)
	# write elapsed time to text file
	total_time_file = subDirName * totalTimeFile
	open(total_time_file, "w") do f
		write(f,"$totalElapsedTime")
	end
end
#*==================================


sleep(30)

end # trials