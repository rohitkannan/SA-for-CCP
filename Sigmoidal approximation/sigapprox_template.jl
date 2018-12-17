
const modelName = "portfolio"

const modelFile = modelName * "_sigmodel_template.jl"

# directory name for storing results
const workingDir = "C:/Users/rkannan/Desktop/SA for CCP/Sigmoidal approximation/"
const baseDirName = workingDir * "experiments/" * modelName * "_sigapprox/"
mkpath(baseDirName)


#*========= INCLUDE MODEL FILES ================
include(modelFile) # parameters of the model
#*==============================================

const details_file = baseDirName * modelName * ".txt"
initializeFiles(details_file)


# loop over the sample sizes
for iter = 1:size(NumSamplesFactor,1)

	riskLevel::Float64 = riskLevels[1]
	
	numScenarios::Int64 = round(NumSamplesFactor[iter]/riskLevel)

	println("")
	println("***************************************************")
	println("Iteration #: ",iter,"/",size(NumSamplesFactor,1),"  with riskLevels = ",riskLevel," and numScenarios = ",numScenarios)
	println("***************************************************")
	println("")

	# create folder for storing results
	dirName::String = baseDirName * string(iter) * "/"
	mkpath(dirName)

	# write num samples and risk level to text file
	risk_file::String = dirName * "specifiedRiskLevel.txt"
	open(risk_file, "w") do f
		write(f,"$riskLevel \n")
	end
	
	scenario_file::String = dirName * "numScenarios.txt"
	open(scenario_file, "w") do f
		write(f,"$numScenarios \n")
	end
	
	#####
	# run as many replicates as needed
	for rep = 1:NumReplicates

		tic()

		println("REPLICATE #",rep)		

		solveSigmoidalApproximationModel(riskLevel,numScenarios,dirName)
		
		gc()

		println("")
		
		elapsed_time::Float64 = toq()

		sleep(min(30,1.0+0.1*elapsed_time))

	end # for replicates
	#####
	
	sleep(5.0)
	
end # for risk levels
