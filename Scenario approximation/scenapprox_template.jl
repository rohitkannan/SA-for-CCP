
const modelName = "resource"

const modelFile = modelName * "/" * modelName * "_scenmodel_template.jl"

# directory name for storing results
const workingDir = "C:/Users/rkannan/Desktop/SA for CCP/Scenario approximation/"
const baseDirName = workingDir * "experiments/" * modelName * "_scenapprox/"
mkpath(baseDirName)


#*========= INCLUDE MODEL FILES ================
include(modelFile) # parameters of the model
#*==============================================


const details_file = baseDirName * modelName * ".txt"
initializeFiles(details_file)


# loop over the sample sizes
for iter = 1:size(NumSamples,1)

	println("")
	println("***************************************************")
	println("Iteration #: ",iter,"/",size(NumSamples,1),"  with NumSamples = ",NumSamples[iter])
	println("***************************************************")
	println("")

	numScenarios::Int64 = NumSamples[iter]

	# create folder for storing results
	dirName::String = baseDirName * string(iter) * "/"
	mkpath(dirName)

	# write num samples to text file
	samples_file::String = dirName * "numScenarios.txt"
	open(samples_file, "w") do f
		write(f,"$numScenarios \n")
	end

	#####
	# run as many replicates as needed
	for rep = 1:NumReplicates

		tic()

		@printf "Replicate #: %d" rep		

		solveScenarioApproximationModel(numScenarios,dirName)
		
		gc()

		println("")
		
		elapsed_time::Float64 = toq()

		sleep(5.0+0.1*elapsed_time)

	end # for replicates
	#####
	
	sleep(5.0)
	
end # for samples
