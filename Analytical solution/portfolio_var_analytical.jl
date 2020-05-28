using JuMP, Gurobi, StatsFuns

# problem parameters
NumAssets = [1000] # number of variables minus one
returnLBD = 1.2 #lower bound on the return
epsilon = logspace(-4.5,-0.3011,50)
epsilon = reverse(epsilon) # constraint violation tolerance

# iterate over the number of assets
for ass = 1:size(NumAssets,1)

	# random variable distribution parameters
	mean = Array{Float64}(NumAssets[ass])
	stdev = Array{Float64}(NumAssets[ass])
	for i = 1:NumAssets[ass]
		mean[i] = 1.05 + 0.3*(NumAssets[ass] - i)/(NumAssets[ass]-1)
		stdev[i] = (0.05 + 0.6*(NumAssets[ass] - i)/(NumAssets[ass]-1))/3
	end

	# matrix with stdevs on the diagonal
	sigma = zeros(Float64, NumAssets[ass], NumAssets[ass])
	for i = 1:NumAssets[ass]
		sigma[i,i] = stdev[i]
	end

	# iterate over the epsilons
	for iter_e = 1:size(epsilon,1)
	
		println("Iter: (",ass,",",iter_e,")  ","NumAssets: ",NumAssets[ass],"  epsilon: ",epsilon[iter_e])

		# create folder for storing results
		dirName = "C:/Users/rkannan/Desktop/SA for CCP/portfolio_var_analytical/" * 
					string(NumAssets[ass]) * "/" * string(iter_e) * "/"
		mkpath(dirName)

		# write details of case study to text file
		details_file = dirName * "portfolio1.txt"
		open(details_file, "w") do f
			write(f,"Model: portfolio1 \n")
			write(f,"Method: EXACT \n")
			write(f,"Number of assets: $(NumAssets[ass]) \n")
			write(f,"epsilon: $(epsilon[iter_e]) \n")
			write(f,"return lower bound: $returnLBD \n")
		end
		# write epsilon to text file
		epsilon_file = dirName * "epsilon.txt"
		open(epsilon_file, "w") do f
			write(f,"$(epsilon[iter_e]) \n")
		end

		# pass params as keyword arguments to GurobiSolver
		m = Model(solver=GurobiSolver(Presolve=0,OutputFlag=0))

		@variable(m, objvar)
		@variable(m, y[1:NumAssets[ass]] >= 0)
		

		@objective(m, Min, objvar)
		@constraint(m, objvar >= 100.0*norm(sigma*y))
		@constraint(m, sum(y[i] for i = 1:NumAssets[ass]) == 1)
		@constraint(m, sum(mean[i]*y[i] for i = 1:NumAssets[ass]) >= returnLBD + norminvcdf(1-epsilon[iter_e])*norm(sigma*y))

		status = solve(m)
		
		objvar_opt = abs2(getvalue(objvar)/100.0)
		time_to_opt = getsolvetime(m)
		
		y_opt = getvalue(y)
		soln_file = dirName * "soln.txt"
		open(soln_file, "w") do f
			for i = 1:NumAssets[ass]
				write(f,"$(y_opt[i]) ")
			end
		end
		
		# write objective
		obj_file = dirName * "objvals.txt"
		writeMode = "w"
		open(obj_file, writeMode) do f
			write(f,"$objvar_opt \n")
		end
		# write solution time
		time_file = dirName * "solntime.txt"
		open(time_file, writeMode) do f
			write(f,"$time_to_opt \n")
		end

		println("Objective: ",objvar_opt,"  Time: ",time_to_opt)
		println("")

		sleep(2)
		
	end

	sleep(5)
end
