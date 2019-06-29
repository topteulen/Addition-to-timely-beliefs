import numpy as np
import csv
import matplotlib.pyplot as plt
from math import sin

def cal_pop_fitness(equation_inputs, pop, real_value):
	"""
	Calculates fitness of a given population.
	@param list of floats : Data used to calcuate the function
	@param list of lists of floats : weights for each individual of the population
	@param list of floats: the result values.
	@return list of negative floats : fitness of each individual in the population
	"""
	#Setting input to x and w.
	fitness = []
	w = pop
	x = equation_inputs
	#Calculating fitness score for each individual.
	for i in range(len(pop)):
		fitness += [[]]
		for j in range(len(equation_inputs)-48):
			guess = 1/6*x[j][0]*w[i][0] + 1/6*x[j][1]*w[i][1] + 1/6*x[j][2]*w[i][2] + 1/6*x[j][3]*w[i][3] + w[i][4]*np.sin((1/3.82)*x[j][4]+w[i][5]) + -(1/w[i][6])*(x[j][5]-6)**2+w[i][7]
			fitness[i] +=  [-((guess - float(real_value[j+48]))**2)]
	#Make it a mean score.
	fitness = np.mean(fitness, axis=1)
	return fitness

def select_mating_pool(pop, fitness, num_parents):
	"""
	Selects the most fit parents for mating.
	@param list of lists of floats : weights for each individual of the population
	@param list of negative floats : fitness of each individual in the population
	@param int : amount of parents that get selected
	@return list of lists of floats : weights for each individual of the selected parents.
	"""
	#Make parent array.
	parents = np.empty((num_parents, pop.shape[1]))
	#Select highest fitness parents in the population.
	for parent_num in range(num_parents):
		max_fitness_idx = np.argmax(fitness)
		parents[parent_num] = pop[max_fitness_idx]
		fitness[max_fitness_idx] = -99999999999
	return parents

def crossover(parents, offspring_size,num_weights):
	"""
	Takes the genes of the selected parents and mixes them together. Also know as sex.
	@param list of lists of floats : weights for each individual of the selected parents.
	@param int: amount of children to create.
	@param int: amount of genes each parent has.
	@return list of lists of floats :  weights for each individual of the born children.
	"""
	#Make offspring array.
	offspring = np.empty((offspring_size,num_weights))
	#Point where genes are split.
	crossover_point = int(num_weights/2)
	#Add gene from one parent to the other.
	for i in range(offspring_size):
		#Get parents index.
		index_1 = i%parents.shape[0]
		index_2 = (i+1)%parents.shape[0]
		#Split and merge parents.
		offspring[i][:crossover_point] = parents[index_1][:crossover_point]
		offspring[i][crossover_point:] = parents[index_2][crossover_point:]
	return offspring

def mutation(offspring,generation,mutation_prob):
	"""
	Changes the values of the genes of the born children randomly.
	@param list of lists of floats :  weights for each individual of the born children.
	@param int: current generation.
	@param float where 0.0 <= x <= 1.0 is True: probability of a mutation
	@return list of lists of floats :  weights for each individual of the born children.
	"""
	random_list = []
	# Mutation changes a single gene in each offspring randomly.
	for i in range(offspring.shape[0]):
		for j in range(len(offspring[i])):
			#Chance for a single gene to mutate.
			if np.random.random() <= mutation_prob:
				# The random value to be added to the gene is chosen depending on generation.
				random_value = np.random.uniform(-7.5,7.5)
				if generation > 50:
					random_value = np.random.uniform(-3.5,3.5)
				if generation > 100:
					random_value = np.random.uniform(-2.0,2.0)
				if generation > 150:
					random_value = np.random.uniform(-1.0,1.0)
				if generation > 175:
					random_value = np.random.uniform(-0.4,0.4)
				random_list += [random_value]
				offspring[i][j] = offspring[i][j] + random_value
	return offspring

def accuracy_test(real_value,horizon,result):
	"""
	Result printer/ accuracy calculating function.
	@param list of floats: actual measurements of the data.
	@param int: horizon of how far in the future is being predicted.
	@param list of floats: predicted results of the data.
	@return float: WAPE accuracy score.
	"""
	newresult = []
	newreal_value = []
	correct = 0
	correct_1 = 0
	correct_2 = 0
	for i in range(len(result)-horizon):
		#Make list for plotting every 96 quarter hours aka a day.
		if i%96 == 0:
			newresult += [round(result[i],2)]
			newreal_value += [round(float(real_value[i+horizon]),2)]
		#Calculate correct amount of values rounded on 0,1 and 2 decimals.
		if round(result[i],2) == round(float(real_value[i+horizon]),2):
			correct_2 += 1
		if round(result[i],1) == round(float(real_value[i+horizon]),1):
			correct_1 += 1
		if int(result[i]) == round(float(real_value[i+horizon]),0):
			correct += 1
	difference = 0
	real_sum = 0
	#Calculate wape score.
	for i in range(len(result)-horizon):
		difference += abs(float(real_value[i+horizon]) - result[i])
		real_sum += float(real_value[i])
	print(real_sum,difference)
	wape =  np.divide(difference,real_sum)
	print("WAPE acc =", wape)
	print('acc_0',correct/len(result)*100)
	print('acc_1',correct_1/len(result)*100)
	print('acc_2',correct_2/len(result)*100)
	#All plotting related code.
	a = plt.scatter(range(len(newresult)),newresult,color="blue")
	b = plt.scatter(range(len(newreal_value)),newreal_value,color="red")
	plt.title("Model predictions for a horizon of " + str(horizon/4) + " hours.")
	plt.xlabel('Time in days')
	plt.ylabel('Temprature in °C')
	plt.legend((a,b),('Predictions','Real values'))
	plt.savefig(str(horizon/4)+".png")
	plt.close()
	return wape

def main(horizon,num_weights = 8,sol_per_pop = 100,num_parents_mating = 40,mutation_prob = 0.9,num_generations=200):
	"""
	Main function of the genetic algorithm. This contains all the function calls
	and parameter initializations.
	@param df : BeliefsDataframe containing all necessary data.
	@param beliefSeries : BeliefSeries object.
	@param model : model to use to generate new data.
	@return float: WAPE accuracy score.
	"""
	#History_size is the size of the linear porition of the formula.
	history_size = num_weights-3
	#Pop_size is parents*genes of each parent.
	pop_size = (sol_per_pop,num_weights)
	#Offspring_size is amount of not livving parents.
	offspring_size = (sol_per_pop-num_parents_mating)
	#First initialization of the population.
	new_population = np.random.uniform(low=-10.0, high=10.0, size=pop_size)

	#Data preprocessing
	with open('energy_data.csv') as csv_file:
	    csv_reader = csv.reader(csv_file,delimiter=',')
	    data = list(csv_reader)[1:]
	data = np.array(data)
	temps = data[:,-1]
	newdata = []
	for i in range(len(data)):
		if i >= history_size*horizon+1:
			newdata += [[]]
			#Makes the linear part of the formula its data, based on the history_size and horizon chosen.
			newdata[i-(history_size*horizon+1)] += [float(data[(i-history_size*horizon+1)+j*horizon][-1]) for j in range(history_size)]
			#Last two values for months and for days.
			months = int(data[i-history_size*horizon+1][0][5:7])
			hours = int(data[i-history_size*horizon+1][0][11:13])
			mins = int(data[i-history_size*horizon+1][0][14:16])
			newdata[i-(history_size*horizon+1)] += [(hours+12+mins/60)%24]
			newdata[i-(history_size*horizon+1)] += [months]
	#Select 70% for training.
	real_value = temps[:int(len(temps)*0.7)]
	equation_inputs = newdata[:int(len(temps)*0.7)]
	#End of data preprocessing.

	for generation in range(num_generations):
		print("Generation : ", generation)
		# Measing the fitness of each chromosome in the population.
		fitness = cal_pop_fitness(equation_inputs, new_population,real_value)
		# Selecting the best parents in the population for mating.
		parents = select_mating_pool(new_population, fitness,num_parents_mating)
		# Generating next generation using crossover.
		offspring = crossover(parents,offspring_size,num_weights)
		# Adding some variations to the offsrping using mutation.
		offspring_mutation = mutation(offspring,generation,mutation_prob)
		# Creating the new population based on the parents and offspring.
		new_population[:parents.shape[0]] = parents
		new_population[parents.shape[0]:] = offspring_mutation
		# The best result in the current iteration.
		print("fitness : ", np.max(fitness))

	# Getting the best solution after iterating finishing all generations.
	fitness = cal_pop_fitness(equation_inputs, new_population,real_value)
	best_match_idx = np.argmax(fitness)
	print("Best solution : ", new_population[best_match_idx])
	print("Best solution fitness : ", fitness[best_match_idx])

	#Get test/validation data.
	x = newdata[int(len(temps)*0.7):]
	real_value = temps[int(len(temps)*0.7):]
	w = new_population[best_match_idx]
	result = []
	#Calc the model its final guesses.
	for j in range(len(x)):
		result += [1/4*x[j][0]*w[0] + 1/4*x[j][1]*w[1] + 1/4*x[j][2]*w[2] + 1/4*x[j][3]*w[3]  +w[4]*np.sin((1/3.82)*x[j][4]+w[5]) + -(1/w[6])*(x[j][5]-6)**2+w[7]]
	#Return accuracy measurement.
	return accuracy_test(w,x,real_value,horizon,result)


wape_list = []
wape_list += [main(1)]
wape_list += [main(4)]
wape_list += [main(8)]
wape_list += [main(12)]
wape_list += [main(16)]
wape_list += [main(24)]
wape_list += [main(36)]
wape_list += [main(48)]
wape_list += [main(64)]
wape_list += [main(80)]
wape_list += [main(96)]
wape_list += [main(120)]
wape_list += [main(144)]
wape_list += [main(168)]
print(wape_list)
