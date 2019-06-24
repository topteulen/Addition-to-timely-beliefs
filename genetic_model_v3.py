import numpy as np
import csv
import matplotlib.pyplot as plt
from math import sin

def cal_pop_fitness(equation_inputs, pop, real_value):
	# Calculating the fitness value of each solution in the current population.
	# The fitness function caulcuates the sum of products between each input and its corresponding weight.
	fitness = []
	w = pop
	x = equation_inputs
	for i in range(len(pop)):
		fitness += [[]]
		for j in range(len(equation_inputs)):
			guess = x[j][0]*w[i][0] + x[j][1]*w[i][1] + x[j][2]*w[i][2] + x[j][3]*w[i][3] + x[j][4]*w[i][4] + x[j][5]*w[i][5] +w[i][6]*np.sin((1/3.82)*x[j][6])
			fitness[i] +=  [-((guess - float(real_value[j]))**2)]

	fitness = np.mean(fitness, axis=1)
	return fitness

def select_mating_pool(pop, fitness, num_parents):
	# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
	parents = np.empty((num_parents, pop.shape[1]))
	for parent_num in range(num_parents):
		max_fitness_idx = np.argmax(fitness)
		parents[parent_num] = pop[max_fitness_idx]
		fitness[max_fitness_idx] = -99999999999
	return parents

def crossover(parents, offspring_size,num_weights):
	offspring = np.empty((offspring_size,num_weights))

	crossover_point = int(num_weights/2)
	for i in range(offspring_size):
		index_1 = i%parents.shape[0]
		# Index of the second parent to mate.
		index_2 = (i+1)%parents.shape[0]
		# The new offspring will have its first half of its genes taken from the first parent.
		offspring[i][:crossover_point] = parents[index_1][:crossover_point]
		# The new offspring will have its second half of its genes taken from the second parent.
		offspring[i][crossover_point:] = parents[index_2][crossover_point:]
	return offspring

def mutation(offspring,generation,mutation_prob):
	random_list = []
	# Mutation changes a single gene in each offspring randomly.
	for i in range(offspring.shape[0]):
		# The random value to be added to the gene.
		for j in range(len(offspring[i])):
			if np.random.random() <= mutation_prob:
				random_value = np.random.uniform(-1.0,1.0)
				if generation > 70:
					random_value = np.random.uniform(-0.2,0.2)
				#np.random.uniform(-(generation/100)/np.sqrt(1+generation**2)-1, 1-(generation/100)/np.sqrt(1+(generation/100)**2))
				random_list += [random_value]
				offspring[i][j] = offspring[i][j] + random_value
	print(np.mean(random_list))
	return offspring




"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.
#x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 = y

# Number of the weights we are looking to optimize.
num_weights = 7

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 100
num_parents_mating = 50
num_generations = 150
mutation_prob = 0.4
pop_size = (sol_per_pop,num_weights)
offspring_size = (sol_per_pop-num_parents_mating)
new_population = np.random.uniform(low=-10.0, high=10.0, size=pop_size)

with open('energy_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    data = list(csv_reader)[1:]

data = np.array(data)
temps = data[:,-1]
history_size = num_weights
newdata = []
for i in range(len(data)):
	if i >= history_size+1:
		newdata += [[]]
		newdata[i-(history_size+1)] += [float(data[i-((history_size+1)-j)][-1]) for j in range(history_size)]
		num1 = int(data[i][0][11:13])
		num2 = int(data[i][0][14:16])
		newdata[i-(history_size+1)] += [num1+num2/60]

real_value = temps[:int(len(temps)*0.5)]
equation_inputs = newdata[:int(len(temps)*0.5)]

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
#At first, the fitness is calculated for each solution in the final generation.
fitness = cal_pop_fitness(equation_inputs, new_population,real_value)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.argmax(fitness)
print(best_match_idx)
print("Best solution : ", new_population[best_match_idx])
print("Best solution fitness : ", fitness[best_match_idx])
x = newdata[int(len(temps)*0.7):]
real_value = temps[int(len(temps)*0.7):]
w = new_population[best_match_idx]
result = []
for j in range(len(x)):
	result += [x[j][0]*w[0] + x[j][1]*w[1] + x[j][2]*w[2] + x[j][3]*w[3] + x[j][4]*w[4] + x[j][5]*w[5] +w[6]*np.sin((1/3.82)*x[j][6])]
print("best weights", w)
print("best guess is:" ,result[:10])
print("real value was:", real_value[:10])

newresult = []
newreal_value = []
correct = 0
correct_1 = 0
correct_2 = 0
for i in range(len(result)):
	if i%100 == 0:
		newresult += [round(result[i],2)]
		newreal_value += [round(float(real_value[i]),2)]
	if round(result[i],2) == round(float(real_value[i]),2):
		correct_2 += 1
	if round(result[i],1) == round(float(real_value[i]),1):
		correct_1 += 1
	if int(result[i]) == round(float(real_value[i]),0):
		correct += 1
print('acc',correct/len(result)*100)
print('acc_1',correct_1/len(result)*100)
print('acc_2',correct_2/len(result)*100)

plt.scatter(range(len(newresult)),newresult,color="blue")
plt.scatter(range(len(newreal_value)),newreal_value,color="red")
plt.show()
