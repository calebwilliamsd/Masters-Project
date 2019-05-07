# Caleb Williams Masters Project


from copy import deepcopy
import math
import time
import random
import numpy as np


fitness_evals = [0]
num_evals = 10000


	
RANGE = [2.048]
poor_size = [5,1,3,7,0,10]
rich_size = [5,9,7,3,10,0]
poor_size = [5]
rich_size = [5]
pop_size = 10
step_size = [ [(1/32)*RANGE[0]] * pop_size] 
bad_cnt = [0]*pop_size
iter = 10
worst_rich = [999999]
DIMENSIONS = [2,10,30]



half = poor_size

#test Functions

# DJ -5.12 - 5.12
def dj(x,n,fitness_evals):
        sum = 0
        for i in range(n):
                sum += x[i]**2
        fitness_evals+=1
        return sum,fitness_evals

# Rosenbrock -2.048 - 2.048
def rosenbrock(x,n,fitness_evals): 
        sum = 0
        for i in range(n-1):
                sum += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        fitness_evals+=1 
        return sum,fitness_evals



# Shubert -5.12 - 5.12 n = 2
def shubert(x,n,fitness_evals):

	first_sum = 0
	second_sum = 0
	for i in range(6):
		first_sum += i * math.cos((i+1)*x[0] + i)
		second_sum += i * math.cos((i+1)*x[1] + i)

	fitness_evals += 1
	return first_sum * second_sum, fitness_evals

# Rastrigin -5.12 - 5.12
def rastrigin(x,n,fitness_evals):
        sum = 0
        for i in range(n):
                sum += x[i]**2 - (10 * math.cos(2 * math.pi * x[i]))

        sum += 10 * n
        fitness_evals+=1
        return sum,fitness_evals

def fitness(x,n,fitness_evals):

	print("placeholder")

# defining a solution which has coordinates, a fitness, and an id

class Solution:

	def __init__(self,index,nn, problem, randomize=True):

		if problem == 'dj':
			fitness = dj
		elif problem == 'shubert':
		 	fitness = shubert
		elif problem == 'rosenbrock':
			fitness = rosenbrock
		elif problem == 'rastrigin':
			fitness = rastrigin

		if randomize:
			self.x = [random.uniform(-RANGE[0],RANGE[0]) for p in range(nn)]
			self.fitness,s = fitness(self.x,nn,nn)
			self.index = index
		else:
			self.x = [8 for p in range(nn)]
			self.fitness,s = fitness(self.x,nn,nn)
			self.index = index

# sort the population based on fitness

def sortPopulation(fit):
	
	# return the sorted list of fitness indices (e.g. fitness = [-1,2,4,0] would return [0,3,1,2])
	indices = sorted(range(len(fit)), key = lambda k: fit[k], reverse = True)

	return indices


# move the poor towards the worst rich

def step_poor(solutions,prey,step):



	dist_to_boundary = RANGE[0] - abs(solutions)

	# if there is a potential we go out of bounds, limit the wiggle room so we don't                  
	if(dist_to_boundary < step):
		if(solutions < 0):
			solutions+=random.uniform(-(dist_to_boundary),step)
		else:
			solutions+=random.uniform(-step, dist_to_boundary)
        
	elif(solutions > prey):
		solutions+=random.uniform(-step,0)
	elif(solutions <= prey):
		solutions+=random.uniform(0,step)
		

	# just in case
	if solutions > RANGE[0]:
		solutions = RANGE[0]
	elif solutions < -RANGE[0]:
		solutions = -RANGE[0]
	return solutions
	
# stochastic hill climbing

def step(solutions,step_size):

	dist_to_boundary = RANGE[0] - abs(solutions)

	# if there is a potential we go out of bounds, limit the wiggle room so we don't                  
	if(dist_to_boundary < step_size):
		if(solutions < 0):
			solutions+=random.uniform(-(dist_to_boundary),step_size)
		else:
			solutions+=random.uniform(-step_size, dist_to_boundary)
        
	else:
		solutions+=random.uniform(-step_size,step_size)

	# just in case
	if solutions > RANGE[0]:
		solutions = RANGE[0]
	elif solutions < -RANGE[0]:
		solutions = -RANGE[0]
	return solutions



# Pass in prevRich and currRich to get the new rich
def checkNewRich(prev, curr):

	d = set.difference(set(curr), set(prev))

	return list(d)


# move all the poor individuals towards the worst rich

def poor(solutions, best, bestx, prey,currst,n,problem):


	if problem == 'dj':
		fitness = dj
	elif problem == 'shubert':
		fitness = shubert
	elif problem == 'rosenbrock':
		fitness = rosenbrock
	elif problem == 'rastrigin':
		fitness = rastrigin

	for i in range(len(solutions)):
		for j in range(len(solutions[i].x)):
			solutions[i].x[j] = step_poor(solutions[i].x[j],prey[j], step_size[currst][solutions[i].index])	
		curr,fitness_evals[0] = fitness(solutions[i].x,n,fitness_evals[0])
		solutions[i].fitness = curr
	
		if(curr < best):
			best = curr
			bestx = solutions[i].x
	

	return best, bestx

# perform stochastic hill climbing for every rich

def rich(solutions, best, bestx,currst,n,problem):

	if problem == 'dj':
		fitness = dj
	elif problem == 'shubert':
		fitness = shubert
	elif problem == 'rosenbrock':
		fitness = rosenbrock
	elif problem == 'rastrigin':
		fitness = rastrigin


	for i in range(len(solutions)):
		newx = []
		
		currSolution = solutions[i].index

		for j in range(len(solutions[i].x)):
			newx.append(step(solutions[i].x[j], step_size[currst][currSolution]))
			
		curr,fitness_evals[0] = fitness(newx,n,fitness_evals[0])
		old = solutions[i].fitness
	
		bad_cnt[currSolution] += 1
	
		if(curr < old):
			solutions[i].x = newx
			solutions[i].fitness = curr
			bad_cnt[currSolution] = 0
			step_size[currst][currSolution] += (1/131072) * RANGE[0]
			step_size[currst][currSolution] = min(RANGE[0], step_size[currst][currSolution])
	
			if(curr < best):
				best = curr
				bestx = solutions[i].x
		if(bad_cnt[currSolution] > (5)):
			step_size[currst][currSolution] -= (1/128) * RANGE[0]
			step_size[currst][currSolution] = max(0.0000000001, step_size[currst][currSolution])
			bad_cnt[currSolution] = 0 

			
	return best, bestx


def main():


	not_swp_cnt = 0

	problems = ['dj','rosenbrock','rastrigin','shubert']

	# run algorithm on each test function

	for problem in problems:
	

		if problem == 'dj':
			fitness = dj
			RANGE[0] = 5.12
		elif problem == 'shubert':
			fitness = shubert
			RANGE[0] = 5.12
		elif problem == 'rosenbrock':
			fitness = rosenbrock
			RANGE[0] = 2.048
		elif problem == 'rastrigin':
			fitness = rastrigin
			RANGE[0] = 5.12

		for poor_pop,rich_pop in zip(poor_size,rich_size):
			for currst in range(len(step_size)):
				for n in DIMENSIONS:
				
	
					half = min(poor_pop,pop_size-1)
				
					# The poor will try to overtake the half one.
					avg_best = [0] * iter

					# run it ite number of times

					for ite in range(iter):
						random.seed(ite)	
						solutions = []	
					
						best = 10**20
					
						# list of all fitnesses
						fit = []

						# initialize solutions
					
						for i in range(pop_size):
							solutions.append(Solution(i,n,problem))
							fitness_evals[0] += 1
							if(solutions[i].fitness < best):
								best = solutions[i].fitness
								bestx = solutions[i]
						
						worst_rich[0] = solutions[half].fitness

						# getting the sorted indices of the population so we can change x to contain first half poor then second half rich
						sortedIndices = sortPopulation([solutions[r].fitness for r in range(pop_size)])
					
						while fitness_evals[0] < num_evals:
					
							# standard procedure
							# perform the different search methods if u are rich or poor
							best, bestx = poor(solutions[:half], best, bestx,solutions[half].x,currst,n,problem)
							best, bestx = rich(solutions[half:], best, bestx,currst,n,problem) 
				
							# extracting the indices for the rich for tracking purposes	
							prevRich = [solutions[d].index for d in range(half,len(solutions))]
					
					
					
							sortedIndices = sortPopulation([solutions[r].fitness for r in range(pop_size)])
					
							# reorder the solutions
							solutions[:] = [solutions[r] for r in sortedIndices]
						
				
							worst_rich[0] = solutions[half].fitness

							# Lottery mechanism

							for j in range(half):
								if(random.uniform(0,1) < .001):
									solutions[j].x = deepcopy(solutions[-1].x)
									solutions[j].fitness = solutions[-1].fitness
									solutions[:] = [solutions[r] for r in sortedIndices]
							currRich = [solutions[d].index for d in range(half,len(solutions))]
					
							
					
							newRich = checkNewRich(prevRich, currRich)
					
					
					
							# if a poor did not become rich, then maybe increment the poors step size
							if(not newRich and half != 0):
								not_swp_cnt += 1
								if(not_swp_cnt > 10):
									solutions[0].x = [random.uniform(-RANGE[0], RANGE[0]) for i in range(n)]
									solutions[0].fitness, fitness_evals[0] = fitness(solutions[0].x,n,fitness_evals[0])
									curr = solutions[0].fitness
									if( solutions[0].fitness < best):
										best = curr
										bestx = solutions[0].x
									# first half of solutions are all poor
									for ind in range(half):
										step_size[currst][solutions[ind].index] = step_size[currst][solutions[ind].index] + ( (1/256) * RANGE[0])
										step_size[currst][solutions[ind].index] = min(RANGE[0], step_size[currst][solutions[ind].index])
									not_swp_cnt = 0
							# if new rich, then just reset counter
							else:
								not_swp_cnt = 0
					
						#print("best is ",best, " at ",bestx  )
						avg_best[ite] += best
						fitness_evals[0] = 0	
						if currst == 0:
							step_size[currst][:] = [(1/32)*RANGE[0]] * pop_size
					
						bad_cnt[:] = [0]*pop_size
					
				
						
					
					std_best = np.std(avg_best)		
					mean_best = np.mean(avg_best)
					print( problem + " avg best for ",mean_best," with std ",std_best, " poor pop ",poor_pop," rich pop ",rich_pop, " step size ",currst, " dimensions ",n )


if __name__ == "__main__":
	main()

