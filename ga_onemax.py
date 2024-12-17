import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def f(X):
    return -np.sum(X)

algorithm_param = {'max_num_iteration': 15,\
                   'population_size':50,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'one_point',\
                   'max_iteration_without_improv':None}

model=ga(function=f, dimension=15, variable_type='bool', algorithm_parameters=algorithm_param)

model.run()
