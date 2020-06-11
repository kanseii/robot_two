import pickle
import numpy as np

std = 4
pop_size = 100
gene_size = 702       # DNA length
population = std*np.random.randn(pop_size,gene_size)
filename = "init_population"
with open(filename + ".pkl", 'wb') as file:
    pickle.dump(population, file)
