import numpy as np


def random_matrix(size,bounds = (0,100)):
    A = np.random.randint(bounds[0], bounds[1] + 1, (size, size))
    return (A+A.T)//2


def generate_data(city_count,how_many_funs,bounds_tab=None):
    data=[]
    if(bounds_tab):
        for lower,higher in bounds_tab:
            data.append(random_matrix(city_count,(lower,higher)))
    else:
        for i in range(how_many_funs):
            data.append(random_matrix(city_count))
    return data



