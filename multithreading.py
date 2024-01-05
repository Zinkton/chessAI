import constants
from multiprocessing import Pool

def multithreading_pool(function, inputs):
    with Pool(processes=constants.THREADS) as p:
        return p.map(function, inputs)