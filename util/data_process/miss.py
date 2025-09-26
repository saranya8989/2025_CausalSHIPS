import pickle
import numpy as np

def read_pickle(filepath=None):
    with open(filepath, "rb") as f:
        x = pickle.load(f)
    return x

def flatten(xss):
    return [x for xs in xss for x in xs]