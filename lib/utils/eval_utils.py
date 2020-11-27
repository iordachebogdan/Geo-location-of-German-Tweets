import numpy as np


def mae_coordinates(true, predicted):
    mae = np.abs(true - predicted).mean(axis=0)
    return (mae[0] + mae[1]) / 2
