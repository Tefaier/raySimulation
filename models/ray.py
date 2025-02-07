import numpy as np


class Ray:
    point: np.array
    vector: np.array  # directional vector, normalised
    hit_count: int  # number of hits experienced
    light_level: float  # from 0 to 1 for how much brightness is left
    color_mask: list[float]  # from 0 to 1 for RGB

    def __init__(self, point: np.array, vector: np.array):
        self.point = point
        self.vector = vector / np.linalg.norm(vector)