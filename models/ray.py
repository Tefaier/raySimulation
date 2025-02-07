import numpy as np


class Ray:
    point: np.array
    vector: np.array  # directional vector, normalised
    hit_count: int  # number of hits experienced
    light_level: float  # from 0 to 1 for how much brightness is left
    color_mask: list[float]  # from 0 to 1 for RGB
    finished: bool  # true when ray hit
    total_fly_distance: float  # if finished it will show total path length

    def __init__(self, point: np.array, vector: np.array):
        self.point = point
        self.vector = vector / np.linalg.norm(vector)
        self.hit_count = 0
        self.light_level = 1.0
        self.color_mask = [1.0, 1.0, 1.0]
        self.finished = False
        self.total_fly_distance = 0
