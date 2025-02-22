import numpy as np


class Ray:
    point: np.array
    vector: np.array  # directional vector, normalised
    hit_count: int  # number of hits experienced
    light_level: float  # from 0 to 1 for how much brightness is left
    color_mask: np.array  # from 0 to 1 for RGB
    finished: bool  # true when ray hit
    total_fly_distance: float  # if finished it will show total path length
    final_color: list[float]  # from 0 to 1 for RGB, determined after finished becomes True
    wavelength: float # wavelength in nm

    def __init__(self, point: np.array, vector: np.array, wavelength: float = 500.0):
        self.point = point
        self.vector = vector / np.linalg.norm(vector)
        self.hit_count = 0
        self.light_level = 1.0
        self.color_mask = np.array([1.0, 1.0, 1.0])
        self.finished = False
        self.total_fly_distance = 0
        self.wavelength = wavelength

    def __str__(self):
        return f"[{round(self.point[0], 3)}, {round(self.point[1], 3)}, {round(self.point[2], 3)}] in direction [{round(self.vector[0], 3)}, {round(self.vector[1], 3)}, {round(self.vector[2], 3)}]\nHit count: {self.hit_count}\nLight level: {self.light_level}\nFinished: {self.finished}"
