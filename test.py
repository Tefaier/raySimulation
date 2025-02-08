import numpy as np

from control_functions import shoot_ray
from models.primitives import get_cube_equations
from models.surface import ReflectionSurface
from models.ray import Ray

surfaces = []
surfaces.append(ReflectionSurface(get_cube_equations(np.array([0, 0, 0]), 2), 0.99))
surfaces.append(ReflectionSurface(get_cube_equations(np.array([-20, 0, 0]), 2), 0.99))

ray = Ray(np.array([-10, 0, 0]), np.array([1, 0, 0]))

path, ray = shoot_ray(surfaces, ray, 10)
print(np.array(path).tolist())
print(ray)
