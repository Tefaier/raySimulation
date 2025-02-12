from random import Random

import numpy as np

from control_functions import shoot_ray
from models.primitives import get_cube_equations
from models.surface import ReflectionSurface, RefractionSurface
from models.ray import Ray

random = Random()

for i in range(10000):
    results = np.roots([random.random(), random.random(), random.random()])
    results = results.real[abs(results.imag) < 1e-5]

