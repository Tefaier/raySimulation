from typing import Any

import numpy as np
import sympy
from sympy import diff
from sympy.solvers import solve

from ray import Ray
from sympy.abc import x, y, z, a, b, c, t
from enum import Enum

min_ray_fly_distance = 0.001

# surface_equation is checked to be 0 at coordinates
# surface_limitations are checked to be negative all
class SurfaceEquation:
    surface_equation: Any
    surface_limitations: list[Any]

    def __init__(self, equation, limitations):
        self.surface_equation = equation
        self.surface_limitations = limitations

    def eval_ray(self, ray: Ray) -> np.array | None:
        substituted = self.surface_equation.subs([(x, ray.point[0] + ray.vector[0] * t),
                                                  (y, ray.point[1] + ray.vector[1] * t),
                                                  (z, ray.point[2] + ray.vector[2] * t)])
        results = solve(substituted, t)
        intersection = None
        for r in results:
            if r >= min_ray_fly_distance:
                intersection = ray.point + ray.vector * r
                for limitation in self.surface_limitations:
                    if limitation.subs([(x, intersection[0]), (y, intersection[1]), (z, intersection[2])]) > 0:
                        intersection = None
                        break
                if intersection is not None:
                    break

        return intersection

    def get_normal_at_point(self, at_point: np.array) -> np.array:
        result = np.zeros((3,), dtype=float)
        result[0] = diff(self.surface_equation, x).subs([(x, at_point[0]), (y, at_point[1]), (z, at_point[2])])
        result[1] = diff(self.surface_equation, y).subs([(x, at_point[0]), (y, at_point[1]), (z, at_point[2])])
        result[2] = diff(self.surface_equation, z).subs([(x, at_point[0]), (y, at_point[1]), (z, at_point[2])])
        normal_length = np.linalg.norm(result)
        return result / normal_length if normal_length > 0 else None


class SurfaceTypes(Enum):
    Coloured = 0,
    Refraction = 1,
    Reflection = 2


class Surface:
    refraction_index_outside: float
    refraction_index_inside: float

    equations_of_parts: list[SurfaceEquation]

    def __init__(self,
                 equations_of_parts: list[SurfaceEquation],
                 refraction_index_outside: float,
                 refraction_index_inside: float):
        self.equations_of_parts = equations_of_parts
        self.refraction_index_outside = refraction_index_outside
        self.refraction_index_inside = refraction_index_inside

    def determine_intersection(self, with_ray: Ray) -> Ray:
        pass
