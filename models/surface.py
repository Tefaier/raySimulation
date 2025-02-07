import math
from typing import Any

import numpy as np
import sympy
from scipy.spatial.transform import Rotation
from sympy import diff, simplify, lambdify
from sympy.solvers import solve

from ray import Ray
from sympy.abc import x, y, z, a, b, c, t
from enum import Enum
from scipy.optimize import root_scalar

from utils import multi_root, rotationToVector

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
        if substituted.is_zero: return ray.point + ray.vector * min_ray_fly_distance
        expr_function = lambdify(t, simplify(substituted, rational=True))
        results = multi_root(expr_function, (min_ray_fly_distance, 1e5), 5)
        intersection = None
        for r in results:
            r = float(r)
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


class Surface:
    equations_of_parts: list[SurfaceEquation]

    def __init__(self, equations_of_parts: list[SurfaceEquation]):
        self.equations_of_parts = equations_of_parts

    def determine_intersection(self, with_ray: Ray) -> Ray | None:
        first_distance = 0
        first_intersection = None
        first_surface = None
        for surface in self.equations_of_parts:
            intersection = surface.eval_ray(with_ray)
            length = np.linalg.norm(intersection - with_ray.point)
            if intersection is not None and (first_intersection is None or first_distance > length):
                first_intersection = intersection
                first_distance = length
                first_surface = surface
        if first_intersection is None: return None
        normal = first_surface.get_normal_at_point(first_intersection)
        if normal is None:
            with_ray.point = first_intersection
            return with_ray
        return self.build_new_ray(with_ray, first_intersection, normal)

    def build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
        pass


class ReflectionSurface(Surface):
    reflection_coefficient: float

    def __init__(self, equations_of_parts: list[SurfaceEquation], reflection_coefficient: float):
        super().__init__(equations_of_parts)
        self.reflection_coefficient = reflection_coefficient

    def build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
        new_vector = from_ray.vector * -1
        rot_to_normal = rotationToVector(new_vector, touch_normal)
        new_vector = rot_to_normal.apply(rot_to_normal.apply(new_vector))
        ray = Ray(touch_point, new_vector)
        ray.hit_count = from_ray.hit_count + 1
        ray.total_fly_distance = from_ray.total_fly_distance + np.linalg.norm(touch_point - from_ray.point)
        ray.light_level = from_ray.light_level * self.reflection_coefficient
        return ray


class RefractionSurface(Surface):
    refractive_index_outside: float
    refractive_index_inside: float
    critical_angle: float

    def __init__(self, equations_of_parts: list[SurfaceEquation], refractive_index_outside: float, refractive_index_inside: float):
        super().__init__(equations_of_parts)
        self.refractive_index_outside = refractive_index_outside
        self.refractive_index_inside = refractive_index_inside
        self.critical_angle = math.asin(min(refractive_index_outside, refractive_index_inside) / max(refractive_index_outside, refractive_index_inside))

    def build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
        cos = np.cos(from_ray.vector, touch_normal)
        sin_multiplier = 0
        if cos > 0:  # hits surface from inside
            sin_multiplier = self.refractive_index_inside / self.refractive_index_outside
        else:  # hits surface from outside
            sin_multiplier = self.refractive_index_outside / self.refractive_index_inside
        angle_to_perp = math.acos(cos)
        angle_to_perp = angle_to_perp if angle_to_perp > math.pi / 2 else math.pi - angle_to_perp
        new_sin_value = math.sin(angle_to_perp) * sin_multiplier
        if new_sin_value > 1.0:  # for now it fully reflects inside after critical angle
            new_vector = from_ray.vector * -1
            rot_to_normal = rotationToVector(new_vector, touch_normal)
            new_vector = rot_to_normal.apply(rot_to_normal.apply(new_vector))
            ray = Ray(touch_point, new_vector)
            ray.hit_count = from_ray.hit_count + 1
            ray.total_fly_distance = from_ray.total_fly_distance + np.linalg.norm(touch_point - from_ray.point)
            ray.light_level = from_ray.light_level
            return ray
        else:
            if cos < 0:
                touch_normal *= -1
            rot_normal_to_ray = rotationToVector(touch_normal, from_ray.vector)
            rot_normal_to_ray = rot_normal_to_ray.as_rotvec(degrees=False)
            rot_normal_to_ray /= np.linalg.norm(rot_normal_to_ray)
            rot_normal_to_ray *= math.asin(new_sin_value)
            rot_normal_to_ray = Rotation.from_rotvec(rot_normal_to_ray, degrees=False)
            new_vector = rot_normal_to_ray.apply(touch_normal)
            ray = Ray(touch_point, new_vector)
            ray.hit_count = from_ray.hit_count + 1
            ray.total_fly_distance = from_ray.total_fly_distance + np.linalg.norm(touch_point - from_ray.point)
            ray.light_level = from_ray.light_level
            return ray


class SolidSurface(Surface):
    color: list[float]
    brightness: float

    def __init__(self, equations_of_parts: list[SurfaceEquation], color: list[float], brightness: float):
        super().__init__(equations_of_parts)
        self.color = color
        self.brightness = brightness

    def build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
        ray = Ray(touch_point, from_ray.vector)
        ray.hit_count = from_ray.hit_count + 1
        ray.total_fly_distance = from_ray.total_fly_distance + np.linalg.norm(touch_point - from_ray.point)
        ray.light_level = from_ray.light_level
        ray.finished = True
        cos = np.cos(from_ray.vector, touch_normal)
        angle_to_perp = math.acos(cos)
        angle_to_perp = angle_to_perp if angle_to_perp > math.pi / 2 else math.pi - angle_to_perp
        self.determine_final_color(ray, angle_to_perp)
        return ray

    def determine_final_color(self, ray: Ray, fall_angle: float):
        # TODO
        ray.final_color = [1, 1, 1]
