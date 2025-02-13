import math
from enum import Enum
from typing import Callable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from sympy import diff, simplify, lambdify, symbols

from models.ray import Ray
from sympy.abc import x, y, z, t
from sympy import Poly

from utils import rotationToVector

min_ray_fly_distance = 1e-6

# surface_equation is checked to be 0 at coordinates
# surface_limitations are checked to be <= 0 ALL!
# normal of equation is in direction of gradient at point
class SurfaceEquation:
    class EquationType(Enum):
        Sphere = 0,
        Plane = 1,
        Cylinder = 2,
        Paraboloid = 3

    equation_type: EquationType
    _solver: Callable[[Ray], float]
    _ray_to_polynom: Callable[[float, float, float, float, float, float], list[float]] # may be None depending on the equation
    _equation_params: list[float] # may be None depending on the equation
    _surface_normal: Callable[[float, float, float], Tuple[float, float, float]]
    _surface_limitations: list[Callable[[float, float, float], float]]

    def __init__(self, equation_type: EquationType, equation, limitations: list):
        self.equation_type = equation_type
        if equation_type == self.EquationType.Sphere:
            self._solver = self._solve_sphere
        elif equation_type == self.EquationType.Plane:
            self._solver = self._solve_plane
        elif equation_type == self.EquationType.Cylinder:
            self._solver = self._solve_cylinder
        elif equation_type == self.EquationType.Paraboloid:
            self._solver = self._solve_paraboloid

        if equation_type in [self.EquationType.Sphere, self.EquationType.Plane, self.EquationType.Cylinder, self.EquationType.Paraboloid]:
            ax, bx, ay, by, az, bz = symbols('ax, bx, ay, by, az, bz')
            coef_lambdas = [lambdify([ax, bx, ay, by, az, bz], coef_equation) for coef_equation in Poly(equation.subs([(x, ax + bx * t), (y, ay + by * t), (z, az + bz * t)]), t).all_coeffs()]
            self._ray_to_polynom = lambda px, vx, py, vy, pz, vz: [coef_lambda(px, vx, py, vy, pz, vz) for coef_lambda in coef_lambdas]
        else:
            self._equation_params = equation

        norm_lambdas = [lambdify([x, y, z], diff(equation, arg)) for arg in [x, y, z]]
        self._surface_normal = lambda x_val, y_val, z_val: [norm_lambda(x_val, y_val, z_val) for norm_lambda in norm_lambdas]
        self._surface_limitations = [lambdify([x, y, z], simplify(limitation, rational=True)) for limitation in limitations]

    def eval_ray(self, ray: Ray) -> float:
        return self._solver(ray)

    def get_normal_at_point(self, at_point: np.array) -> np.array:
        return np.array(self._surface_normal(at_point[0], at_point[1], at_point[2]))

    def _solve_sphere(self, ray: Ray) -> float:
        return self._solve_by_poly_roots(ray)

    def _solve_plane(self, ray: Ray) -> float:
        return self._solve_by_poly_roots(ray)

    def _solve_cylinder(self, ray: Ray) -> float:
        return self._solve_by_poly_roots(ray)

    def _solve_paraboloid(self, ray: Ray) -> float:
        return self._solve_by_poly_roots(ray)

    def _solve_by_poly_roots(self, ray: Ray) -> float:
        results: np.ndarray = np.roots(
            self._ray_to_polynom(ray.point[0], ray.vector[0], ray.point[1], ray.vector[1], ray.point[2], ray.vector[2]))
        results = results.real[abs(results.imag) < 1e-5]
        results = results[results >= min_ray_fly_distance]
        results.sort()
        for result in results:
            px, py, pz = ray.point + ray.vector * result
            if all(limit(px, py, pz) < 1e-5 for limit in self._surface_limitations):
                return result
        return None


class Surface:
    equations_of_parts: list[SurfaceEquation]

    def __init__(self, equations_of_parts: list[SurfaceEquation]):
        self.equations_of_parts = equations_of_parts

    # returns distance to first intersection, and index of surface equation that intersected
    def determine_intersection(self, with_ray: Ray) -> (float, int):
        first_distance = 0
        first_surface_index = None
        for index, surface in enumerate(self.equations_of_parts):
            intersection = surface.eval_ray(with_ray)
            if intersection is not None and (first_surface_index is None or first_distance > intersection):
                first_distance = intersection
                first_surface_index = index
        if first_surface_index is None: return -1, -1
        return first_distance, first_surface_index

    def effect_ray(self, ray: Ray, fly_distance: np.array, surface_eq_index: int) -> Ray:
        touch_point = ray.point + ray.vector * fly_distance
        normal = self.equations_of_parts[surface_eq_index].get_normal_at_point(touch_point)
        if normal is None:
            ray.point = touch_point
            return ray
        return self._build_new_ray(ray, touch_point, normal)

    def _build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
        pass


class ReflectionSurface(Surface):
    reflection_coefficient: float

    def __init__(self, equations_of_parts: list[SurfaceEquation], reflection_coefficient: float):
        super().__init__(equations_of_parts)
        self.reflection_coefficient = reflection_coefficient

    def _build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
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

    def _build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
        cos = np.dot(from_ray.vector, touch_normal)  # touch_normal length is expected to be 1!
        sin_multiplier = 0
        if cos > 0:  # hits surface from inside
            sin_multiplier = self.refractive_index_inside / self.refractive_index_outside
        else:  # hits surface from outside
            sin_multiplier = self.refractive_index_outside / self.refractive_index_inside
        angle_to_perp = math.acos(cos)
        angle_to_perp = angle_to_perp if angle_to_perp < math.pi / 2 else math.pi - angle_to_perp
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
            if np.linalg.norm(rot_normal_to_ray) != 0:
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
    color: np.array
    brightness: float

    def __init__(self, equations_of_parts: list[SurfaceEquation], color: np.array, brightness: float):
        super().__init__(equations_of_parts)
        self.color = color
        self.brightness = brightness

    def _build_new_ray(self, from_ray: Ray, touch_point: np.array, touch_normal: np.array) -> Ray:
        ray = Ray(touch_point, from_ray.vector)
        ray.hit_count = from_ray.hit_count + 1
        ray.total_fly_distance = from_ray.total_fly_distance + np.linalg.norm(touch_point - from_ray.point)
        ray.light_level = from_ray.light_level
        ray.finished = True
        # cos = np.cos(from_ray.vector, touch_normal)
        cos = np.dot(from_ray.vector, touch_normal)
        angle_to_perp = math.acos(max(-1, min(1, cos)))
        angle_to_perp = angle_to_perp if (angle_to_perp < math.pi / 2) else (math.pi - angle_to_perp)
        self._determine_final_color(ray, angle_to_perp)
        return ray

    # fall angle is an angle between ray and normal in radians
    def _determine_final_color(self, ray: Ray, fall_angle: float):
        fall_brigntness = np.cos(fall_angle)
        final_brigntness = fall_brigntness * ray.light_level * self.brightness
        ray.final_color = self.color * ray.color_mask * final_brigntness
