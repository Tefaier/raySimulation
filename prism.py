import numpy as np
from scipy.spatial.transform import Rotation
import math

from control_functions import shoot_ray
from models.primitives import get_cube_equations, get_sphere_equations, get_triangle_equation, get_cylinder_equation, \
    get_hyperboloid_equation
from models.ray import Ray
from models.surface import ReflectionSurface, RefractionSurface, SolidSurface, SurfaceEquation

from scene_running import Camera, render_scene
from utils import sellmeier_lambda

dist_to_mirror = 1
camera = Camera(np.array([-3.2, -0.1, -0.5], dtype=float),
                np.array([-3.2, -0.1, 0.5]),
                np.array([-2.88, -1.04, 0.5]), 50)  # change angle of view
surfaces = []

side_length = 1.5 * 2
medium_length = math.sqrt(3) / 2 * side_length
top_z = 0.5
bottom_z = -0.5
points = np.array([[-side_length*0.5, -medium_length/3, top_z], [0, 2*medium_length/3, top_z], [side_length*0.5, -medium_length/3, top_z]])
points = np.concatenate([points, points.copy()], axis=0)
points[3:, 2] = bottom_z
sides_surfaces = [
    *get_triangle_equation(points[2], points[5], points[0]),
    *get_triangle_equation(points[0], points[5], points[3]),
    *get_triangle_equation(points[0], points[3], points[1]),
    *get_triangle_equation(points[1], points[3], points[4]),
    *get_triangle_equation(points[1], points[4], points[2]),
    *get_triangle_equation(points[2], points[4], points[5]),
]
surfaces.append(RefractionSurface(sides_surfaces, lambda alpha: 1, sellmeier_lambda(1.0396, 6000, 0.2318, 20000)))
#surfaces.append(SolidSurface(sides_surfaces, np.array([1, 1, 1]), 1))
surfaces.append(SolidSurface(get_sphere_equations(np.array([3.05, -0.57, 0]), 0.3), np.array([1, 1, 1]), 1))


# surfaces.append(SolidSurface(get_cube_equations(np.array([0, 0, 80]), 2), 1, 100))
# surfaces.append(SolidSurface(get_triangle_equation(np.array([-5, 0, 40]), np.array([0, 5, 40]), np.array([5, 0, 40])), 1, 100))
surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 0, 0]), 100), np.array([1, 1, 1]), 0.2))

color_split = 3
render_scene(f"prism_{color_split}.png", surfaces, camera, 100, 100, color_split=color_split)
