import pygame
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from control_functions import shoot_ray
from models.primitives import get_cube_equations, get_sphere_equations, get_triangle_equation, get_cylinder_equation, \
    get_hyperboloid_equation
from models.surface import ReflectionSurface, RefractionSurface, SolidSurface, SurfaceEquation
from models.ray import Ray
from sympy.abc import x, y, z

from scene_running import Camera, render_scene
from utils import sellmeier_lambda

dist_to_mirror = 0.8
camera = Camera(np.array([-0.5, -0.5, dist_to_mirror], dtype=float),
                np.array([-0.5, 0.5, dist_to_mirror]),
                np.array([0.5, 0.5, dist_to_mirror]), 90)  # change angle of view
surfaces = []


surfaces.append(RefractionSurface(get_cube_equations(np.array([0.01, 0, 0.4]), (0.7, 0.7, 0.7), Rotation.from_rotvec([0, 0, 0])), lambda alpha: 1, sellmeier_lambda(1.0396, 6000, 0.2318, 20000)))
# surfaces.append(RefractionSurface([SurfaceEquation(True, z+x,[])], 1, 2.4))
# surfaces.append(RefractionSurface([SurfaceEquation(True, z-1,[y - mirror_side, -y - mirror_side, x - mirror_side, -x - mirror_side])], 1, 1.4))

# surfaces.append(SolidSurface(get_cube_equations(np.array([0, 0, 80]), 2), 1, 100))
# surfaces.append(SolidSurface(get_triangle_equation(np.array([-5, 0, 40]), np.array([0, 5, 40]), np.array([5, 0, 40])), 1, 100))
surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 0, 0]), 100), np.array([1, 1, 1]), 0.2))
surfaces.append(SolidSurface(get_sphere_equations(np.array([0.2, 0, -0.5]), 0.3), np.array([1, 1, 1]), 1))

color_split = 7
render_scene(f"glass_cube_{color_split}.png", surfaces, camera, 1000, 1000, color_split=color_split)
