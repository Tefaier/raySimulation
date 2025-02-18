import pygame
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from control_functions import shoot_ray
from models.primitives import get_cube_equations, get_sphere_equations, get_triangle_equation, get_cylinder_equation, \
    get_hyperboloid_equation, get_converging_lens_equations
from models.surface import ReflectionSurface, RefractionSurface, SolidSurface, SurfaceEquation
from models.ray import Ray
from sympy.abc import x, y, z

from scene_running import Camera, render_scene


with_lens = True
dist_to_mirror = 100
camera = Camera(np.array([-0.5, -dist_to_mirror, -0.5], dtype=float),
                np.array([-0.5, -dist_to_mirror, 0.5]),
                np.array([0.5, -dist_to_mirror, 0.5]), 90)  # change angle of view
surfaces = []

center = np.array([0, 0, 0])
height = 100
# surfaces.append(SolidSurface(get_converging_lens_equations(center, height, Rotation.from_rotvec([0, 0, np.pi / 2]), sphere_radius=50), np.array([1, 1, 1]), 0.9))
if with_lens:
    surfaces.append(RefractionSurface(get_converging_lens_equations(center, height, Rotation.from_rotvec([0, 0, np.pi / 2]), sphere_radius=50), 1, 1.5))
surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 300, 0]), 80, []), np.array([1, 1, 1]), 0.9))

render_scene(f"image_sphere_{'with' if with_lens else 'without'}_lense.png", surfaces, camera, 800, 800, 1)
