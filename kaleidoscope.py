import math
from random import Random

from scipy.spatial.transform import Rotation

from models.primitives import get_sphere_equations, get_cylinder_equation, get_hyperboloid_equation, \
    get_paraboloid_equation, get_triangle_equation, get_cube_equations
from models.surface import SolidSurface, RefractionSurface, SurfaceEquation, Surface, ReflectionSurface
from scene_running import Camera, render_scene
import numpy as np
from sympy.abc import x, y, z

from utils import apply_rotation_move

random = Random()

camera_height = 0.5
camera = Camera(np.array([-0.5, -0.5, camera_height-0.5], dtype=float),
                np.array([-0.5, 0.5, camera_height-0.5]),
                np.array([0.5, 0.5, camera_height-0.5]), 90)  # change angle of view
surfaces = []
surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 0, 0]), 100), np.array([1, 1, 1]), 0.2))

side_length = 0.209957 * 2
medium_length = math.sqrt(3) / 2 * side_length
top_z = 0.23322
bottom_z = -1.87888
points = np.array([[-side_length*0.5, -medium_length/3, top_z], [0, 2*medium_length/3, top_z], [side_length*0.5, -medium_length/3, top_z]])
points = np.concatenate([points, points.copy()], axis=0)
points[3:, 2] = bottom_z
sides_surfaces = [
    *get_triangle_equation(points[5], points[2], points[0]),
    *get_triangle_equation(points[5], points[0], points[3]),
    *get_triangle_equation(points[3], points[0], points[1]),
    *get_triangle_equation(points[3], points[1], points[4]),
    *get_triangle_equation(points[4], points[1], points[2]),
    *get_triangle_equation(points[4], points[2], points[5]),
]
surfaces.append(SolidSurface(sides_surfaces, np.array([0, 1, 0]), 1))
# surfaces.append(ReflectionSurface(sides_surfaces, 0.99))
surfaces.append(SolidSurface([SurfaceEquation(SurfaceEquation.EquationType.Plane, z - bottom_z, [x**2 + y**2 - (medium_length * 2 / 3) ** 2])], np.array([1, 1, 1]), 0.3))

cube_size = 0.076123
cube_z_pos = cube_size + bottom_z
number_of_cubes = 7
spawn_area = side_length / 3
for _ in range(number_of_cubes):
    spawn_pos = np.array([random.uniform(-spawn_area, spawn_area), random.uniform(-spawn_area, spawn_area), cube_z_pos])
    spawn_rot = Rotation.from_rotvec(np.array([random.random() * 90, random.random() * 90, random.random() * 90]), degrees=True)
    spawn_color = np.array([random.random(), random.random(), random.random()])
    surfaces.append(SolidSurface(get_cube_equations(spawn_pos, (cube_size, cube_size, cube_size), spawn_rot), spawn_color, 1))


render_scene(f"kaleidoscope_distance_{round(top_z - bottom_z, 1)}.png", surfaces, camera, 100, 100, 1)
