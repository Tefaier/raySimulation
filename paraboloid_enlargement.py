import math

from scipy.spatial.transform import Rotation

from models.primitives import get_sphere_equations, get_cylinder_equation, get_hyperboloid_equation, \
    get_paraboloid_equation
from models.surface import SolidSurface, RefractionSurface, SurfaceEquation, Surface, ReflectionSurface
from scene_running import Camera, render_scene
import numpy as np
from sympy.abc import x, y, z

from utils import apply_rotation_move

desired_enlargement = 0.02
camera_height = 1 / (4 * desired_enlargement)
camera = Camera(np.array([-0.5, -0.5, camera_height-0.5], dtype=float),
                np.array([-0.5, 0.5, camera_height-0.5]),
                np.array([0.5, 0.5, camera_height-0.5]), 90)  # change angle of view
surfaces = []
surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 0, 0]), 100), np.array([1, 1, 1]), 0.2))

surfaces.append(ReflectionSurface(get_paraboloid_equation(math.sqrt(1 / desired_enlargement), True, np.array([0, 0, 0]), Rotation.from_rotvec([0, 0, 0]), [z-5]), 0.99))
# surfaces.append(SolidSurface(get_paraboloid_equation(math.sqrt(1 / desired_enlargement), True, np.array([0, 0, 0]), Rotation.from_rotvec([0, 0, 0]), [z-5]), np.array([1, 0, 0]), 1))


cone_pose = np.array([0.25, 0, 3]) # np.array([4.04965, 0, 0.767727])
cone_rotation = Rotation.from_rotvec([0, 90, 0], degrees=True)
cone_length = 0.5
cone_base_radius = 0.5 / 2
surfaces.append(SolidSurface([
    *get_hyperboloid_equation(0, 53.13, False, cone_pose, cone_rotation, [x - cone_pose[0], -x + (cone_pose[0] - cone_length)]),
     SurfaceEquation(SurfaceEquation.EquationType.Plane, -x + (cone_pose[0] - cone_length), [apply_rotation_move(x ** 2 + y ** 2 - cone_base_radius ** 2, cone_pose, cone_rotation)]),
], np.array([0, 1, 1]), 1))


render_scene(f"paraboloid_{desired_enlargement}.png", surfaces, camera, 400, 400, 1)
