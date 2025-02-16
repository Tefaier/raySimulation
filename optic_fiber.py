from scipy.spatial.transform import Rotation

from models.primitives import get_sphere_equations, get_cylinder_equation, get_hyperboloid_equation
from models.surface import SolidSurface, RefractionSurface, SurfaceEquation, Surface
from scene_running import Camera, render_scene
import numpy as np
from sympy.abc import x, y, z

from utils import apply_rotation_move

camera_height = 0.448113
camera = Camera(np.array([-4.1, -0.5, camera_height], dtype=float),
                np.array([-4.1, 0.5, camera_height]),
                np.array([-3.1, 0.5, camera_height]), 90)  # change angle of view
surfaces = []
surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 0, 0]), 100), np.array([1, 1, 1]), 0.2))

radius_outer = 8.37947 * 0.5
radius_inner = 2.98932
height_half = 1.41247 * 0.5
fiber_equations = [
    *get_cylinder_equation(np.array([0, 0, 0]), radius_outer, False, Rotation.from_rotvec([90, 0, 0], degrees=True), [y - height_half, -y - height_half, z]),
    *get_cylinder_equation(np.array([0, 0, 0]), radius_inner, True, Rotation.from_rotvec([90, 0, 0], degrees=True),
                           [y - height_half, -y - height_half, z]),
    SurfaceEquation(SurfaceEquation.EquationType.Plane, y - height_half, [z, x ** 2 + z ** 2 - radius_outer ** 2, -1 * x ** 2 - z ** 2 + radius_inner ** 2]),
    SurfaceEquation(SurfaceEquation.EquationType.Plane, -y - height_half, [z, x ** 2 + z ** 2 - radius_outer ** 2, -1 * x ** 2 - z ** 2 + radius_inner ** 2]),
    SurfaceEquation(SurfaceEquation.EquationType.Plane, z, [y - height_half, -y - height_half, x - radius_outer, -x + radius_inner]),
    SurfaceEquation(SurfaceEquation.EquationType.Plane, z,[y - height_half, -y - height_half, -x - radius_outer, x + radius_inner])
]

# surfaces.append(SolidSurface(fiber_equations, np.array([1, 0, 0]), 1))
surfaces.append(RefractionSurface(fiber_equations, 1, 1.4))
cone_pose = np.array([4.04965, 0, 0.767727]) # np.array([4.04965, 0, 0.767727])
cone_rotation = Rotation.from_rotvec([0, 90, 0], degrees=True)
cone_length = 0.794607
cone_base_radius = 0.794607 / 2
surfaces.append(SolidSurface([
    *get_hyperboloid_equation(0, 53.13, False, cone_pose, cone_rotation, [x - cone_pose[0], -x + (cone_pose[0] - cone_length)]),
     SurfaceEquation(SurfaceEquation.EquationType.Plane, -x + (cone_pose[0] - cone_length), [apply_rotation_move(x ** 2 + y ** 2 - cone_base_radius ** 2, cone_pose, cone_rotation)]),
], np.array([0, 1, 0]), 1))


render_scene("optic_fiber.png", surfaces, camera, 1000, 1000, 1)
