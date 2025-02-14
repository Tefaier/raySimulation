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

radius_main = 3.61
radius_cut = 1.19763 * 0.5
fiber_equations = [
    SurfaceEquation(SurfaceEquation.EquationType.Torus, apply_rotation_move((x**2 + y**2 + z**2 + radius_main ** 2 - radius_cut ** 2) ** 2 - 4 * radius_main ** 2 * (x ** 2 + y ** 2), np.array([0, 0, 0]), Rotation.from_rotvec([90, 0, 0], degrees=True)), [z]),
    SurfaceEquation(SurfaceEquation.EquationType.Plane, z, [apply_rotation_move(x ** 2 + y ** 2 - radius_cut ** 2, np.array([radius_main, 0, 0]), Rotation.from_rotvec([0, 0, 0]))]),
    SurfaceEquation(SurfaceEquation.EquationType.Plane, z,[apply_rotation_move(x ** 2 + y ** 2 - radius_cut ** 2, np.array([-radius_main, 0, 0]), Rotation.from_rotvec([0, 0, 0]))])
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


render_scene("optic_fiber_torus.png", surfaces, camera, 100, 100, 1)
