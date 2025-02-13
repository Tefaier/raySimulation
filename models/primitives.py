from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from sympy.abc import x, y, z

from models.surface import SurfaceEquation
from utils import setRotationAngle, rotationToVector, apply_rotation_move


# dump cube without rotation
def get_cube_equations(center_at: np.array, side_lengths: Tuple[float, float, float], rotation: Rotation) -> list[SurfaceEquation]:
    equations = []
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Plane,
        apply_rotation_move(x - side_lengths[0] / 2, center_at, rotation),
        [apply_rotation_move(y - side_lengths[1] / 2, center_at, rotation),
            apply_rotation_move(-y - side_lengths[1] / 2, center_at, rotation),
            apply_rotation_move(z - side_lengths[2] / 2, center_at, rotation),
            apply_rotation_move(-z - side_lengths[2] / 2, center_at, rotation)]))
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Plane,
        apply_rotation_move(-x - side_lengths[0] / 2, center_at, rotation),
        [apply_rotation_move(y - side_lengths[1] / 2, center_at, rotation),
         apply_rotation_move(-y - side_lengths[1] / 2, center_at, rotation),
         apply_rotation_move(z - side_lengths[2] / 2, center_at, rotation),
         apply_rotation_move(-z - side_lengths[2] / 2, center_at, rotation)]))
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Plane,
        apply_rotation_move(y - side_lengths[1] / 2, center_at, rotation),
        [apply_rotation_move(x - side_lengths[0] / 2, center_at, rotation),
         apply_rotation_move(-x - side_lengths[0] / 2, center_at, rotation),
         apply_rotation_move(z - side_lengths[2] / 2, center_at, rotation),
         apply_rotation_move(-z - side_lengths[2] / 2, center_at, rotation)]))
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Plane,
        apply_rotation_move(-y - side_lengths[1] / 2, center_at, rotation),
        [apply_rotation_move(x - side_lengths[0] / 2, center_at, rotation),
         apply_rotation_move(-x - side_lengths[0] / 2, center_at, rotation),
         apply_rotation_move(z - side_lengths[2] / 2, center_at, rotation),
         apply_rotation_move(-z - side_lengths[2] / 2, center_at, rotation)]))
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Plane,
        apply_rotation_move(z - side_lengths[2] / 2, center_at, rotation),
        [apply_rotation_move(y - side_lengths[1] / 2, center_at, rotation),
         apply_rotation_move(-y - side_lengths[1] / 2, center_at, rotation),
         apply_rotation_move(x - side_lengths[0] / 2, center_at, rotation),
         apply_rotation_move(-x - side_lengths[0] / 2, center_at, rotation)]))
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Plane,
        apply_rotation_move(-z - side_lengths[2] / 2, center_at, rotation),
        [apply_rotation_move(y - side_lengths[1] / 2, center_at, rotation),
         apply_rotation_move(-y - side_lengths[1] / 2, center_at, rotation),
         apply_rotation_move(x - side_lengths[0] / 2, center_at, rotation),
         apply_rotation_move(-x - side_lengths[0] / 2, center_at, rotation)]))
    return equations

# simple sphere
def get_sphere_equations(center_at: np.array, radius: float) -> list[SurfaceEquation]:
    equations = []
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Sphere, (x-center_at[0])**2+(y-center_at[1])**2+(z-center_at[2])**2-radius*radius, []))
    return equations

# triangle in 3D, considers outside to be where points 0->1->2 are seen in CW order
# TODO check if this formula is correct at all
def get_triangle_equation(point0: np.array, point1: np.array, point2: np.array) -> list[SurfaceEquation]:
    equations = []
    vector01 = point1 - point0
    vector12 = point2 - point1
    vector20 = point2 - point0
    normal = np.cross(vector12, vector01)
    perp2to01 = setRotationAngle(rotationToVector(vector01, -vector20), 90, True).apply(vector01)
    perp0to12 = -setRotationAngle(rotationToVector(vector12, -vector01), 90, True).apply(vector12)
    perp1to20 = -setRotationAngle(rotationToVector(vector20, -vector12), 90, True).apply(vector20)
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Plane, (x - point0[0]) * normal[0] + (y - point0[1]) * normal[1] + (z - point0[2]) * normal[2], [
        (x - point0[0]) * perp2to01[0] + (y - point0[1]) * perp2to01[1] + (z - point0[2]) * perp2to01[2],
        (x - point1[0]) * perp0to12[0] + (y - point1[1]) * perp0to12[1] + (z - point1[2]) * perp0to12[2],
        (x - point2[0]) * perp1to20[0] + (y - point2[1]) * perp1to20[1] + (z - point2[2]) * perp1to20[2]
    ]))
    return equations

def get_cylinder_equation(ray_through: np.array, radius: float, reverse_normal: bool, rotation: Rotation) -> list[SurfaceEquation]:
    equations = []
    basic_equation = (x ** 2 + y ** 2 - radius ** 2) if not reverse_normal else (-1 * x ** 2 - y ** 2 + radius ** 2)
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Cylinder, apply_rotation_move(basic_equation, ray_through, rotation), []))
    return equations

# add_part = 0 then it is cone
# add_part < 0 then it is one parted hyperboloid
# add_part > 0 then it is two parted hyperboloid
def get_paraboloid_equation(add_part: float, reverse_normal: bool, center_at: np.array, rotation: Rotation) -> list[SurfaceEquation]:
    equations = []
    basic_equation = (x ** 2 + y ** 2 - z ** 2 + add_part) if not reverse_normal else (-1 * x ** 2 - y ** 2 + z ** 2 - add_part)
    equations.append(SurfaceEquation(SurfaceEquation.EquationType.Paraboloid, apply_rotation_move(basic_equation, center_at, rotation), []))
    return equations

