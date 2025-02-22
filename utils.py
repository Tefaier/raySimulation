import math
from typing import Callable

import numpy as np
from scipy.optimize import root_scalar
from sympy import symbols
from sympy.abc import x, y, z

from scipy.spatial.transform import Rotation

noRotation = Rotation.from_euler('x', 0)
vectorUp = np.array([0, 0, 1])


def vectorLerp(vec1: np.array, vec2: np.array, ratio: float):
    return (1 - ratio) * vec1 + ratio * vec2


def clearValues():
    global total_fuel, total_time_thruster_on
    total_fuel = 0
    total_time_thruster_on = 0


def setRotationAngle(rot: Rotation, angle: float, degrees: bool = False) -> Rotation:
    vec = rot.as_rotvec()
    return Rotation.from_rotvec(vecNormalize(vec) * angle, degrees=degrees)


def magnitudeOfProjection(vecToProject: np.array, vecToProjectOn: np.array) -> float:
    len1 = np.linalg.norm(vecToProject)
    len2 = np.linalg.norm(vecToProjectOn)
    cos = np.dot(vecToProject, vecToProjectOn) / (len1 * len2)
    return cos * len1


def angleBetweenVectors(vec1: np.array, vec2: np.array) -> float:
    len1 = np.linalg.norm(vec1)
    len2 = np.linalg.norm(vec2)
    cos = min(np.dot(vec1, vec2) / (len1 * len2), 1)
    return math.acos(cos)


def rotationToVectorFromBase(vecTo: np.array) -> Rotation:
    return rotationToVector(vectorUp, vecTo)


def vecNormalize(vec: np.array) -> np.array:
    vecNorm = np.linalg.norm(vec)
    return np.divide(vec, np.linalg.norm(vec), where=vecNorm != 0)


# returns any perpendicular vector
def getPerpendicularVector(vec: np.array) -> np.array:
    vec2 = np.copy(vec)
    for i in range(np.shape(vec)[0]):
        if vec[i] != 0:
            vec2[0 if i != 0 else 1] += vec[i]
    return np.cross(vec, vec2)


# based on https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
def rotationToVector(vecFrom: np.array, vecTo: np.array) -> Rotation:
    a = vecNormalize(vecFrom)
    b = vecNormalize(vecTo)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return noRotation if np.allclose(a, b) else Rotation.from_rotvec(180 * vecNormalize(getPerpendicularVector(a)),
                                                                         degrees=True)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return Rotation.from_matrix(rotation_matrix)

# based on https://gist.github.com/maxiimilian/67113eb1d60a5d8ceca212fbcad100c9
def multi_root(f: Callable, bracket: (float, float), n: int = 30) -> np.ndarray:
    x = np.linspace(*bracket, n)
    y = [f(arg) for arg in x]

    # Find where adjacent signs are not equal
    sign_changes = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]

    # Find roots around sign changes
    root_finders = (
        root_scalar(
            f=f,
            bracket=(x[s], x[s+1])
        )
        for s in sign_changes
    )

    roots = np.array([
        r.root if r.converged else np.nan
        for r in root_finders
    ])

    if np.any(np.isnan(roots)):
        roots = roots[~np.isnan(roots)]

    roots_unique = np.unique(roots)
    if len(roots_unique) == 0:
        root_finders = (root_scalar(f=f, x0=x0) for x0 in x)
        roots = np.array([
            r.root if r.converged else np.nan
            for r in root_finders
        ])
        if np.any(np.isnan(roots)):
            roots = roots[~np.isnan(roots)]
        roots = roots[np.where((bracket[0] <= roots) & (roots <= bracket[1]))]
        roots = np.sort(roots)
        roots = roots[np.unique(roots.round(4), return_index=True)[1]]
        return roots

    return np.unique(roots)

def apply_rotation_move(equation, move_to: np.array, rotation: Rotation):
    xn, yn, zn = symbols('xn, yn, zn')
    new_basis = [rotation.apply([1, 0, 0]), rotation.apply([0, 1, 0]), rotation.apply([0, 0, 1])]
    new_basis = [xn * base[0] + yn * base[1] + zn * base[2] for base in new_basis]
    return equation.subs([(x, new_basis[0]), (y, new_basis[1]), (z, new_basis[2])], simultanious=True).subs([(xn, x - move_to[0]), (yn, y - move_to[1]), (zn, z - move_to[2])], simultanious=True)

def build_spectrum_split(split_number: int) -> np.array:
    if split_number == 1: return np.array([[600, 1, 1, 1]])
    result = []
    wavelengths = np.linspace(400, 730, split_number)
    for wavelength in wavelengths:
        if wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation)
            G = 0.0
            B = (1.0 * attenuation)
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440))
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490))
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510))
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580))
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation)
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        result.append([wavelength, R, G, B])
    result = np.array(result)
    for i in range(1, 4):
        c_sum = result[:, i].sum()
        if c_sum == 0:
            result[:, i] = 1 / split_number
        else:
            result[:, i] /= c_sum
    return result

# for glass 1.0396, 6000, 0.2318, 20000
# for air 0, 0
def sellmeier_lambda(B1: float, C1: float, B2: float = 0, C2: float = 0) -> Callable[[float], float]:
    return lambda alpha: math.sqrt(1 + B1 / (1 - C1 / (alpha * alpha)) + B2 / (1 - C2 / (alpha * alpha)))

