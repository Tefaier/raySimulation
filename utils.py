import math
from typing import Callable

import numpy as np
from scipy.optimize import root_scalar

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
