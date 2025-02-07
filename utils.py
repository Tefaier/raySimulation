import math

import numpy as np

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
