import numpy as np
from models.ray import Ray
from models.surface import Surface


# returns path of ray in first argument and final ray in second argument
def shoot_ray(surfaces: list[Surface], ray: Ray, hit_limit: int) -> (list[np.array], Ray):
    path = [ray.point]
    while (not ray.finished and ray.hit_count <= hit_limit):
        min_distance = 0
        min_eq_index = -1
        min_surface = None
        for surface in surfaces:
            distance, eq_index = surface.determine_intersection(ray)
            if eq_index == -1: continue
            if distance < min_distance:
                min_distance = min_distance
                min_eq_index = eq_index
                min_surface = surface
        if min_surface is None:
            path.append(ray.point + ray.vector * 1e2)
            return path, ray
        ray = min_surface.effect_ray(ray, min_distance, min_eq_index)
        path.append(ray.point)
        if ray.finished: return path, ray
    path.append(ray.point + ray.vector * 1e2)
    return path, ray