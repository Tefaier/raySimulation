import numpy as np
from models.ray import Ray
from models.surface import Surface, RefractionSurface


# returns path of ray in first argument and final ray in second argument
def shoot_ray(surfaces: list[Surface], ray: Ray, hit_limit: int) -> (list[np.array], Ray):
    path = [ray.point]
    while (not ray.finished and ray.hit_count < hit_limit):
        min_distance = 0
        min_eq_index = -1
        min_surface = None
        for surface in surfaces:
            distance, eq_index = surface.determine_intersection(ray)
            if eq_index == -1: continue
            if distance < min_distance or min_surface is None:
                min_distance = distance
                min_eq_index = eq_index
                min_surface = surface
        if min_surface is None:
            path.append(ray.point + ray.vector * 1e2)
            return path, ray
        min_surface.effect_ray(ray, min_distance, min_eq_index)
        path.append(ray.point)
        if ray.finished: return path, ray
    if ray.hit_count < hit_limit:
        path.append(ray.point + ray.vector * 1e2)
    return path, ray

def shoot_rays_spectrum(surfaces: list[Surface], rays: list[Ray], hit_limit: int) -> (list[Ray]):
    copy_ray = None
    for ray in rays:
        had_refraction = False
        while (not ray.finished and ray.hit_count < hit_limit):
            min_distance = 0
            min_eq_index = -1
            min_surface = None
            for surface in surfaces:
                distance, eq_index = surface.determine_intersection(ray)
                if eq_index == -1: continue
                if distance < min_distance or min_surface is None:
                    min_distance = distance
                    min_eq_index = eq_index
                    min_surface = surface
            if min_surface is None:
                break
            min_surface.effect_ray(ray, min_distance, min_eq_index)
            if isinstance(min_surface, RefractionSurface): had_refraction = True
            if ray.finished: break
        if not had_refraction:
            if not ray.finished:
                return rays
            else:
                copy_ray = ray
                break

    if copy_ray is not None:
        for ray in filter(lambda r: not r.finished, rays):
            ray.final_color = copy_ray.final_color_unmasked * ray.color_mask
            ray.finished = True
    return rays
