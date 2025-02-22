from models.surface import Surface
import numpy as np
import pygame
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time

from control_functions import shoot_ray, shoot_rays_spectrum
from models.primitives import get_cube_equations, get_sphere_equations, get_triangle_equation, get_cylinder_equation, \
    get_hyperboloid_equation
from models.surface import ReflectionSurface, RefractionSurface, SolidSurface, SurfaceEquation
from models.ray import Ray
from sympy.abc import x, y, z

from utils import build_spectrum_split


class Camera:
    def __init__(self, A: np.array, B: np.array, C: np.array, width_fov):
        self.A = A
        self.B = B
        self.C = C
        self.D = A + C - B
        center = (A + C) / 2
        normal = np.cross(self.D - self.A, self.B - self.A)
        normal /= np.linalg.norm(normal)
        normal *= np.linalg.norm(self.A - self.D) / 2 / np.tan(np.radians(width_fov / 2))
        self.fov_pos = center + normal


def render_scene(file_name: str, surfaces: list[Surface], camera: Camera, width: int, height: int, step: int = 1, color_split: int = 1):
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    p_bar = tqdm(range(int(width * height / step / step)))
    spectrums = build_spectrum_split(color_split)

    def cast_ray(ray: Ray):
        _, ray = shoot_ray(surfaces, ray, 10)
        if ray.finished:
            screen.set_at((i * step, height - j * step), tuple(map(int, ray.final_color * 255)))
        p_bar.update()
        p_bar.refresh()

    d_x = (camera.D - camera.A) / width * step
    d_y = (camera.B - camera.A) / height * step
    for i in range(width // step):
        # start_time = time.time()
        for j in range(height // step):
            point = camera.A + d_x * i + d_y * j
            vector = point - camera.fov_pos
            pixel_color = np.array([0.0, 0.0, 0.0])
            rays = []
            for spectrum in spectrums:
                ray = Ray(camera.fov_pos, vector, spectrum[0])
                ray.color_mask = spectrum[1:]
                rays.append(ray)
            for ray in shoot_rays_spectrum(surfaces, rays, 10):
                pixel_color += ray.final_color if ray.finished else 0
            screen.set_at((i * step, height - j * step), tuple(map(int, pixel_color * 255)))
            p_bar.update()
            p_bar.refresh()
        # print(f"{i}/{width // step_x} width iteration taked {time.time() - start_time:.2f} seconds")
        pygame.event.get()
        pygame.display.update((i, 0, 1, height))

    pygame.display.flip()
    pygame.image.save(screen, file_name)
    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()
        # clock.tick(240)
