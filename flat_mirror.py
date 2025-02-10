import time

import pygame
import numpy as np

from control_functions import shoot_ray
from models.primitives import get_cube_equations, get_sphere_equations, get_triangle_equation
from models.surface import ReflectionSurface, RefractionSurface, SolidSurface
from models.ray import Ray


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


width = 120
height = 120
step_x = step_y = 10
screen = pygame.display.set_mode((width, height))

dist_to_mirror = 2
camera = Camera(np.array([-2, -2, dist_to_mirror], dtype=float),
                np.array([-2, 2, dist_to_mirror]),
                np.array([2, 2, dist_to_mirror]), 90)  # change angle of view
surfaces = []
mirror_side = 10
surfaces.append(ReflectionSurface(get_cube_equations(np.array([0, 0, -mirror_side / 2]), mirror_side), 0.99))

# surfaces.append(SolidSurface(get_cube_equations(np.array([0, 0, 80]), 2), 1, 100))
# surfaces.append(SolidSurface(get_triangle_equation(np.array([-5, 0, 40]), np.array([0, 5, 40]), np.array([5, 0, 40])), 1, 100))
surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 0, 20]), 10), 1, 100))


d_x = (camera.D - camera.A) / width * step_x
d_y = (camera.B - camera.A) / height * step_y
for i in range(width // step_x):
    start_time = time.time()
    for j in range(height // step_y):
        point = camera.A + d_x * i + d_y * j
        ray = Ray(point, point - camera.fov_pos)
        path, ray = shoot_ray(surfaces, ray, 2)
        if ray.finished:
            screen.set_at((i * step_x, height - j * step_y), tuple(map(int, np.array([255., 255., 255.]) * ray.light_level)))
            # print(np.array(path).tolist())
            # print(ray)
    print(f"{i}/{width // step_x} width iteration taked {time.time() - start_time:.2f} seconds")


clock = pygame.time.Clock()
running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(240)