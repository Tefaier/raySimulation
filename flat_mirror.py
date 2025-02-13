import pygame
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from control_functions import shoot_ray
from models.primitives import get_cube_equations, get_sphere_equations, get_triangle_equation, get_cylinder_equation, \
    get_paraboloid_equation
from models.surface import ReflectionSurface, RefractionSurface, SolidSurface, SurfaceEquation
from models.ray import Ray
from sympy.abc import x, y, z


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

width = 100
height = 100
step_x = step_y = 1
screen = pygame.display.set_mode((width, height))
p_bar = tqdm(range(int(width * height / step_x / step_y)))

dist_to_mirror = 2
camera = Camera(np.array([-0.5, -0.5, dist_to_mirror], dtype=float),
                np.array([-0.5, 0.5, dist_to_mirror]),
                np.array([0.5, 0.5, dist_to_mirror]), 90)  # change angle of view
surfaces = []
mirror_side = 3


# surfaces.append(RefractionSurface(get_cube_equations(np.array([0.013, 0, 0.4]), 0.7), 1, 1.4))
# surfaces.append(RefractionSurface([SurfaceEquation(True, z+x,[])], 1, 2.4))
# surfaces.append(RefractionSurface([SurfaceEquation(True, z-1,[y - mirror_side, -y - mirror_side, x - mirror_side, -x - mirror_side])], 1, 1.4))
surfaces.append(SolidSurface([SurfaceEquation(SurfaceEquation.EquationType.Sphere, -z+100,[])], np.array([1, 1, 1]), 0.05))

# surfaces.append(SolidSurface(get_cube_equations(np.array([0, 0, 80]), 2), 1, 100))
# surfaces.append(SolidSurface(get_triangle_equation(np.array([-5, 0, 40]), np.array([0, 5, 40]), np.array([5, 0, 40])), 1, 100))
# surfaces.append(SolidSurface(get_sphere_equations(np.array([0, 0, 0]), 100), np.array([1, 1, 1]), 0.2))
# surfaces.append(SolidSurface(get_sphere_equations(np.array([0.2, 0, -0.5]), 0.3), np.array([1, 0, 1]), 1))



def cast_ray(ray: Ray):
    _, ray = shoot_ray(surfaces, ray, 10)
    if ray.finished:
        screen.set_at((i * step_x, height - j * step_y), tuple(map(int, ray.final_color * 255)))
    p_bar.update()
    p_bar.refresh()


d_x = (camera.D - camera.A) / width * step_x
d_y = (camera.B - camera.A) / height * step_y
for i in range(width // step_x):
    # start_time = time.time()
    for j in range(height // step_y):
        point = camera.A + d_x * i + d_y * j
        ray = Ray(point, point - camera.fov_pos)
        path, ray = shoot_ray(surfaces, ray, 100)
        if ray.finished:
            screen.set_at((i * step_x, height - j * step_y), tuple(map(int, ray.final_color * 255)))
        p_bar.update()
        p_bar.refresh()
    # print(f"{i}/{width // step_x} width iteration taked {time.time() - start_time:.2f} seconds")


clock = pygame.time.Clock()
running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()
    pygame.image.save(screen, "test.png")
    running = False
    # clock.tick(240)