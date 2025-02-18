import matplotlib.pyplot as plt
import numpy as np
from models.ray import Ray
from control_functions import shoot_ray
from models.primitives import get_converging_lens_equations, get_cube_equations
from models.surface import RefractionSurface, SolidSurface

from scipy.spatial.transform import Rotation



# Функция для рисования линзы
def draw_lens(center_1, radius_1, center_2, radius_2):
    # Создание массива для точек на границах линзы
    theta = np.linspace(0, 2 * np.pi, 100)
    x1 = center_1[0] + radius_1 * np.cos(theta)
    y1 = center_1[1] + radius_1 * np.sin(theta)

    x2 = center_2[0] + radius_2 * np.cos(theta)
    y2 = center_2[1] + radius_2 * np.sin(theta)

    # Рисование линзы
    plt.fill(x1, y1, color='lightblue', alpha=0.5)
    plt.fill(x2, y2, color='lightblue', alpha=0.5)


# Plot figure
plt.figure()

center_1 = np.array([-10, 0, 0])
center_2 = np.array([8, 0, 0])
center = center_1 / 2 + center_2 / 2
sphere_radius = 10
height = 10
surfaces = []
surfaces.append(RefractionSurface(get_converging_lens_equations(center, height, Rotation.from_rotvec([0, 0, 0]), sphere_radius=sphere_radius), 1, 1.5))
# surfaces.append(SolidSurface(get_cube_equations(np.array([9.0005, 0, 0]), (0.001, 1, 10), Rotation.from_rotvec([0, 0, 0])), 1, 0.9))

draw_lens(center_1, sphere_radius, center_2, sphere_radius)


for z in range(-5, 6):
    # Draw ray
    vertices, ray = shoot_ray(surfaces, Ray(np.array([-20, 0, z / 2]), np.array([1, 0, 0])), 10)
    print(vertices)

    x = []
    z = []
    for i in np.array(vertices).tolist():
        x.append(i[0])
        z.append(i[2])

    plt.scatter(x[0], z[0], color='red')  # Start point of the ray
    plt.plot(x, z)

# Settings
plt.xlabel('Ось X')
plt.ylabel('Ось z')
plt.xlim(-6, 10)
plt.ylim(-6, 6)
plt.grid()
plt.title('Траектории лучей через линзу')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Optional: Axis for reference
plt.legend([
    f"left_center: {center_1[0], center_1[2]}",
    f"left_radius: {sphere_radius}",
    f"right_center: {center_2[0], center_2[2]}",
    f"right_radius: {sphere_radius}",
])
# plt.show()
plt.savefig("lens_2d_graphic.png")