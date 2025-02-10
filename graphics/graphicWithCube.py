import matplotlib.pyplot as plt
import numpy as np
from models.ray import Ray
import numpy as np
from control_functions import shoot_ray
from models.primitives import get_cube_equations
from models.surface import ReflectionSurface

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create cube
surfaces = []
surfaces.append(ReflectionSurface(get_cube_equations(np.array([0, 1, 1]), 0), 0.99))

# Draw cube
axes = [2, 2, 2]
data = np.ones(axes, dtype=np.bool_)
colors = np.empty(axes + [4], dtype=np.float32)
colors[:] = [1, 0, 0, 0.9]  # red
ax.voxels(data, facecolors=colors)

# Draw ray
vertices, ray = shoot_ray(surfaces, Ray(np.array([-9, 0, 0]), np.array([10, 1, 1])), 10)
x = []
y = []
z = []
for i in np.array(vertices).tolist():
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])
ax.scatter(x[0], y[0], z[0])
ax.plot(x, y, z)

# Settings
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_zlabel('Ось Z')
plt.xlim(-10, 2)
plt.ylim(-2, 12)
plt.show()
