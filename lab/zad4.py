import matplotlib.pyplot as plt

def sphere_n_volume(n, r):
    """
    Calculate the volume of an n-dimensional sphere with radius r.

    Parameters:
    n (int): The dimension of the sphere.
    r (float): The radius of the sphere.

    Returns:
    float: The volume of the n-dimensional sphere.
    """
    from math import pi, gamma

    if n < 0:
        raise ValueError("Dimension n must be a non-negative integer.")
    if r < 0:
        raise ValueError("Radius r must be a non-negative number.")

    volume = ((pi ** (n / 2)) * (r ** n)) / gamma((n / 2) + 1)
    return volume

def plot_sphere_volume(volumes, r):
    plt.figure(figsize=(10, 6))

    plt.xlabel("Dimension")
    plt.ylabel("Volume")
    plt.title(f"Volume of n-Dimensional Spheres for Radius r={r}")

    # plt.yscale('log')

    plt.plot(list(range(1, 51)), volumes)

    plt.grid()

    plt.savefig(f"plots/zad4/sphere_volumes_r={r}.png", dpi=300)

volumes = []
dimensions = list(range(1, 51))
for r in [0.5, 1, 2]:
    volumes_r = []
    for dim in dimensions:
        volumes_r.append(sphere_n_volume(dim, r))
    plot_sphere_volume(volumes_r, r)
    volumes.append(volumes_r)

plt.figure(figsize=(10, 6))

plt.xlabel("Dimension")
plt.ylabel("Volume of n-Dimensional Spheres")
plt.title(f"Volume of n-Dimensional Spheres for Different Radii")

plt.yscale('log')

for i, r in enumerate([0.5, 1, 2]):
    plt.plot(dimensions, volumes[i], label=f"r={r}")

plt.grid()
plt.legend()
plt.savefig(f"plots/zad4/sphere_volumes.png", dpi=300)