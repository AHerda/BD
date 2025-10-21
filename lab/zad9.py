import math
import random
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

k = 50

def d(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def test(n):
    points = [tuple(random.uniform(-1, 1) for _ in range(n)) for _ in range(k)]
    dsits = [d(a[1], b) for i, a in enumerate(points) for j, b in enumerate(points) if i < j]
    max_d = max(dsits)
    min_d = min(dsits)

    return max_d / min_d

dimensions = list(10 ** i for i in range(5))

ratios = []
trials_per_dimension = 100
for dim in dimensions:
    print(f"Testing dimension: {dim}")
    ratios_dim = []
    for trial in range(trials_per_dimension):
        print(f"\tTrial {trial + 1}/{trials_per_dimension}")
        ratios_dim.append(test(dim))
    ratios.append(ratios_dim)

# prepare flat lists for all trial points
x_vals = []
y_vals = []

for dim, rlist in zip(dimensions, ratios):
    x_vals.extend([dim] * len(rlist))
    y_vals.extend(rlist)
# compute means per dimension
means = [statistics.mean(rlist) for rlist in ratios]

# plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_vals, y=y_vals, alpha=0.7, edgecolor='w', linewidth=0.5)
sns.lineplot(x=dimensions, y=means, color='black', marker='o', markersize=8, label='mean', zorder=10)

plt.xlabel('Dimension')
plt.ylabel('dmax / dmin')
plt.xscale('log')
plt.yscale('log')
plt.xticks(dimensions)
plt.title('Ratio of max/min distances per dimension')
plt.legend()
plt.savefig('./plots/zad9_multidimensional_distance_ratio.png')
