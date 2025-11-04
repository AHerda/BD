import heapq
import random
import statistics as stats

P = 2_147_483_647


class Hash:
    def __init__(self, a, b, p=P):
        self.p = p
        self.a = a
        self.b = b

    def __call__(self, x: int) -> float:
        return ((self.a * x + self.b) % self.p) / self.p


class MinCount:
    def __init__(self, k, h):
        assert k >= 2
        self.k = k
        self.h = h

        self.heap = []

    def on_tick(self, x: int):
        hx = self.h(x)

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, -hx)
            return

        vk = -self.heap[0]

        if hx < vk:
            heapq.heapreplace(self.heap, -hx)

    def on_get(self) -> float:
        if len(self.heap) < self.k:
            return float(len(self.heap))

        vk = -self.heap[0]
        if vk <= 0.0:
            return float("inf")
        return (self.k - 1) / vk


def test_single_counter(k, seeds):
    n_list = [500, 1000, 2000, 5000, 10000, 50000]

    results = []
    for n in n_list:
        errs = []
        ests = []
        for seed in range(seeds):
            rng = random.Random(seed)

            a = rng.randrange(1, P)
            b = rng.randrange(0, P)
            h = Hash(a, b)
            mc = MinCount(k, h)

            for x in range(n):
                mc.on_tick(x)

            est = mc.on_get()
            ests.append(est)
            err = (est - n) / n
            errs.append(err)

        results.append({
            "n": n,
            "mean_est": stats.mean(ests),
            "mean_err": stats.mean(errs),
            "std_err": stats.pstdev(errs),
        })
    return results


def test_two_counters(k, seeds):
    n_list = [500, 1000, 2000, 5000, 10000, 50000]

    results = []
    for n in n_list:
        errs = []
        ests = []
        for seed in range(seeds):
            rng = random.Random(seed)

            a1 = rng.randrange(1, P)
            b1 = rng.randrange(0, P)
            a2 = rng.randrange(1, P)
            b2 = rng.randrange(0, P)

            h1 = Hash(a1, b1)
            h2 = Hash(a2, b2)

            c1 = MinCount(k/2, h1)
            c2 = MinCount(k/2, h2)

            for x in range(n):
                if rng.random() < 0.5:
                    c1.on_tick(x)
                else:
                    c2.on_tick(x)

            est_total = c1.on_get() + c2.on_get()
            ests.append(est_total)
            err = (est_total - n) / n
            errs.append(err)

        results.append({
            "n": n,
            "mean_est": stats.mean(ests),
            "mean_err": stats.mean(errs),
            "std_err": stats.pstdev(errs),
        })
    return results


def print_results(label, results, k):
    print(f"{label} (k = {k})")
    for r in results:
        print(
            f"n={r['n']:<8} | "
            f"mean_est={r['mean_est']:.2f} | "
            f"mean_err={r['mean_err']:+.4f} | "
            f"std_err={r['std_err']:.4f}"
        )
    print()

if __name__ == "__main__":
    K = 400
    SEEDS = 60

    single = test_single_counter(K, SEEDS)
    two = test_two_counters(K, SEEDS)

    print_results("Jeden licznik", single, K)
    print_results("Dwa liczniki (n/2 + n/2)", two, K)
