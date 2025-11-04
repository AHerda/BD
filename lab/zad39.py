import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

PLOTS_DIR = Path(__file__).parent / "plots" / "39"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HASHING + BIT UTILS
# =============================================================================


def hash_value(item, seed=0) -> int:
    """
    64-bit hash using blake2b (digest_size=8).
    Deterministic across runs and platforms.
    """
    data = f"{seed}_{item}".encode("utf-8")
    h = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False)


def count_leading_zeros(value: int, max_bits: int) -> int:
    """
    Count leading zeros in max_bits-wide representation of value.
    Uses bit_length() for speed.
    """
    if value == 0:
        return max_bits
    return max_bits - value.bit_length()


def rho(value: int, max_bits: int) -> int:
    """
    rho(w) = position (1-indexed) of first 1-bit from the left in max_bits bits.

    IMPORTANT:
    If value == 0 (all zeros), then rho should be max_bits + 1.
    """
    return count_leading_zeros(value, max_bits) + 1


# =============================================================================
# HYPERLOG (LOGLOG)
# =============================================================================


class HyperLog:
    def __init__(self, b=10):
        self.b = b
        self.m = 1 << b
        self.registers = np.zeros(self.m, dtype=np.int8)
        self.alpha = self._compute_alpha()

    def _compute_alpha(self) -> float:
        m = self.m
        return 0.39701 - (2 * np.pi**2 + (np.log(2)) ** 2) / (48 * m)

    def add(self, item, seed=0):
        h = hash_value(item, seed=seed)

        j = h >> (64 - self.b)
        w = h & ((1 << (64 - self.b)) - 1)

        r = rho(w, 64 - self.b)
        if r > self.registers[j]:
            self.registers[j] = r

    def estimate(self) -> float:
        avg = float(np.mean(self.registers))
        return self.alpha * self.m * (2.0**avg)

    def reset(self):
        self.registers.fill(0)


# =============================================================================
# HYPERLOGLOG
# =============================================================================


class HyperLogLog:
    def __init__(self, b=10):
        self.b = b
        self.m = 1 << b
        self.registers = np.zeros(self.m, dtype=np.int8)
        self.alpha = self._compute_alpha()

        # Diagnostics from last estimate call
        self.last_raw_E = None
        self.last_V = None
        self.last_used_LC = None

    def _compute_alpha(self) -> float:
        m = self.m
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1.0 + 1.079 / m)

    def add(self, item, seed=0):
        h = hash_value(item, seed=seed)

        j = h >> (64 - self.b)
        w = h & ((1 << (64 - self.b)) - 1)

        r = rho(w, 64 - self.b)
        if r > self.registers[j]:
            self.registers[j] = r

    def estimate(self) -> float:
        M = self.registers.astype(np.float64)
        indicator = np.sum(2.0 ** (-M))
        E = self.alpha * (self.m * self.m) / indicator
        self.last_raw_E = float(E)

        # Count empty registers
        V = int(np.sum(self.registers == 0))
        self.last_V = V

        used_LC = False

        # Small-range correction (as in HLL paper / common implementations)
        if E <= 2.5 * self.m:
            if V > 0:
                E = self.m * np.log(self.m / V)
                used_LC = True

        # Large-range correction
        elif E > (1.0 / 30.0) * (2**64):
            E = -(2**64) * np.log(1.0 - E / (2**64))

        self.last_used_LC = used_LC
        return float(E)

    def reset(self):
        self.registers.fill(0)


# =============================================================================
# TESTING
# =============================================================================


def test_algorithm(algo_class, b, n_values, n_trials=30, base_seed=12345):
    """
    For each n: run n_trials independent trials and compute per-trial error stats,
    plus diagnostics on empty registers and LC usage.
    """
    m = 1 << b

    rows = []
    for n in n_values:
        estimates = np.zeros(n_trials, dtype=np.float64)
        abs_errors = np.zeros(n_trials, dtype=np.float64)
        rel_errors = np.zeros(n_trials, dtype=np.float64)
        V_list = np.zeros(n_trials, dtype=np.int32)
        used_LC_list = np.zeros(n_trials, dtype=np.int8)

        for t in range(n_trials):
            algo = algo_class(b=b)

            # Use different hash seed per trial so trials are statistically independent
            seed = base_seed + t

            for i in range(n):
                # Unique items
                algo.add(f"element_{t}_{i}", seed=seed)

            est = algo.estimate()
            estimates[t] = est
            abs_errors[t] = abs(est - n)
            rel_errors[t] = (est - n) / n if n > 0 else 0.0

            # Diagnostics (works for both; HLL has last_V too)
            if hasattr(algo, "registers"):
                V_list[t] = int(np.sum(algo.registers == 0))
            if hasattr(algo, "last_used_LC") and algo.last_used_LC is not None:
                used_LC_list[t] = 1 if algo.last_used_LC else 0
            else:
                used_LC_list[t] = 0

        bias = float(np.mean(rel_errors))
        mae = float(np.mean(abs_errors))  # Mean Absolute Error (|estimate - n|)
        mare = float(np.mean(np.abs(rel_errors)))  # Mean Absolute Relative Error
        rmse = float(np.sqrt(np.mean(rel_errors**2)))
        std = float(np.std(rel_errors))

        V_mean = float(np.mean(V_list))
        V_frac_mean = float(V_mean / m)
        V0_prob = float(np.mean(V_list == 0))
        lc_frac = float(np.mean(used_LC_list))

        rows.append(
            {
                "b": b,
                "m": m,
                "n": int(n),
                "estimate_mean": float(np.mean(estimates)),
                "estimate_std": float(np.std(estimates)),
                "bias": bias,
                "mae": mae,
                "mare": mare,
                "rmse": rmse,
                "std": std,
                "V_mean": V_mean,
                "V_frac_mean": V_frac_mean,
                "V0_prob": V0_prob,
                "lc_frac": lc_frac,
            }
        )

    return pd.DataFrame(rows)


def analyze_boundary(b_values=(8, 10, 12), n_trials=30):
    """
    Tests both algorithms on a log grid around n = m ln m, returns one DataFrame.
    """
    all_rows = []

    for b in b_values:
        m = 1 << b
        boundary = int(m * np.log(m))

        # log-spaced in [0.1*boundary, 10*boundary]
        n_values = np.unique(
            np.logspace(
                np.log10(max(1, boundary // 10)), np.log10(boundary * 10), 25
            ).astype(int)
        )

        print(f"\n[b={b}] m={m}, boundary=m ln m={boundary}, trials={n_trials}")

        df_hl = test_algorithm(HyperLog, b, n_values, n_trials=n_trials)
        df_hl["algorithm"] = "HyperLog"
        df_hl["boundary"] = boundary
        df_hl["n_over_boundary"] = df_hl["n"] / boundary
        all_rows.append(df_hl)

        df_hll = test_algorithm(HyperLogLog, b, n_values, n_trials=n_trials)
        df_hll["algorithm"] = "HyperLogLog"
        df_hll["boundary"] = boundary
        df_hll["n_over_boundary"] = df_hll["n"] / boundary
        all_rows.append(df_hll)

    return pd.concat(all_rows, ignore_index=True)


# =============================================================================
# PLOTS
# =============================================================================


def plot_comparison(df: pd.DataFrame):
    # -----------------------------
    # Plot 1: Estimates vs actual
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, b in enumerate(sorted(df["b"].unique())):
        ax = axes[idx]
        df_b = df[df["b"] == b]
        m = int(df_b["m"].iloc[0])
        boundary = int(df_b["boundary"].iloc[0])

        for algo in ["HyperLog", "HyperLogLog"]:
            d = df_b[df_b["algorithm"] == algo].sort_values("n")
            ax.plot(
                d["n"],
                d["estimate_mean"],
                marker="o",
                markersize=4,
                alpha=0.85,
                label=algo,
            )

        n_range = np.sort(df_b["n"].unique())
        ax.plot(n_range, n_range, "k--", alpha=0.5, label="Idealne")

        ax.axvline(
            boundary,
            color="red",
            linestyle=":",
            alpha=0.8,
            label=f"n=m·ln(m)={boundary}",
        )
        ax.axvline(
            int(2.5 * m),
            color="purple",
            linestyle=":",
            alpha=0.6,
            label="n≈2.5m (LC cutoff)",
        )

        ax.set_title(f"b={b}, m={m} rejestrów")
        ax.set_xlabel("Rzeczywista liczba elementów (n)")
        ax.set_ylabel("Oszacowanie (średnia z prób)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_estimates_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 2: MARE vs n/boundary  (this is the "quality" plot)
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, b in enumerate(sorted(df["b"].unique())):
        ax = axes[idx]
        df_b = df[df["b"] == b]
        m = int(df_b["m"].iloc[0])
        boundary = int(df_b["boundary"].iloc[0])

        for algo in ["HyperLog", "HyperLogLog"]:
            d = df_b[df_b["algorithm"] == algo].sort_values("n_over_boundary")
            ax.plot(
                d["n_over_boundary"],
                100.0 * d["mare"],
                marker="o",
                markersize=4,
                alpha=0.85,
                label=algo,
            )

        ax.axvline(1.0, color="red", linestyle=":", alpha=0.8, label="n = m·ln(m)")
        ax.axvline(
            (2.5 * m) / boundary,
            color="purple",
            linestyle=":",
            alpha=0.6,
            label="~LC cutoff",
        )

        ax.set_title(f"b={b}, m={m}")
        ax.set_xlabel("n / (m·ln(m))")
        ax.set_ylabel("MARE [%]")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_mare_vs_boundary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 3: Improvement (MARE_HL - MARE_HLL)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for b in sorted(df["b"].unique()):
        df_b = df[df["b"] == b]
        m = int(df_b["m"].iloc[0])
        boundary = int(df_b["boundary"].iloc[0])

        hl = (
            df_b[df_b["algorithm"] == "HyperLog"]
            .set_index("n_over_boundary")
            .sort_index()
        )
        hll = (
            df_b[df_b["algorithm"] == "HyperLogLog"]
            .set_index("n_over_boundary")
            .sort_index()
        )

        common = hl.index.intersection(hll.index)
        improvement = 100.0 * (hl.loc[common, "mare"] - hll.loc[common, "mare"])

        ax.plot(
            common,
            improvement,
            marker="o",
            markersize=4,
            alpha=0.85,
            label=f"m={m} (b={b})",
        )

    ax.axhline(0, color="black", alpha=0.3)
    ax.axvline(1.0, color="red", linestyle=":", alpha=0.8, label="n = m·ln(m)")
    ax.set_xscale("log")
    ax.set_xlabel("n / (m·ln(m))")
    ax.set_ylabel("Poprawa MARE: HyperLog - HyperLogLog [pp]")
    ax.set_title("Różnica jakości (MARE)\n(wartości dodatnie = HyperLogLog lepszy)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_hll_improvement_mare.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 4: Empty registers diagnostics (this shows the boundary effect!)
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, b in enumerate(sorted(df["b"].unique())):
        ax = axes[idx]
        df_b = df[df["b"] == b]
        m = int(df_b["m"].iloc[0])
        boundary = int(df_b["boundary"].iloc[0])

        # Show only one algorithm's V behavior (same registers update rule), pick HLL
        d = df_b[df_b["algorithm"] == "HyperLogLog"].sort_values("n_over_boundary")

        ax.plot(
            d["n_over_boundary"],
            d["V_frac_mean"],
            marker="o",
            markersize=4,
            alpha=0.9,
            label="E[V]/m",
        )
        ax.plot(
            d["n_over_boundary"],
            d["V0_prob"],
            marker="o",
            markersize=4,
            alpha=0.9,
            label="P(V=0)",
        )

        ax.axvline(1.0, color="red", linestyle=":", alpha=0.8, label="n = m·ln(m)")
        ax.axvline(
            (2.5 * m) / boundary,
            color="purple",
            linestyle=":",
            alpha=0.6,
            label="~LC cutoff",
        )

        ax.set_title(f"b={b}, m={m}")
        ax.set_xlabel("n / (m·ln(m))")
        ax.set_ylabel("Wartość")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_empty_registers.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 5: MAE (Mean Absolute Error) vs n
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, b in enumerate(sorted(df["b"].unique())):
        ax = axes[idx]
        df_b = df[df["b"] == b]
        m = int(df_b["m"].iloc[0])
        boundary = int(df_b["boundary"].iloc[0])

        for algo in ["HyperLog", "HyperLogLog"]:
            d = df_b[df_b["algorithm"] == algo].sort_values("n")
            ax.plot(d["n"], d["mae"], marker="o", markersize=4, alpha=0.85, label=algo)

        ax.axvline(
            boundary,
            color="red",
            linestyle=":",
            alpha=0.8,
            label=f"n=m·ln(m)={boundary}",
        )
        ax.axvline(
            int(2.5 * m),
            color="purple",
            linestyle=":",
            alpha=0.6,
            label="n≈2.5m (LC cutoff)",
        )

        ax.set_title(f"b={b}, m={m}")
        ax.set_xlabel("Rzeczywista liczba elementów (n)")
        ax.set_ylabel("MAE (średni błąd bezwzględny)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_mae_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wykresy zapisane w: {PLOTS_DIR}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("ZADANIE 39: HyperLog vs HyperLogLog")
    print("Poprawne metryki + analiza granicy n = m·ln(m)")
    print("=" * 70)

    b_values = [8, 10, 12]
    n_trials = 30  # increase to see boundary variance clearly

    df = analyze_boundary(b_values=b_values, n_trials=n_trials)
    plot_comparison(df)

    # Quick summary table
    print("\nPODSUMOWANIE (średnie MARE [%] poniżej i powyżej granicy):")
    for b in b_values:
        df_b = df[df["b"] == b]
        below = df_b[df_b["n_over_boundary"] < 1.0]
        above = df_b[df_b["n_over_boundary"] >= 1.0]

        m = int(df_b["m"].iloc[0])
        boundary = int(df_b["boundary"].iloc[0])
        print(f"\nb={b}, m={m}, boundary={boundary}")

        for algo in ["HyperLog", "HyperLogLog"]:
            mare_below = 100.0 * below[below["algorithm"] == algo]["mare"].mean()
            mare_above = 100.0 * above[above["algorithm"] == algo]["mare"].mean()
            print(
                f"  {algo:12s} | MARE below: {mare_below:6.2f}% | MARE above: {mare_above:6.2f}%"
            )

    print(f"\nWYKRESY: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
