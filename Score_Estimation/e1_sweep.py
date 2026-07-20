"""
Parallel sweep of the Stein-ramp end epoch (E1) for dsm_with_stein.train().

Runs N_SEEDS independent training runs at each E1 value, all against the
same fixed dataset (seeded), so the only things varying between runs are
the E1 schedule and the model's training-time randomness (init + batch
sampling). Results (per-run final loss, mean, std) are written to
e1_sweep_results.json and a mean +/- std plot to e1_sweep_results.png.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dsm_with_stein as d

E1_VALUES = [0, 500, 1000, 1500, 2000, 2500, 3000]
N_SEEDS = 5
MAX_WORKERS = 8
EPOCHS = 3000
DATA_SEED = 42

# Pre-seeded from a prior run of this sweep that was killed after completing
# these (E1, seed) pairs (see e1_sweep.log) -- resume instead of redoing them.
RESUME_RESULTS = {
    (500, 0): 386.4200,
    (0, 3): 396.6007,
    (0, 4): 389.2590,
    (0, 0): 386.3635,
    (0, 2): 378.3071,
    (0, 1): 406.2759,
}


def run_one(E1, run_seed):
    # Each worker gets one core; without this, every process spawns its own
    # multi-threaded BLAS pool and they all fight over the same 8 cores.
    torch.set_num_threads(1)
    torch.manual_seed(DATA_SEED)
    data = d.generate_scalar_data(d.F, d.a, d.b, d.c, d.sigma, d.dt, d.num_steps)
    _, loss = d.train(data, epochs=EPOCHS, E1=E1, seed=run_seed, verbose=False)
    return E1, run_seed, loss


def main():
    all_jobs = [(e1, s) for e1 in E1_VALUES for s in range(N_SEEDS)]
    results = {e1: [] for e1 in E1_VALUES}

    total = len(all_jobs)
    done = 0
    for (e1, s), loss in RESUME_RESULTS.items():
        results[e1].append(loss)
        done += 1
        print(f"[{done}/{total}] E1={e1} seed={s} loss={loss:.4f} (resumed)", flush=True)

    jobs = [job for job in all_jobs if job not in RESUME_RESULTS]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, e1, s): (e1, s) for e1, s in jobs}
        for fut in as_completed(futures):
            e1, s = futures[fut]
            _, _, loss = fut.result()
            results[e1].append(loss)
            done += 1
            elapsed = time.time() - t0
            print(
                f"[{done}/{total}] E1={e1} seed={s} loss={loss:.4f} "
                f"(elapsed {elapsed:.0f}s)",
                flush=True,
            )

    summary = {
        e1: {
            "data": results[e1],
            "mean": float(np.mean(results[e1])),
            "std": float(np.std(results[e1])),
        }
        for e1 in E1_VALUES
    }

    with open("e1_sweep_results.json", "w") as f:
        json.dump(
            {"E1_values": E1_VALUES, "n_seeds": N_SEEDS, "epochs": EPOCHS, "results": summary},
            f,
            indent=2,
        )

    means = [summary[e1]["mean"] for e1 in E1_VALUES]
    stds = [summary[e1]["std"] for e1 in E1_VALUES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(E1_VALUES, means, yerr=stds, marker="o", capsize=4)
    axes[0].set_xlabel("E1")
    axes[0].set_ylabel("Final loss (mean ± std)")
    axes[0].set_title(f"Loss vs E1 ({N_SEEDS} seeds/value)")

    axes[1].plot(E1_VALUES, stds, marker="o")
    axes[1].set_xlabel("E1")
    axes[1].set_ylabel("Std of final loss")
    axes[1].set_title("Run-to-run variability vs E1")

    fig.tight_layout()
    fig.savefig("e1_sweep_results.png", dpi=150)

    print("\n=== Summary ===")
    for e1 in E1_VALUES:
        print(f"E1={e1:5d}  mean={summary[e1]['mean']:.4f}  std={summary[e1]['std']:.4f}")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
