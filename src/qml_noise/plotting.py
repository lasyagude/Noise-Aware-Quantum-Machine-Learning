"""Plots (assignment section 13), generated ONLY from saved result files.

Nothing here runs a circuit, trains a model, or invents a number: every
plotted value is read from ``results/*.jsonl`` and aggregated. Each figure
carries a provenance footnote naming its source file and the number of
seeds actually found (never an assumed 5).

Required plots (assignment numbering)
-------------------------------------
1. ``01_degradation_vs_noise.png``        accuracy degradation + TVD vs noise level
2. ``02_accuracy_vs_depth.png``           accuracy vs logical circuit depth
3. ``03_noise_comparison.png`` (+ table)  accuracy per noise condition; the table
                                          is ``noise_comparison_table.csv/.md``
4. ``04_qubits_vs_performance.png``       accuracy and transpiled depth vs qubits
5. ``05_output_distributions.png``        ideal vs noisy measurement distributions
6. ``06_classical_vs_qml.png``            LogReg/SVM vs ideal vs noisy VQC
7. ``07_standard_vs_adaptive.png``        standard VQC baselines vs AN-VQC, ideal + medium noise
8. ``08_accuracy_mean_std.png``           mean +/- std across seeds, all experiments

Extension plots (``extension_*.png``) cover the noise sweep and Breast Cancer.
They are only drawn when the extension's result files are COMPLETE (every
expected ``(experiment, seed)`` pair present); a partial run is skipped with
a logged reason rather than plotted as if it were finished.

Plot 7 reads ``results/an_vqc_results.jsonl`` (methods ``fixed_vqc``/
``ideal_best``/``shallow_vqc``/``an_vqc``, each with ``ideal`` and
``medium`` rows) and ``results/an_vqc_classical_results.jsonl``
(``logistic_regression``/``svm``). It is skipped if either file is missing,
empty, or any of the four quantum methods lacks both noise conditions.

Colour: categorical slots 1-3 of the dataviz skill's validated palette
(blue, orange, aqua; passes the all-pairs CVD/normal-vision checks). The
aqua slot is below 3:1 contrast on a light surface, so every chart also has
distinct markers/hatching and value labels, and tables are saved as CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import ParameterVector  # noqa: E402

from qml_noise.circuits import build_feature_map  # noqa: E402
from qml_noise.config import (  # noqa: E402
    NOISE_SWEEP_TWO_QUBIT_PROBABILITIES,
    PLOTS_DIR,
    RANDOM_SEEDS,
    get_logger,
)
from qml_noise.experiments import (  # noqa: E402
    CLASSICAL_RESULTS_FILE,
    EXPERIMENT_MATRIX,
    VQC_RESULTS_FILE,
    load_jsonl,
)
from qml_noise.extensions import (  # noqa: E402
    BC_CLASSICAL_FILE,
    BC_RESULTS_FILE,
    NOISE_SWEEP_ARCHITECTURES,
    NOISE_SWEEP_FILE,
)
from qml_noise.an_vqc import (  # noqa: E402
    AN_VQC_CLASSICAL_FILE,
    AN_VQC_RESULTS_FILE,
)

logger = get_logger(__name__)

# Validated categorical slots 1-3 (light surface) + neutral ink tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, INK, INK2, GRID, SURFACE = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
NOISE_COLORS = {"ideal": BLUE, "low": ORANGE, "medium": AQUA}
NOISE_ORDER = ("ideal", "low", "medium")
ARCH_COLORS = {2: BLUE, 4: ORANGE, 6: AQUA}
ARCH_MARKERS = {2: "o", 4: "s", 6: "^"}

_META = {eid: (q, d, noise) for eid, q, d, noise in EXPERIMENT_MATRIX}


class PlotSkipped(Exception):
    """Raised when a plot cannot be drawn from the saved results (with the reason)."""


@dataclass
class PlotReport:
    generated: list[Path] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.family": "sans-serif",
        }
    )


def _frame(path: Path) -> pd.DataFrame:
    return pd.DataFrame(load_jsonl(path))


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "status" in df.columns:
        df = df[df["status"] == "completed"]
    return df.reset_index(drop=True)


def _require(df: pd.DataFrame, ids: list[str], label: str) -> None:
    if df.empty:
        raise PlotSkipped(f"{label}: no saved results found")
    present = set(df["experiment_id"])
    missing = [i for i in ids if i not in present]
    if missing:
        raise PlotSkipped(f"{label}: no saved results for {missing}")


def _stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("experiment_id")[col]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()})


def _n_seeds(df: pd.DataFrame) -> int:
    return int(df["seed"].nunique()) if not df.empty else 0


def _footnote(fig, source: str, n: int, extra: str = "", *, encoding_note: bool = True) -> None:
    text = f"Source: {source}. n = {n} seed(s)."
    if encoding_note:
        text += " Markers/bars = mean, error bars = population std, dots = individual seeds."
    if extra:
        text += " " + extra
    fig.text(0.01, 0.005, text, fontsize=7, color=INK2, ha="left", va="bottom")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def _seed_dots(ax, x: float, values, color: str, marker: str = "o") -> None:
    values = np.asarray(list(values), dtype=float)
    jitter = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else np.zeros(1)
    ax.scatter(x + jitter, values, s=13, color=color, alpha=0.38, marker=marker, zorder=2, linewidths=0)


def _classical_means(classical: pd.DataFrame) -> dict[str, tuple[float, float, int]]:
    out = {}
    if classical.empty:
        return out
    for model, grp in classical.groupby("model_name"):
        out[model] = (float(grp["accuracy"].mean()), float(grp["accuracy"].std(ddof=0)), int(len(grp)))
    return out


# ---------------------------------------------------------------------------
# architecture diagram
# ---------------------------------------------------------------------------


def _draw_only_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    """A circuit for DIAGRAMS ONLY -- gate-by-gate identical to
    ``circuits.build_vqc_circuit`` (same encoding, same per-block RY/RZ/CNOT-ring
    structure, same parameter count and indexing), with one addition: a
    ``barrier()`` after the encoding and after every ansatz block.

    Qiskit's mpl drawer packs each wire's gates as early as that wire's own
    dependencies allow (ASAP scheduling), not in program-append order. Without
    a barrier, a qubit whose ring-CNOT finishes early can have the *next*
    block's RY drawn before another qubit's *current* block has finished --
    visually interleaving two blocks even though the actual circuit (and its
    simulation) is correct. Barriers force every wire to re-synchronize at the
    block boundary so each block renders as its own column group. This
    function is never used for simulation, training, or any saved metric --
    only ``build_vqc_circuit`` is -- so it cannot affect any result in
    ``results/``.
    """
    feature_map, _ = build_feature_map(n_qubits)
    weights = ParameterVector("w", 2 * n_qubits * depth)
    qc = QuantumCircuit(n_qubits, name=f"vqc_q{n_qubits}_d{depth}_diagram")
    qc.compose(feature_map, inplace=True)
    qc.barrier()
    k = 0
    for block in range(depth):
        for i in range(n_qubits):
            qc.ry(weights[k], i)
            k += 1
        for i in range(n_qubits):
            qc.rz(weights[k], i)
            k += 1
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
        if block < depth - 1:
            qc.barrier()  # last block's boundary is already marked by measure_all()'s own barrier
    qc.measure_all()
    return qc


def plot_circuit_diagram(
    out_dir: Path = PLOTS_DIR, *, n_qubits: int = 4, depth: int = 2
) -> Path:
    """Qiskit ``draw("mpl")`` of the logical (un-transpiled) VQC circuit.

    No circuit is simulated or trained here. Uses a diagram-only circuit
    (see ``_draw_only_circuit``) with block-separating barriers purely so
    the rendering doesn't visually interleave blocks; the underlying gate
    sequence is identical to the real training circuit.
    """
    qc = _draw_only_circuit(n_qubits, depth)
    fig = qc.draw(
        "mpl", fold=-1, style={"backgroundcolor": SURFACE}, plot_barriers=True
    )
    fig.set_size_inches(fig.get_size_inches()[0] * 1.15, fig.get_size_inches()[1])
    fig.suptitle(
        f"VQC circuit: {n_qubits} qubits, logical depth {depth} "
        f"(angle encoding H+RY(x), then {depth} x [RY, RZ, CNOT ring], measure all; "
        f"barriers mark block boundaries for readability only)",
        fontsize=10,
        y=1.02,
    )
    return _save(fig, Path(out_dir) / f"circuit_{n_qubits}q_depth{depth}.png")


# ---------------------------------------------------------------------------
# required plots 1-8
# ---------------------------------------------------------------------------


def plot_degradation_vs_noise(main: pd.DataFrame, out_dir: Path, source: str) -> Path:
    groups = [(4, ["E4", "E5", "E6"]), (6, ["E7", "E8"])]
    _require(main, [i for _, ids in groups for i in ids], "degradation vs noise")
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    panels = [
        ("degradation_pp", "Accuracy degradation (percentage points)\n(negative = noisy model scored higher)"),
        ("distribution_total_variation_distance", "Output-distribution TVD vs ideal"),
    ]
    # Per-series label offset: both series can land on the exact same value
    # (e.g. both are 0 at "ideal" by construction), so a single fixed offset
    # stacks their text labels exactly on top of each other. The degradation
    # panel has headroom on both sides of 0, so mirroring one series above
    # and one below its marker keeps them apart. The TVD panel's floor is 0
    # (no negative values) -- a downward offset there collides with the
    # x-axis tick labels instead, so both series go upward there, staggered
    # at different heights so they still don't stack on each other.
    mirrored_offset = {4: (12, 8), 6: (12, -17)}
    stacked_offset = {4: (10, 19), 6: (10, 5)}
    for ax, (col, ylabel) in zip(axes, panels):
        label_offset = mirrored_offset if col == "degradation_pp" else stacked_offset
        st = _stats(main, col)
        for q, ids in groups:
            xs = [NOISE_ORDER.index(_META[i][2]) for i in ids]
            skips_low = len(ids) < len(NOISE_ORDER)  # the matrix has no 6-qubit "low" run
            ax.errorbar(
                xs, st.loc[ids, "mean"], yerr=st.loc[ids, "std"], color=ARCH_COLORS[q],
                marker=ARCH_MARKERS[q], capsize=3, lw=1.6, ms=6, zorder=3,
                ls="--" if skips_low else "-",
                label=f"{q} qubits, depth 4" + (" (no 'low' run; dashed)" if skips_low else ""),
            )
            for x, i in zip(xs, ids):
                _seed_dots(ax, x, main.loc[main.experiment_id == i, col], ARCH_COLORS[q], ARCH_MARKERS[q])
                ax.annotate(f"{st.loc[i, 'mean']:.3g}", (x, st.loc[i, "mean"]), textcoords="offset points",
                            xytext=label_offset[q], fontsize=8, color=INK2,
                            va="bottom" if label_offset[q][1] > 0 else "top")
        ax.set_xticks(range(len(NOISE_ORDER)), NOISE_ORDER)
        ax.set_xlabel("Noise condition")
        ax.set_ylabel(ylabel)
        if col == "degradation_pp":
            ax.axhline(0, color=NEUTRAL, lw=1, ls="--")
    axes[0].legend(loc="upper left")
    fig.suptitle("Accuracy degradation and output-distribution shift vs noise level", fontsize=12)
    _footnote(fig, source, _n_seeds(main), "Degradation is paired per seed against the same model's ideal accuracy.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, out_dir / "01_degradation_vs_noise.png")


def plot_accuracy_vs_depth(main: pd.DataFrame, out_dir: Path, source: str, classical: pd.DataFrame) -> Path:
    ideal_ids = [i for i, (_, _, n) in _META.items() if n == "ideal"]
    _require(main, ideal_ids, "accuracy vs depth")
    _style()
    st = _stats(main, "accuracy")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for q in (2, 4, 6):
        ids = [i for i in ideal_ids if _META[i][0] == q]
        ids.sort(key=lambda i: _META[i][1])
        xs = [_META[i][1] for i in ids]
        ax.errorbar(xs, st.loc[ids, "mean"], yerr=st.loc[ids, "std"], color=ARCH_COLORS[q],
                    marker=ARCH_MARKERS[q], capsize=3, lw=1.6, ms=6, label=f"{q} qubits (ideal)", zorder=3)
        for x, i in zip(xs, ids):
            _seed_dots(ax, x, main.loc[main.experiment_id == i, "accuracy"], ARCH_COLORS[q], ARCH_MARKERS[q])
            ax.annotate(f"{st.loc[i, 'mean']:.2f}", (x, st.loc[i, "mean"]), textcoords="offset points",
                        xytext=(8, -12), fontsize=8, color=INK2)
    by_mean: dict[float, list[str]] = {}
    for model, (mean, _, _) in _classical_means(classical).items():
        by_mean.setdefault(round(mean, 4), []).append(model.replace("_", " "))
    for mean, names in by_mean.items():
        ax.axhline(mean, color=NEUTRAL, lw=1, ls=":")
        ax.annotate(f"{' / '.join(names)} ({mean:.2f})", (3.0, mean), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks([2, 4])
    ax.set_xlabel("Logical ansatz depth (repeated blocks)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs circuit depth (ideal simulation)")
    ax.legend(loc="lower right")
    _footnote(fig, source, _n_seeds(main))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out_dir / "02_accuracy_vs_depth.png")


def noise_comparison_table(main: pd.DataFrame) -> pd.DataFrame:
    """Per-experiment summary (mean / population std over the seeds present)."""
    _require(main, list(_META), "noise comparison table")
    rows = []
    for eid, (q, d, noise) in _META.items():
        g = main[main.experiment_id == eid]
        row = {"experiment": eid, "qubits": q, "depth": d, "noise": noise, "n_seeds": len(g)}
        for col, name in [
            ("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("f1", "f1"),
            ("degradation_pp", "degradation_pp"), ("relative_degradation_pct", "rel_degradation_pct"),
            ("robustness_ratio", "robustness_ratio"),
            ("distribution_total_variation_distance", "tvd"),
            ("distribution_hellinger_distance", "hellinger"),
        ]:
            row[f"{name}_mean"] = float(g[col].mean())
            row[f"{name}_std"] = float(g[col].std(ddof=0))
        for col in ("transpiled_depth", "total_gate_count", "two_qubit_gate_count", "n_trainable_parameters", "shots"):
            row[col] = int(g[col].iloc[0])
        row["runtime_s_mean"] = float(g["runtime_seconds"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    def cell(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def plot_noise_comparison(main: pd.DataFrame, out_dir: Path, source: str) -> Path:
    table = noise_comparison_table(main)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "noise_comparison_table.csv", index=False)
    (out_dir / "noise_comparison_table.md").write_text(
        _markdown_table(
            table,
            ["experiment", "qubits", "depth", "noise", "n_seeds", "accuracy_mean", "accuracy_std",
             "degradation_pp_mean", "degradation_pp_std", "robustness_ratio_mean", "tvd_mean", "tvd_std"],
        ),
        encoding="utf-8",
    )
    logger.info("Wrote noise comparison table (csv + md) to %s", out_dir)

    _style()
    st = _stats(main, "accuracy")
    groups = [(4, ["E4", "E5", "E6"]), (6, ["E7", "E8"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width, pos, ticks, tick_labels = 0.26, 0.0, [], []
    for q, ids in groups:
        start = pos
        for i in ids:
            noise = _META[i][2]
            ax.bar(pos, st.loc[i, "mean"], width=width, color=NOISE_COLORS[noise],
                   yerr=st.loc[i, "std"], capsize=3, error_kw={"ecolor": INK2, "lw": 1}, zorder=3,
                   hatch={"ideal": "", "low": "//", "medium": "xx"}[noise], edgecolor=SURFACE, linewidth=0.6)
            _seed_dots(ax, pos, main.loc[main.experiment_id == i, "accuracy"], INK2)
            ax.annotate(f"{st.loc[i, 'mean']:.2f}", (pos, st.loc[i, "mean"] + st.loc[i, "std"]),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=INK)
            pos += width + 0.04
        ticks.append((start + pos - width - 0.04) / 2)
        tick_labels.append(f"{q} qubits, depth 4")
        pos += 0.35
    ax.set_xticks(ticks, tick_labels)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Noise comparison: same trained weights evaluated under each noise condition")
    ax.legend(handles=[Patch(facecolor=NOISE_COLORS[n], hatch={"ideal": "", "low": "//", "medium": "xx"}[n],
                             edgecolor=SURFACE, label=n) for n in NOISE_ORDER], loc="upper right", ncol=3)
    _footnote(fig, source, _n_seeds(main), "Full table: noise_comparison_table.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out_dir / "03_noise_comparison.png")


def plot_qubits_vs_performance(main: pd.DataFrame, out_dir: Path, source: str) -> Path:
    series = [
        ("ideal, depth 2", ["E1", "E3"], BLUE, "o"),
        ("ideal, depth 4", ["E2", "E4", "E7"], ORANGE, "s"),
        ("medium noise, depth 4", ["E6", "E8"], AQUA, "^"),
    ]
    _require(main, [i for _, ids, _, _ in series for i in ids], "qubits vs performance")
    _style()
    st = _stats(main, "accuracy")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for label, ids, color, marker in series:
        xs = [_META[i][0] for i in ids]
        ax.errorbar(xs, st.loc[ids, "mean"], yerr=st.loc[ids, "std"], color=color, marker=marker,
                    capsize=3, lw=1.6, ms=6, label=label, zorder=3)
        for x, i in zip(xs, ids):
            _seed_dots(ax, x, main.loc[main.experiment_id == i, "accuracy"], color, marker)
    ax.set_xticks([2, 4, 6])
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs qubit count")
    ax.legend(loc="lower left")

    ideal_ids = [i for i, (_, _, n) in _META.items() if n == "ideal"]
    for depth, color, marker in ((2, BLUE, "o"), (4, ORANGE, "s")):
        pts = sorted((_META[i][0], int(main.loc[main.experiment_id == i, "transpiled_depth"].iloc[0]))
                     for i in ideal_ids if _META[i][1] == depth)
        ax2.plot(*zip(*pts), color=color, marker=marker, lw=1.6, ms=6, label=f"logical depth {depth}")
        for x, y in pts:
            ax2.annotate(str(y), (x, y), textcoords="offset points", xytext=(6, -11), fontsize=8, color=INK2)
    ax2.set_xticks([2, 4, 6])
    ax2.set_xlabel("Number of qubits")
    ax2.set_ylabel("Transpiled circuit depth")
    ax2.set_title("Circuit cost vs qubit count")
    ax2.legend(loc="upper left")
    fig.suptitle("Qubit count vs performance and cost", fontsize=12)
    _footnote(fig, source, _n_seeds(main), "Transpiled depth is deterministic (no error bars).")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, out_dir / "04_qubits_vs_performance.png")


def _mean_distribution(df: pd.DataFrame, eid: str) -> dict[str, float]:
    dists = list(df.loc[df.experiment_id == eid, "pooled_output_distribution"])
    keys = set().union(*dists) if dists else set()
    return {k: float(np.mean([d.get(k, 0.0) for d in dists])) for k in keys}


def plot_output_distributions(main: pd.DataFrame, out_dir: Path, source: str, top_k: int = 10) -> Path:
    groups = [(4, ["E4", "E5", "E6"]), (6, ["E7", "E8"])]
    _require(main, [i for _, ids in groups for i in ids], "output distributions")
    _style()
    tvd = _stats(main, "distribution_total_variation_distance")["mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (q, ids) in zip(axes, groups):
        dists = {i: _mean_distribution(main, i) for i in ids}
        ref = dists[ids[0]]
        top = sorted(ref, key=ref.get, reverse=True)[:top_k]
        labels = top + ["other"]
        width = 0.8 / len(ids)
        for k, i in enumerate(ids):
            noise = _META[i][2]
            vals = [dists[i].get(b, 0.0) for b in top]
            vals.append(max(0.0, 1.0 - sum(vals)))
            ax.bar(np.arange(len(labels)) + (k - (len(ids) - 1) / 2) * width, vals, width=width * 0.92,
                   color=NOISE_COLORS[noise], hatch={"ideal": "", "low": "//", "medium": "xx"}[noise],
                   edgecolor=SURFACE, linewidth=0.5, zorder=3,
                   label=f"{noise}" + ("" if noise == "ideal" else f" (TVD {tvd[i]:.3f})"))
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_xlabel("Measured bitstring (qubit 0 rightmost)")
        ax.set_ylabel("Mean probability over seeds")
        ax.set_title(f"{q} qubits, depth 4")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.legend(loc="upper center")
    fig.suptitle("Output-distribution comparison: ideal vs noisy (same trained weights)", fontsize=12)
    _footnote(fig, source, _n_seeds(main),
              f"Pooled over all test circuits, then averaged over seeds; top {top_k} bitstrings of the ideal run.",
              encoding_note=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, out_dir / "05_output_distributions.png")


def plot_classical_vs_qml(main: pd.DataFrame, classical: pd.DataFrame, out_dir: Path, source: str) -> Path:
    ids = ["E4", "E5", "E6", "E7", "E8"]
    _require(main, ids, "classical vs QML")
    cl = _classical_means(classical)
    if not cl:
        raise PlotSkipped("classical vs QML: no classical baseline results found")
    _style()
    st = _stats(main, "accuracy")
    bars = []
    for model in sorted(cl):
        mean, std, _ = cl[model]
        bars.append((model.replace("_", " "), mean, std, NEUTRAL, "//", classical[classical.model_name == model]["accuracy"]))
    for i in ids:
        q, _, noise = _META[i]
        bars.append((f"VQC {q}q\n{noise}", st.loc[i, "mean"], st.loc[i, "std"], NOISE_COLORS[noise],
                     {"ideal": "", "low": "//", "medium": "xx"}[noise], main.loc[main.experiment_id == i, "accuracy"]))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for x, (label, mean, std, color, hatch, vals) in enumerate(bars):
        ax.bar(x, mean, 0.62, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0.6,
               yerr=std, capsize=3, error_kw={"ecolor": INK2, "lw": 1}, zorder=3)
        _seed_dots(ax, x, vals, INK2)
        ax.annotate(f"{mean:.2f}", (x, mean + std), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Classical baselines vs ideal QML vs noisy QML (Iris setosa/versicolor)")
    ax.legend(handles=[Patch(facecolor=NEUTRAL, hatch="//", edgecolor=SURFACE, label="classical")] +
              [Patch(facecolor=NOISE_COLORS[n], hatch={"ideal": "", "low": "//", "medium": "xx"}[n],
                     edgecolor=SURFACE, label=f"VQC, {n}") for n in NOISE_ORDER], loc="upper center", ncol=4)
    _footnote(fig, source + " + classical_baseline_results.jsonl", _n_seeds(main))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out_dir / "06_classical_vs_qml.png")


_AN_VQC_METHOD_ORDER: tuple[str, ...] = ("fixed_vqc", "ideal_best", "shallow_vqc", "an_vqc")
_AN_VQC_METHOD_LABELS: dict[str, str] = {
    "fixed_vqc": "fixed\n(4q, d4)",
    "ideal_best": "ideal-\nvalidation-best",
    "shallow_vqc": "shallow\n(2q, d2)",
    "an_vqc": "AN-VQC",
}


def plot_standard_vs_adaptive(
    an_vqc: pd.DataFrame, an_vqc_classical: pd.DataFrame, out_dir: Path, source: str
) -> Path:
    """Standard VQC baselines vs AN-VQC, plus Logistic Regression/SVM, ideal and medium noise.

    Reads ``results/an_vqc_results.jsonl`` (methods ``fixed_vqc``/``ideal_best``/``shallow_vqc``/
    ``an_vqc``, each with ``ideal`` and ``medium`` rows) and
    ``results/an_vqc_classical_results.jsonl`` (``logistic_regression``/``svm``, no noise axis).
    """
    if an_vqc.empty:
        raise PlotSkipped("standard vs adaptive VQC: results/an_vqc_results.jsonl not found / empty")
    missing = {"method", "seed", "accuracy", "noise_condition", "status"} - set(an_vqc.columns)
    if missing:
        raise PlotSkipped(f"standard vs adaptive VQC: an_vqc results missing columns {sorted(missing)}")
    an_vqc = _completed(an_vqc)
    present_methods = set(an_vqc["method"])
    missing_methods = set(_AN_VQC_METHOD_ORDER) - present_methods
    if missing_methods:
        raise PlotSkipped(f"standard vs adaptive VQC: missing method(s) {sorted(missing_methods)}")
    for method in _AN_VQC_METHOD_ORDER:
        sub = an_vqc[an_vqc.method == method]
        conditions = set(sub["noise_condition"])
        if not {"ideal", "medium"} <= conditions:
            raise PlotSkipped(f"standard vs adaptive VQC: method '{method}' missing ideal/medium rows")
    if an_vqc_classical.empty:
        raise PlotSkipped("standard vs adaptive VQC: results/an_vqc_classical_results.jsonl not found / empty")

    _style()
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    width, group_gap = 0.34, 0.5
    pos = 0.0
    ticks, tick_labels = [], []
    for method in _AN_VQC_METHOD_ORDER:
        start = pos
        for condition in ("ideal", "medium"):
            sub = an_vqc[(an_vqc.method == method) & (an_vqc.noise_condition == condition)]
            mean, std = sub["accuracy"].mean(), sub["accuracy"].std(ddof=0)
            ax.bar(pos, mean, width=width, color=NOISE_COLORS[condition],
                   hatch={"ideal": "", "medium": "xx"}[condition], edgecolor=SURFACE, linewidth=0.6,
                   yerr=std, capsize=3, error_kw={"ecolor": INK2, "lw": 1}, zorder=3)
            _seed_dots(ax, pos, sub["accuracy"], INK2)
            ax.annotate(f"{mean:.2f}", (pos, mean + std), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8)
            pos += width + 0.04
        ticks.append((start + pos - width - 0.04) / 2)
        tick_labels.append(_AN_VQC_METHOD_LABELS[method])
        pos += group_gap

    # Classical baselines: real bars (one each, no noise axis), not a reference
    # line -- a dotted line at the top of the chart reads as an unexplained
    # ceiling rather than a directly comparable data point.
    cl_means = _classical_means(an_vqc_classical)
    for model_name in ("logistic_regression", "svm"):
        if model_name not in cl_means:
            continue
        mean, std, _ = cl_means[model_name]
        sub = an_vqc_classical[an_vqc_classical.model_name == model_name]["accuracy"]
        ax.bar(pos, mean, width=width * 1.7, color=NEUTRAL, hatch="//", edgecolor=SURFACE,
               linewidth=0.6, yerr=std, capsize=3, error_kw={"ecolor": INK2, "lw": 1}, zorder=3)
        _seed_dots(ax, pos, sub, INK2)
        ax.annotate(f"{mean:.2f}", (pos, mean + std), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
        ticks.append(pos)
        tick_labels.append(model_name.replace("_", " ").replace(" ", "\n"))
        pos += width * 1.7 + group_gap

    ax.set_xlim(-0.5, pos - group_gap + 0.5)
    ax.set_xticks(ticks, tick_labels)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Standard VQC baselines vs AN-VQC, and classical baselines")
    ax.legend(
        handles=[Patch(facecolor=NOISE_COLORS["ideal"], edgecolor=SURFACE, label="ideal"),
                 Patch(facecolor=NOISE_COLORS["medium"], hatch="xx", edgecolor=SURFACE, label="medium"),
                 Patch(facecolor=NEUTRAL, hatch="//", edgecolor=SURFACE, label="classical (no noise axis)")],
        loc="lower right",
    )
    _footnote(fig, source, _n_seeds(an_vqc),
              "AN-VQC's architecture (qubits, depth) can differ per seed; see report/report.md Section 9.4. "
              "Classical bars use results/an_vqc_classical_results.jsonl (same 3 seeds).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out_dir / "07_standard_vs_adaptive.png")


def plot_accuracy_mean_std(main: pd.DataFrame, classical: pd.DataFrame, out_dir: Path, source: str) -> Path:
    _require(main, list(_META), "mean/std across seeds")
    _style()
    st = _stats(main, "accuracy")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for x, i in enumerate(_META):
        noise = _META[i][2]
        ax.errorbar(x, st.loc[i, "mean"], yerr=st.loc[i, "std"], color=NOISE_COLORS[noise],
                    marker=ARCH_MARKERS[_META[i][0]], ms=7, capsize=4, lw=1.6, zorder=3)
        _seed_dots(ax, x, main.loc[main.experiment_id == i, "accuracy"], NOISE_COLORS[noise],
                   ARCH_MARKERS[_META[i][0]])
        ax.annotate(f"{st.loc[i, 'mean']:.2f}\n±{st.loc[i, 'std']:.2f}", (x, st.loc[i, "mean"]),
                    textcoords="offset points", xytext=(11, -4), fontsize=7.5, color=INK2)
    for model, (mean, _, _) in _classical_means(classical).items():
        ax.axhline(mean, color=NEUTRAL, lw=1, ls=":")
        ax.annotate(model.replace("_", " "), (len(_META) - 0.5, mean), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(_META)), [f"{i}\n{_META[i][0]}q d{_META[i][1]}\n{_META[i][2]}" for i in _META])
    ax.set_ylim(0.6, 1.08)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy per experiment: mean ± std across seeds")
    ax.legend(handles=[plt.Line2D([], [], color=NOISE_COLORS[n], marker="o", ls="", label=n) for n in NOISE_ORDER],
              loc="lower left", ncol=3, title="noise", title_fontsize=8)
    _footnote(fig, source, _n_seeds(main), "Marker shape = qubit count (o 2, square 4, triangle 6).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out_dir / "08_accuracy_mean_std.png")


# ---------------------------------------------------------------------------
# extension plots (only when the extension's results are complete)
# ---------------------------------------------------------------------------


def _check_complete(df: pd.DataFrame, expected_ids: list[str], seeds: list[int], label: str) -> None:
    if df.empty:
        raise PlotSkipped(f"{label}: no results file / no records yet")
    present = set(zip(df["experiment_id"], df["seed"]))
    expected = {(i, s) for i in expected_ids for s in seeds}
    missing = expected - present
    if missing:
        raise PlotSkipped(
            f"{label}: incomplete ({len(expected) - len(missing)}/{len(expected)} "
            f"(experiment, seed) pairs present) - run it to completion first"
        )


def plot_extension_noise_sweep(
    sweep: pd.DataFrame, out_dir: Path, source: str, *, seeds: list[int] | None = None
) -> Path:
    seeds = list(RANDOM_SEEDS) if seeds is None else seeds
    probs = list(NOISE_SWEEP_TWO_QUBIT_PROBABILITIES)
    ids = [f"NSWEEP-Q{q}D{d}-P{p:g}" for q, d in NOISE_SWEEP_ARCHITECTURES for p in probs]
    sweep = _completed(sweep)
    _check_complete(sweep, ids, seeds, "extension noise sweep")
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    panels = [("accuracy", "Test accuracy"), ("degradation_pp", "Degradation (pp; negative = better)"),
              ("distribution_total_variation_distance", "TVD vs ideal")]
    for ax, (col, ylabel) in zip(axes, panels):
        for q, d in NOISE_SWEEP_ARCHITECTURES:
            sub = sweep[sweep.architecture == f"{q}q_depth{d}"]
            g = sub.groupby("two_qubit_probability")[col]
            means, stds = g.mean().reindex(probs), g.std(ddof=0).reindex(probs)
            ax.errorbar(range(len(probs)), means, yerr=stds, color=ARCH_COLORS[q], marker=ARCH_MARKERS[q],
                        capsize=3, lw=1.6, ms=6, label=f"{q} qubits, depth {d}", zorder=3)
        ax.set_xticks(range(len(probs)), [f"{p:g}" for p in probs])
        ax.set_xlabel("Two-qubit depolarizing probability (1-qubit = 1/10 of this)")
        ax.set_ylabel(ylabel)
        if col == "degradation_pp":
            ax.axhline(0, color=NEUTRAL, lw=1, ls="--")
    axes[0].legend(loc="lower left")
    fig.suptitle("EXTENSION - depolarizing noise sweep (evaluation only; weights trained ideally)", fontsize=12)
    _footnote(fig, source, _n_seeds(sweep),
              "Noise probabilities are this extension's own chosen defaults, not assignment-specified "
              "values (unlike Section 6's low/medium profiles).")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, out_dir / "extension_noise_sweep.png")


def plot_extension_breast_cancer(
    bc: pd.DataFrame, bc_classical: pd.DataFrame, out_dir: Path, source: str, *, seeds: list[int] | None = None
) -> Path:
    seeds = list(RANDOM_SEEDS) if seeds is None else seeds
    ids = [f"BC-{i}" for i in _META]
    bc = _completed(bc)
    _check_complete(bc, ids, seeds, "extension Breast Cancer (VQC)")
    if bc_classical.empty or len(bc_classical) < 2 * len(seeds):
        raise PlotSkipped("extension Breast Cancer: classical baselines incomplete")
    _style()
    st = _stats(bc, "accuracy")
    cl = _classical_means(bc_classical)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for x, i in enumerate(ids):
        q, d, noise = _META[i[3:]]
        ax.errorbar(x, st.loc[i, "mean"], yerr=st.loc[i, "std"], color=NOISE_COLORS[noise], marker=ARCH_MARKERS[q],
                    ms=7, capsize=4, lw=1.6, zorder=3)
        _seed_dots(ax, x, bc.loc[bc.experiment_id == i, "accuracy"], NOISE_COLORS[noise], ARCH_MARKERS[q])
        ax.annotate(f"{st.loc[i, 'mean']:.2f}", (x, st.loc[i, "mean"]), textcoords="offset points",
                    xytext=(10, -4), fontsize=8, color=INK2)
    # Classical models' means can be only a few hundredths apart (LogReg vs.
    # SVM here), which on this axis range puts their labels a few pixels
    # apart -- a fixed offset makes them overlap into illegible text. Stagger
    # each label's vertical offset by its row index so they never collide,
    # regardless of how close the underlying means are.
    for row, (model, (mean, _, _)) in enumerate(cl.items()):
        ax.axhline(mean, color=NEUTRAL, lw=1, ls=":")
        ax.annotate(f"{model.replace('_', ' ')} ({mean:.2f})", (len(ids) - 0.5, mean),
                    xytext=(0, 3 + row * 20), textcoords="offset points",
                    ha="right", fontsize=7.5, color=INK2, annotation_clip=False)
    ax.set_xticks(range(len(ids)), [f"{i}\n{_META[i[3:]][0]}q d{_META[i[3:]][1]}\n{_META[i[3:]][2]}" for i in ids])
    # The classical labels anchor at x = len(ids) - 0.5 = 7.5, past the last
    # data point (x=7) -- autoscale's default margin stops at ~7.43, so that
    # anchor sits just OUTSIDE the axes and matplotlib drops the whole
    # annotation (not just overlaps it: it never renders at all). Extend the
    # right edge explicitly so the anchor -- and the text extending left from
    # it -- are fully inside the visible area.
    ax.set_xlim(-0.6, len(ids) - 0.1)
    # Without explicit headroom, autoscale caps the axis right at the highest
    # classical mean, clipping its label (and the second, stacked label
    # above it) off the top of the chart entirely -- not just overlapping,
    # invisible. Leave room above the highest classical mean for both
    # staggered labels.
    highest_classical_mean = max(mean for mean, _, _ in cl.values())
    ax.set_ylim(top=highest_classical_mean + 0.09)
    ax.set_ylabel("Test accuracy (full held-out test set)")
    ax.set_title("Accuracy per experiment")
    noisy = [i for i in ids if _META[i[3:]][2] != "ideal"]
    dg = _stats(bc, "degradation_pp")
    for x, i in enumerate(noisy):
        noise = _META[i[3:]][2]
        ax2.bar(x, dg.loc[i, "mean"], 0.6, yerr=dg.loc[i, "std"], capsize=3, color=NOISE_COLORS[noise],
                hatch={"low": "//", "medium": "xx"}[noise], edgecolor=SURFACE, error_kw={"ecolor": INK2, "lw": 1}, zorder=3)
        ax2.annotate(f"{dg.loc[i, 'mean']:+.2f}", (x, dg.loc[i, "mean"]), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=8)
    ax2.axhline(0, color=NEUTRAL, lw=1, ls="--")
    ax2.set_xticks(range(len(noisy)), [f"{i}\n{_META[i[3:]][2]}" for i in noisy])
    ax2.set_ylabel("Degradation (pp; negative = better)")
    ax2.set_title("Noise degradation vs same-seed ideal")
    sub = int(bc["train_subsample_size"].iloc[0]) if "train_subsample_size" in bc.columns else -1
    fig.suptitle(f"EXTENSION - Breast Cancer Wisconsin (PCA to qubit count; {sub}-sample stratified training subset)",
                 fontsize=12)
    _footnote(fig, source, _n_seeds(bc), "Classical baselines use the identical subsampled training split.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return _save(fig, out_dir / "extension_breast_cancer.png")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def generate_all_plots(
    out_dir: Path = PLOTS_DIR,
    *,
    vqc_results_file: Path = VQC_RESULTS_FILE,
    classical_results_file: Path = CLASSICAL_RESULTS_FILE,
    noise_sweep_file: Path = NOISE_SWEEP_FILE,
    bc_results_file: Path = BC_RESULTS_FILE,
    bc_classical_file: Path = BC_CLASSICAL_FILE,
    an_vqc_results_file: Path = AN_VQC_RESULTS_FILE,
    an_vqc_classical_file: Path = AN_VQC_CLASSICAL_FILE,
    noise_sweep_seeds: list[int] | None = None,
    breast_cancer_seeds: list[int] | None = None,
) -> PlotReport:
    """Draw every plot the saved results support; report what was skipped and why.

    ``noise_sweep_seeds``/``breast_cancer_seeds`` are the seeds each
    extension's completeness check requires (default: all of
    ``config.RANDOM_SEEDS``, i.e. 5) -- set independently because the two
    extensions can finish with different seed counts (e.g. a 3-seed noise
    sweep alongside a 5-seed Breast Cancer run). This only affects the
    completeness check; the plot itself always aggregates every seed
    actually present in the result file, and the footnote's ``n = ...``
    reports that real count, not this parameter.
    """
    out_dir = Path(out_dir)
    main = _completed(_frame(vqc_results_file))
    classical = _frame(classical_results_file)
    report = PlotReport()

    jobs = {
        "circuit_diagram": lambda: plot_circuit_diagram(out_dir),
        "01_degradation_vs_noise": lambda: plot_degradation_vs_noise(main, out_dir, vqc_results_file.name),
        "02_accuracy_vs_depth": lambda: plot_accuracy_vs_depth(main, out_dir, vqc_results_file.name, classical),
        "03_noise_comparison": lambda: plot_noise_comparison(main, out_dir, vqc_results_file.name),
        "04_qubits_vs_performance": lambda: plot_qubits_vs_performance(main, out_dir, vqc_results_file.name),
        "05_output_distributions": lambda: plot_output_distributions(main, out_dir, vqc_results_file.name),
        "06_classical_vs_qml": lambda: plot_classical_vs_qml(main, classical, out_dir, vqc_results_file.name),
        "07_standard_vs_adaptive": lambda: plot_standard_vs_adaptive(
            _frame(an_vqc_results_file), _frame(an_vqc_classical_file), out_dir, an_vqc_results_file.name),
        "08_accuracy_mean_std": lambda: plot_accuracy_mean_std(main, classical, out_dir, vqc_results_file.name),
        "extension_noise_sweep": lambda: plot_extension_noise_sweep(
            _frame(noise_sweep_file), out_dir, noise_sweep_file.name, seeds=noise_sweep_seeds),
        "extension_breast_cancer": lambda: plot_extension_breast_cancer(
            _frame(bc_results_file), _frame(bc_classical_file), out_dir, bc_results_file.name, seeds=breast_cancer_seeds),
    }
    for name, job in jobs.items():
        try:
            report.generated.append(job())
        except PlotSkipped as exc:
            report.skipped[name] = str(exc)
            logger.warning("Skipped %s: %s", name, exc)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate all plots from saved results.")
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument(
        "--noise-sweep-seeds", type=str, default=None,
        help="Comma-separated seeds the noise-sweep completeness check requires, e.g. '0,1,2'. "
        "Default: all of config.RANDOM_SEEDS.",
    )
    parser.add_argument(
        "--breast-cancer-seeds", type=str, default=None,
        help="Comma-separated seeds the Breast Cancer completeness check requires, e.g. '0,1,2,3,4'. "
        "Default: all of config.RANDOM_SEEDS.",
    )
    args = parser.parse_args(argv)
    noise_sweep_seeds = [int(s) for s in args.noise_sweep_seeds.split(",")] if args.noise_sweep_seeds else None
    breast_cancer_seeds = [int(s) for s in args.breast_cancer_seeds.split(",")] if args.breast_cancer_seeds else None
    report = generate_all_plots(
        args.out_dir, noise_sweep_seeds=noise_sweep_seeds, breast_cancer_seeds=breast_cancer_seeds
    )
    print(f"\nGenerated {len(report.generated)} plot(s):")
    for p in report.generated:
        print(f"  {p}")
    print(f"Skipped {len(report.skipped)}:")
    for name, why in report.skipped.items():
        print(f"  {name}: {why}")


if __name__ == "__main__":
    main()
