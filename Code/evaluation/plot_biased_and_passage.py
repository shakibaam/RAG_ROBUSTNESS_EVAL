#!/usr/bin/env python3
"""
Generate Biased_Pool and Passage_Based stance-alignment plots for TREC 2020/2021.

Reads Experiment_Results and writes:
  Plots/{gemini|gpt4omini}/Biased_Pool/biased_pool_<Model>_2020_2021.png
  Plots/{gemini|gpt4omini}/Passage_Based/passage_based_<Model>_2020_2021.png
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "Experiment_Results"
PLOTS = REPO / "Plots"

MODELS = [
    "GPT_4.1",
    "GPT_5",
    "Phi_4",
    "Llama-3.3-70B-Instruct",
    "Qwen3-30B-A3B-Instruct-2507",
]

PROMPT_TYPES = ["consistent", "neutral", "inconsistent"]
PROMPT_TITLES = ["Consistent Query", "Neutral Query", "Inconsistent Query"]

ATTACK_ORDER = [
    "rewriter_attack",
    "paraphraser_attack",
    "fact_inversion_attack",
    "fsap_interq",
    "fsap_intraq",
    "liar_attack",
]
ATTACK_LABELS = {
    "rewriter_attack": "Rewriter",
    "paraphraser_attack": "Paraphraser",
    "fact_inversion_attack": "Fact Inversion",
    "liar_attack": "Liar",
    "fsap_interq": "FSAP-InterQ",
    "fsap_intraq": "FSAP-IntraQ",
}

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    }
)


def normalize_attack(name: str) -> str:
    s = str(name)
    if s.startswith("fsap_interq"):
        return "fsap_interq"
    if s.startswith("fsap_intraq"):
        return "fsap_intraq"
    return s


def stance_col(df: pd.DataFrame, use_gemini: bool) -> str | None:
    if use_gemini and "predicted_stance_gemini" in df.columns:
        return "predicted_stance_gemini"
    if "predicted_stance" in df.columns:
        return "predicted_stance"
    if "predicted_stance_gemini" in df.columns:
        return "predicted_stance_gemini"
    return None


def find_judge_dir(model_root: Path, prefer_gemini: bool) -> Path | None:
    if not model_root.exists():
        return None
    names = sorted(p.name for p in model_root.iterdir() if p.is_dir())
    if prefer_gemini:
        for n in ("gemini2.5flash", "gemini2.0flash"):
            if n in names:
                return model_root / n
        for n in names:
            if n.startswith("gemini"):
                return model_root / n
    else:
        if "gpt4omini" in names:
            return model_root / "gpt4omini"
    return None


def bootstrap_ci(k: int, n: int, n_boot: int = 500, seed: int = 42):
    if n <= 0:
        return 0.0, 0.0, 0.0
    pct = 100.0 * k / n
    rng = np.random.default_rng(seed)
    data = np.concatenate([np.ones(k), np.zeros(n - k)])
    props = rng.choice(data, size=(n_boot, n), replace=True).mean(axis=1) * 100.0
    lo, hi = np.percentile(props, [2.5, 97.5])
    return float(pct), float(lo), float(hi)


def summarize_biased(judge_root: Path, tone: str, bias: str, use_gemini: bool):
    root = judge_root / f"{tone}_results" / bias
    counts = defaultdict(lambda: {"aligned": 0, "total": 0})
    if not root.exists():
        return {}
    for csv in root.rglob("*.csv"):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        col = stance_col(df, use_gemini)
        if col is None or "gt_stance" not in df.columns:
            continue
        if "attack_type" not in df.columns:
            continue
        sub = df
        if "tone" in df.columns:
            sub = df[df["tone"] == tone]
        for atk, g in sub.groupby("attack_type", dropna=True):
            a = normalize_attack(atk)
            counts[a]["aligned"] += int((g[col] == g["gt_stance"]).sum())
            counts[a]["total"] += len(g)
    out = {}
    for a, v in counts.items():
        out[a] = bootstrap_ci(v["aligned"], v["total"], seed=hash(a) % 10000)
    return out


def pair_setting_from_name(path: Path, tone: str) -> str | None:
    """Infer pair_setting from CSV stem when the column is missing."""
    stem = path.stem
    # strip tone_ prefix and trailing generator tags
    name = stem
    for pref in (f"{tone}_", "consistent_", "neutral_", "inconsistent_"):
        if name.startswith(pref):
            name = name[len(pref) :]
            break
    # drop __Model__ragnarok... suffixes
    if "__" in name:
        name = name.split("__", 1)[0]
    if name.endswith("_ragnarok_results_format"):
        name = name[: -len("_ragnarok_results_format")]
    if name.endswith("_ragnarok_format_results"):
        name = name[: -len("_ragnarok_format_results")]
    if "-" not in name:
        return None
    return name


def summarize_passage(judge_root: Path, tone: str, position: str, use_gemini: bool):
    """position: 'first' (helpful-*) or 'second' (*-helpful)."""
    root = judge_root / f"{tone}_results"
    counts = defaultdict(lambda: {"aligned": 0, "total": 0})
    if not root.exists():
        return {}
    csvs = list(root.rglob("*.csv"))
    # Prefer GPT-4o adversary-source files when that subset exists (matches paper plots)
    gpt4o = [c for c in csvs if "__GPT-4o__" in c.name]
    if gpt4o:
        csvs = gpt4o
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        col = stance_col(df, use_gemini)
        if col is None or "gt_stance" not in df.columns:
            continue
        sub = df
        if "tone" in df.columns:
            sub = df[df["tone"] == tone]
        if sub.empty:
            continue

        pairs: list[str]
        if "pair_setting" in sub.columns:
            pairs = [str(p) for p in sub["pair_setting"].dropna().unique()]
        else:
            inferred = pair_setting_from_name(csv, tone)
            pairs = [inferred] if inferred else []

        for pair in pairs:
            if "-" not in pair:
                continue
            left, right = pair.rsplit("-", 1)
            if left == "helpful" and right == "helpful":
                continue
            if position == "first" and left == "helpful" and right != "helpful":
                atk = normalize_attack(right)
            elif position == "second" and right == "helpful" and left != "helpful":
                atk = normalize_attack(left)
            else:
                continue
            g = sub[sub["pair_setting"] == pair] if "pair_setting" in sub.columns else sub
            counts[atk]["aligned"] += int((g[col] == g["gt_stance"]).sum())
            counts[atk]["total"] += len(g)
    out = {}
    for a, v in counts.items():
        out[a] = bootstrap_ci(v["aligned"], v["total"], seed=hash(a + position) % 10000)
    return out


def style_ax(ax):
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")


def plot_biased(model: str, judge_key: str, use_gemini: bool) -> Path | None:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=True)
    any_data = False
    for row, year in enumerate(("TREC2021", "TREC2020")):
        model_root = RESULTS / year / f"Biased_Pool_{model}"
        judge_root = find_judge_dir(model_root, prefer_gemini=use_gemini)
        if judge_root is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center")
                axes[row, col].set_title(f"{year[-4:]} – {PROMPT_TITLES[col]}")
            continue
        for col, tone in enumerate(PROMPT_TYPES):
            ax = axes[row, col]
            help_s = summarize_biased(judge_root, tone, "helpful_biased", use_gemini)
            harm_s = summarize_biased(judge_root, tone, "harmful_biased", use_gemini)
            atks = [a for a in ATTACK_ORDER if a in help_s or a in harm_s]
            if not atks:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                any_data = True
                x = np.arange(len(atks))
                w = 0.35
                mh = [help_s.get(a, (0, 0, 0))[0] for a in atks]
                mb = [harm_s.get(a, (0, 0, 0))[0] for a in atks]
                yeh = np.array(
                    [
                        [max(0, help_s.get(a, (0, 0, 0))[0] - help_s.get(a, (0, 0, 0))[1]) for a in atks],
                        [max(0, help_s.get(a, (0, 0, 0))[2] - help_s.get(a, (0, 0, 0))[0]) for a in atks],
                    ]
                )
                yeb = np.array(
                    [
                        [max(0, harm_s.get(a, (0, 0, 0))[0] - harm_s.get(a, (0, 0, 0))[1]) for a in atks],
                        [max(0, harm_s.get(a, (0, 0, 0))[2] - harm_s.get(a, (0, 0, 0))[0]) for a in atks],
                    ]
                )
                ax.bar(
                    x - w / 2,
                    mh,
                    w,
                    yerr=yeh,
                    capsize=3,
                    color="#98FB98",
                    hatch="//",
                    edgecolor="black",
                    ecolor="black",
                )
                ax.bar(
                    x + w / 2,
                    mb,
                    w,
                    yerr=yeb,
                    capsize=3,
                    color="#FF7F7F",
                    hatch="xx",
                    edgecolor="black",
                    ecolor="black",
                )
                ax.set_xticks(x)
                ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in atks], rotation=45, ha="right")
            ax.set_title(f"{year[-4:]} – {PROMPT_TITLES[col]}", fontweight="bold")
            if col == 0:
                ax.set_ylabel("Stance Alignment (%)", fontweight="bold")
            style_ax(ax)

    if not any_data:
        plt.close(fig)
        return None

    legend = [
        Patch(facecolor="#98FB98", hatch="//", edgecolor="black", label="Biased to Helpful"),
        Patch(facecolor="#FF7F7F", hatch="xx", edgecolor="black", label="Biased to Harmful"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=True)
    fig.suptitle(f"Biased Pool — {model} ({judge_key})", fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.02, 0.06, 1, 0.95])
    out_dir = PLOTS / judge_key / "Biased_Pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"biased_pool_{model}_2020_2021.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_passage(model: str, judge_key: str, use_gemini: bool) -> Path | None:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=True)
    any_data = False
    for row, year in enumerate(("TREC2021", "TREC2020")):
        model_root = RESULTS / year / f"Passage_Based_{model}"
        judge_root = find_judge_dir(model_root, prefer_gemini=use_gemini)
        if judge_root is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center")
                axes[row, col].set_title(f"{year[-4:]} – {PROMPT_TITLES[col]}")
            continue
        for col, tone in enumerate(PROMPT_TYPES):
            ax = axes[row, col]
            first = summarize_passage(judge_root, tone, "first", use_gemini)
            second = summarize_passage(judge_root, tone, "second", use_gemini)
            atks = [a for a in ATTACK_ORDER if a in first or a in second]
            if not atks:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                any_data = True
                x = np.arange(len(atks))
                w = 0.35
                mf = [first.get(a, (0, 0, 0))[0] for a in atks]
                ms = [second.get(a, (0, 0, 0))[0] for a in atks]
                yef = np.array(
                    [
                        [max(0, first.get(a, (0, 0, 0))[0] - first.get(a, (0, 0, 0))[1]) for a in atks],
                        [max(0, first.get(a, (0, 0, 0))[2] - first.get(a, (0, 0, 0))[0]) for a in atks],
                    ]
                )
                yes = np.array(
                    [
                        [max(0, second.get(a, (0, 0, 0))[0] - second.get(a, (0, 0, 0))[1]) for a in atks],
                        [max(0, second.get(a, (0, 0, 0))[2] - second.get(a, (0, 0, 0))[0]) for a in atks],
                    ]
                )
                ax.bar(
                    x - w / 2,
                    mf,
                    w,
                    yerr=yef,
                    capsize=3,
                    color="#FFD966",
                    hatch="//",
                    edgecolor="black",
                    ecolor="black",
                )
                ax.bar(
                    x + w / 2,
                    ms,
                    w,
                    yerr=yes,
                    capsize=3,
                    color="#9B89FF",
                    hatch="xx",
                    edgecolor="black",
                    ecolor="black",
                )
                ax.set_xticks(x)
                ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in atks], rotation=45, ha="right")
            ax.set_title(f"{year[-4:]} – {PROMPT_TITLES[col]}", fontweight="bold")
            if col == 0:
                ax.set_ylabel("Stance Alignment (%)", fontweight="bold")
            style_ax(ax)

    if not any_data:
        plt.close(fig)
        return None

    legend = [
        Patch(facecolor="#FFD966", hatch="//", edgecolor="black", label="Helpful First"),
        Patch(facecolor="#9B89FF", hatch="xx", edgecolor="black", label="Helpful Second"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=True)
    fig.suptitle(f"Passage-Based — {model} ({judge_key})", fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.02, 0.06, 1, 0.95])
    out_dir = PLOTS / judge_key / "Passage_Based"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"passage_based_{model}_2020_2021.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    written = []
    for model in MODELS:
        for judge_key, use_gemini in (("gemini", True), ("gpt4omini", False)):
            p1 = plot_biased(model, judge_key, use_gemini)
            p2 = plot_passage(model, judge_key, use_gemini)
            for p in (p1, p2):
                if p:
                    written.append(str(p.relative_to(REPO)))
                    print("wrote", p.relative_to(REPO), flush=True)
                else:
                    print(f"skip {model} {judge_key}", flush=True)
    print(f"DONE n={len(written)}")


if __name__ == "__main__":
    main()
