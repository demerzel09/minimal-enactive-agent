"""Visualization utilities for the minimal enactive PoC."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(log: Dict, layout: Dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(log["x"])
    y = np.asarray(log["y"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, lw=1.5, alpha=0.9, label="trajectory")
    ax.scatter([x[0]], [y[0]], c="green", s=60, label="start")
    ax.scatter([x[-1]], [y[-1]], c="red", s=60, label="end")

    # Draw food patches (supports single and multi-patch layouts)
    centers = layout.get("patch_centers", [layout["patch_center"].tolist()] if "patch_center" in layout else [])
    radii = layout.get("patch_radii", [layout.get("patch_radius", 2.8)])
    colors = ["orange", "green", "cyan", "yellow"]
    for i, (c, r) in enumerate(zip(centers, radii)):
        label = f"patch {i}" if len(centers) > 1 else "food patch"
        circle = plt.Circle(c, r, color=colors[i % len(colors)], alpha=0.25, label=label)
        ax.add_patch(circle)

    risk = plt.Circle(layout["risk_center"], layout["risk_radius"], color="purple", alpha=0.2, label="risk core")
    ax.add_patch(risk)

    ax.set_xlim(0, layout["world_size"])
    ax.set_ylim(0, layout["world_size"])
    ax.set_aspect("equal")
    ax.set_title("Agent trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_states(log: Dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = np.arange(len(log["h"]))
    h = np.asarray(log["h"])  # [T, h_dim]
    m = np.asarray(log["m"])  # [T, m_dim]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for i in range(h.shape[1]):
        axes[0].plot(t, h[:, i], lw=1.5, label=f"h[{i}]")
    axes[0].set_ylabel("h")
    axes[0].set_title("Internal state and mode dynamics")
    axes[0].legend(fontsize=8)

    for i in range(m.shape[1]):
        axes[1].plot(t, m[:, i], lw=1.5, label=f"m[{i}]")
    axes[1].set_ylabel("m")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, log["local_food"], label="local_food", lw=1.3)
    axes[2].plot(t, log["local_risk"], label="local_risk", lw=1.3)
    axes[2].plot(t, log["in_patch"], label="in_patch", lw=1.3)
    axes[2].set_ylabel("obs/context")
    axes[2].set_xlabel("time step")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
