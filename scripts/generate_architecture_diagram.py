"""Render the system architecture diagram referenced from the README.

    python -m scripts.generate_architecture_diagram
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOXES = [
    ("Client", 0.05, 0.5, "#cbd5e1"),
    ("FastAPI\n/predict /health\n/metrics /drift /retrain", 0.28, 0.5, "#93c5fd"),
    ("Feature\nPipeline", 0.55, 0.78, "#a7f3d0"),
    ("RandomForest\nEnsemble Model", 0.78, 0.78, "#a7f3d0"),
    ("SQLAlchemy\n(SQLite / Postgres)", 0.55, 0.5, "#fde68a"),
    ("Prediction Logs\n+ Experiment Runs", 0.78, 0.5, "#fde68a"),
    ("PSI Drift\nTracker", 0.55, 0.2, "#fca5a5"),
]

ARROWS = [
    (0, 1),  # client -> api
    (1, 2),  # api -> feature pipeline
    (2, 3),  # feature pipeline -> model
    (1, 4),  # api -> db
    (4, 5),  # db -> logs/runs
    (5, 6),  # logs -> drift tracker
    (6, 1),  # drift tracker feedback -> api /drift
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)
    ax.axis("off")
    ax.set_title("Learner Attrition Radar — Architecture", fontsize=14, fontweight="bold")

    positions = {}
    for i, (label, x, y, color) in enumerate(BOXES):
        w, h = 0.18, 0.16
        box = FancyBboxPatch(
            (x, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=9)
        positions[i] = (x, y, w, h)

    for src, dst in ARROWS:
        x1, y1, w1, h1 = positions[src]
        x2, y2, w2, h2 = positions[dst]
        start = (x1 + w1, y1) if x2 > x1 else (x1, y1 - h1 / 2)
        end = (x2, y2) if x2 > x1 else (x2 + w2 / 2, y2 + h2 / 2)
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12, color="#334155", linewidth=1
        )
        ax.add_patch(arrow)

    fig.tight_layout()
    fig.savefig("screenshots/architecture.png", dpi=150)


if __name__ == "__main__":
    main()
