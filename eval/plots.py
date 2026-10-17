import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, fixed order


def _style(ax, title: str, xlabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8)


def grouped_hbar(path, labels, groups, title, xlabel, value_fmt="{:.0%}", xlim=None, errors=None):
    """Horizontal bars, one row per label, one bar per group. groups: {group name: [values]}.
    errors: optional {group name: [(low, high), ...]} drawn as whiskers."""
    n_groups = len(groups)
    height = 0.8 / n_groups
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.55 * len(labels) * max(1, n_groups * 0.7)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for g, (name, values) in enumerate(groups.items()):
        ys = [i + (g - (n_groups - 1) / 2) * height for i in range(len(labels))]
        ax.barh(ys, values, height=height * 0.85, color=SERIES[g], label=name)  # 0.85: surface gap between bars
        whiskers = errors.get(name) if errors else None
        if whiskers:
            for y, (lo, hi) in zip(ys, whiskers):
                ax.plot([lo, hi], [y, y], color=INK_2, linewidth=1)
        for i, (y, v) in enumerate(zip(ys, values)):
            x = max(v, whiskers[i][1]) if whiskers else v  # put the label past the whisker, not on it
            ax.annotate(value_fmt.format(v), (x, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    _style(ax, title, xlabel)
    if n_groups > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=n_groups, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stacked_hbar(path, labels, parts, title, xlabel):
    """Horizontal stacked bars. parts: {component name: [values]} (e.g. routing / tools / llm ms)."""
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.5 * len(labels)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    left = [0.0] * len(labels)
    for k, (name, values) in enumerate(parts.items()):
        ax.barh(range(len(labels)), values, left=left, height=0.6, color=SERIES[k], label=name,
                edgecolor=SURFACE, linewidth=2)  # 2px surface gap between segments
        left = [l + v for l, v in zip(left, values)]
    for i, total in enumerate(left):
        ax.annotate(f"{total:.0f} ms", (total, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    _style(ax, title, xlabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(parts), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability(path, confidences, correct, title, bins=5):
    """Calibration plot: for each confidence bin, how often was the router actually right?
    Points on the diagonal = honest confidence."""
    edges = [i / bins for i in range(bins + 1)]
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (lo == 0 and c == 0)]
        if idx:
            xs.append(sum(confidences[i] for i in idx) / len(idx))
            ys.append(sum(correct[i] for i in idx) / len(idx))
            ns.append(len(idx))
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="perfectly honest")
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label="router")  # surface ring on markers
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), xytext=(6, -10), textcoords="offset points", fontsize=7, color=INK_2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share actually correct", color=INK_2, fontsize=9)
    _style(ax, title, "calibrated confidence")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
