import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Graph LSE color-descent energy landscape on PCA plane
# ============================================================

def sbm_adjacency(n, sizes, p_in, p_out, seed=None):
    rng = np.random.default_rng(seed)
    z = np.concatenate([[k] * sizes[k] for k in range(len(sizes))])
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if z[i] == z[j] else p_out
            A[i, j] = A[j, i] = rng.random() < p
    return A

def laplacian(A):
    return np.diag(A.sum(axis=1)) - A

def op_norm_symmetric(M):
    return np.max(np.abs(np.linalg.eigvalsh(M)))

def graph_lse_energy(Lq, stored_Ls, beta):
    d2 = np.array([op_norm_symmetric(Lq - Li) ** 2 for Li in stored_Ls])
    a = -beta * d2
    m = np.max(a)
    return -(m + np.log(np.sum(np.exp(a - m)))) / beta


# 1. Well-separated sampled SBM memories

n = 60
sizes = [30, 30]

stored_params = [
    (0.92, 0.02),
    (0.75, 0.06),
    (0.58, 0.16),
    (0.38, 0.30),
    (0.92, 0.30),
]

stored_Ls = []
for s, (p_in, p_out) in enumerate(stored_params):
    A = sbm_adjacency(n, sizes, p_in, p_out, seed=500 + 17 * s)
    stored_Ls.append(laplacian(A))

# 2. PCA plane of sampled Laplacians


X = np.array([L.reshape(-1) for L in stored_Ls])
xbar = X.mean(axis=0)
Xc = X - xbar

_, _, Vt = np.linalg.svd(Xc, full_matrices=False)

v1 = Vt[0]
v2 = Vt[1]

coords = Xc @ np.vstack([v1, v2]).T

Lbar = xbar.reshape(n, n)
V1 = v1.reshape(n, n)
V2 = v2.reshape(n, n)

V1 = 0.5 * (V1 + V1.T)
V2 = 0.5 * (V2 + V2.T)


# 3. Grid on PCA plane


margin = 0.65

x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

xr = x_max - x_min
yr = y_max - y_min

x_grid = np.linspace(x_min - margin * xr, x_max + margin * xr, 150)
y_grid = np.linspace(y_min - margin * yr, y_max + margin * yr, 150)

betas = [0.001, 0.01, 0.1,0.5, 1, 10, 100]


# 4. Plot color-descent landscapes


fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

for ax, beta in zip(axes.ravel(), betas):

    E = np.zeros((len(y_grid), len(x_grid)))

    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            Lq = Lbar + x * V1 + y * V2
            E[i, j] = graph_lse_energy(Lq, stored_Ls, beta)

    # Optional: normalize per panel for prettier color descent
    Eplot = E
    # Eplot = (E - E.min()) / (E.max() - E.min())

    im = ax.imshow(
        Eplot,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        aspect="auto",
        cmap="turbo",      # try also: "viridis", "plasma", "inferno", "cividis"
        interpolation="bilinear"
    )

    ax.contour(
        x_grid,
        y_grid,
        E,
        levels=18,
        colors="black",
        linewidths=0.45,
        alpha=0.45
    )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        marker="*",
        s=220,
        c="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=5
    )

    for k, (x, y) in enumerate(coords):
        ax.text(
            x, y, f" {k+1}",
            fontsize=11,
            weight="bold",
            color="black",
            zorder=6
        )

    ax.set_title(rf"$\beta={beta}$")
    ax.set_xlabel("PC1 direction")
    ax.set_ylabel("PC2 direction")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$E(q)$")

plt.suptitle(
    r"Graph LSE Energy Landscape on PCA Slice of Sampled SBM Laplacians",
    fontsize=16
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Publication settings
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 17,

    # Make text/lines sharper and more publication-like
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,

    # Embed fonts properly in vector output
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Better math rendering
    "mathtext.fontset": "stix",
    "font.family": "serif",
})


# ============================================================
# Graph LSE color-descent energy landscape on PCA plane
# ============================================================

def sbm_adjacency(n, sizes, p_in, p_out, seed=None):
    rng = np.random.default_rng(seed)
    z = np.concatenate([[k] * sizes[k] for k in range(len(sizes))])
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if z[i] == z[j] else p_out
            A[i, j] = A[j, i] = rng.random() < p
    return A


def laplacian(A):
    return np.diag(A.sum(axis=1)) - A


def op_norm_symmetric(M):
    return np.max(np.abs(np.linalg.eigvalsh(M)))


def graph_lse_energy(Lq, stored_Ls, beta):
    d2 = np.array([
        op_norm_symmetric(Lq - Li) ** 2
        for Li in stored_Ls
    ])

    a = -beta * d2
    m = np.max(a)

    return -(m + np.log(np.sum(np.exp(a - m)))) / beta


# ============================================================
# 1. Well-separated sampled SBM memories
# ============================================================

n = 60
sizes = [30, 30]

stored_params = [
    (0.92, 0.02),
    (0.75, 0.06),
    (0.58, 0.16),
    (0.38, 0.30),
    (0.92, 0.30),
]

stored_Ls = []

for s, (p_in, p_out) in enumerate(stored_params):
    A = sbm_adjacency(
        n,
        sizes,
        p_in,
        p_out,
        seed=500 + 17 * s
    )
    stored_Ls.append(laplacian(A))


# ============================================================
# 2. PCA plane of sampled Laplacians
# ============================================================

X = np.array([L.reshape(-1) for L in stored_Ls])

xbar = X.mean(axis=0)
Xc = X - xbar

_, _, Vt = np.linalg.svd(Xc, full_matrices=False)

v1 = Vt[0]
v2 = Vt[1]

coords = Xc @ np.vstack([v1, v2]).T

Lbar = xbar.reshape(n, n)

V1 = v1.reshape(n, n)
V2 = v2.reshape(n, n)

V1 = 0.5 * (V1 + V1.T)
V2 = 0.5 * (V2 + V2.T)


# ============================================================
# 3. Grid on PCA plane
# ============================================================

margin = 0.65

x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

xr = x_max - x_min
yr = y_max - y_min

x_grid = np.linspace(
    x_min - margin * xr,
    x_max + margin * xr,
    150
)

y_grid = np.linspace(
    y_min - margin * yr,
    y_max + margin * yr,
    150
)

# Six panels for a 2 x 3 figure
betas = [0.001, 0.01, 0.1, 0.5, 1, 10]


# ============================================================
# 4. Plot publication-quality landscapes
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 8.3),
    constrained_layout=True
)

for ax, beta in zip(axes.ravel(), betas):

    E = np.zeros((len(y_grid), len(x_grid)))

    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            Lq = Lbar + x * V1 + y * V2
            E[i, j] = graph_lse_energy(
                Lq,
                stored_Ls,
                beta
            )

    im = ax.imshow(
        E,
        origin="lower",
        extent=[
            x_grid.min(),
            x_grid.max(),
            y_grid.min(),
            y_grid.max()
        ],
        aspect="auto",
        cmap="turbo",

        # For publication output, avoid relying on
        # raster interpolation to fake smoothness.
        interpolation="bilinear",

        # Ensure high-quality rasterization inside PDF
        rasterized=True
    )

    ax.contour(
        x_grid,
        y_grid,
        E,
        levels=18,
        colors="black",
        linewidths=0.55,
        alpha=0.50
    )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        marker="*",
        s=190,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=5
    )

    for k, (x, y) in enumerate(coords):
        ax.annotate(
            rf"${k+1}$",
            xy=(x, y),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=6
        )

    ax.set_title(
        rf"$\beta = {beta}$",
        pad=6
    )

    ax.set_xlabel("PC1 direction")
    ax.set_ylabel("PC2 direction")

    ax.tick_params(
        axis="both",
        which="major",
        width=0.9,
        length=4
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.035
    )

    cbar.set_label(
        r"$E(q)$",
        rotation=90,
        labelpad=8
    )

    cbar.ax.tick_params(
        labelsize=9,
        width=0.8,
        length=3
    )


fig.suptitle(
    "Graph LSE Energy Landscape on PCA Slice of Sampled SBM Laplacians",
    fontsize=17,
    fontweight="semibold"
)


# ============================================================
# 5. SAVE — publication quality
# ============================================================

# High-resolution PNG
fig.savefig(
    "graph_lse_energy_landscape.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white"
)

# Strongly recommended for ICLR / LaTeX:
# text and contour lines remain vector-sharp.
fig.savefig(
    "graph_lse_energy_landscape.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white"
)

# Optional SVG for editing in Illustrator/Inkscape
fig.savefig(
    "graph_lse_energy_landscape.svg",
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white"
)

plt.show()
