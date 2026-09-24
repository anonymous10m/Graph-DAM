import os
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.9
})

OUTDIR = os.path.join("outputs", "sbm")

os.makedirs(
    OUTDIR,
    exist_ok=True
)



# SETTINGS


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

N_NODES = 120

K = 4

assert N_NODES % K == 0

COMM_SIZE = N_NODES // K


# Number of stored memories


N_MEM = 50


# Actual corruption:
# remove 20% of EXISTING graph edges


MASK_RATE = 0.20


# Independent experiment repetitions used to report SE


N_TRIALS = 5

# DAM iterations
N_STEPS = 5

# inverse temperatures
BETA_N = np.logspace(
    -1,
    3,
    25
)

EPS = 1e-12


print("=" * 94)
print("SBM ASSOCIATIVE MEMORY RETRIEVAL")
print("=" * 94)

print(
    f"N memories       : {N_MEM}"
)

print(
    f"Nodes            : {N_NODES}"
)

print(
    f"Communities      : {K}"
)

print(
    f"Edge masking     : {100*MASK_RATE:.0f}%"
)

print(
    f"Independent runs : {N_TRIALS}"
)


# 1. COMMON COMMUNITY ASSIGNMENT


z = np.repeat(
    np.arange(K),
    COMM_SIZE
)

# 2. AUTOMATICALLY GENERATE 50 SBM POPULATION MODELS

print(
    "\n[1/9] Constructing 50 SBM population models..."
)

rng_B = np.random.default_rng(
    SEED
)

BASE_DIAG = 0.35
BASE_OFF  = 0.070

DIAG_SD = 0.065
OFF_SD  = 0.035

DIAG_MIN = 0.21
DIAG_MAX = 0.49

OFF_MIN = 0.025
OFF_MAX = 0.140


MIN_B_SEPARATION = 0.052

OFF_PAIRS = [
    (0,1),
    (0,2),
    (0,3),
    (1,2),
    (1,3),
    (2,3)
]


def generate_B():

  
    # Within-community probabilities
    
    # zero-mean perturbation preserves average diagonal level
    

    diag_delta = rng_B.normal(
        0,
        DIAG_SD,
        K
    )

    diag_delta -= diag_delta.mean()

    diag = (
        BASE_DIAG
        +
        diag_delta
    )

    if (
        np.any(diag < DIAG_MIN)
        or
        np.any(diag > DIAG_MAX)
    ):
        return None
    # Between-community probabilities
    
    # zero-mean perturbation preserves mean off-block density


    off_delta = rng_B.normal(
        0,
        OFF_SD,
        len(OFF_PAIRS)
    )

    off_delta -= off_delta.mean()

    off = (
        BASE_OFF
        +
        off_delta
    )

    if (
        np.any(off < OFF_MIN)
        or
        np.any(off > OFF_MAX)
    ):
        return None

    B = np.zeros(
        (K,K),
        dtype=float
    )

    for k in range(K):

        B[k,k] = diag[k]

    for value, (a,b) in zip(
        off,
        OFF_PAIRS
    ):

        B[a,b] = value
        B[b,a] = value

    return B


B_LIST = []

attempts = 0

while len(B_LIST) < N_MEM:

    attempts += 1

    B = generate_B()

    if B is None:
        continue

    # Avoid virtually duplicate population memories

    if len(B_LIST) > 0:

        sep = min(
            np.linalg.norm(
                B - B_old,
                ord=2
            )
            for B_old in B_LIST
        )

        if sep < MIN_B_SEPARATION:
            continue

    B_LIST.append(B)


B_LIST = np.stack(
    B_LIST
)


print(
    f"Generated {N_MEM} models "
    f"after {attempts} proposals."
)



# POPULATION DENSITY DIAGNOSTIC


def population_probability(B):

    return B[
        z[:,None],
        z[None,:]
    ]


def expected_density(B):

    P = population_probability(B)

    mask = ~np.eye(
        N_NODES,
        dtype=bool
    )

    return P[
        mask
    ].mean()


POP_DENSITY = np.array([
    expected_density(B)
    for B in B_LIST
])


print(
    "Expected density:",
    f"{POP_DENSITY.mean():.4f}",
    "+/-",
    f"{POP_DENSITY.std():.4f}"
)

print(
    "Density range:",
    f"[{POP_DENSITY.min():.4f}, "
    f"{POP_DENSITY.max():.4f}]"
)



# 3. SBM SAMPLER


def sample_sbm(
    B,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    P = population_probability(
        B
    )

    U = rng.random(
        (
            N_NODES,
            N_NODES
        )
    )

    upper = np.triu(
        U < P,
        1
    )

    A = (
        upper
        +
        upper.T
    ).astype(
        np.float64
    )

    return A


def laplacian(A):

    d = A.sum(
        axis=1
    )

    return (
        np.diag(d)
        -
        A
    )


# 4. MASK EXACTLY 20% OF EXISTING EDGES


def mask_existing_edges(
    A,
    rate,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    Aq = A.copy()

    # unordered existing edges
    edges = np.argwhere(
        np.triu(
            A > 0,
            1
        )
    )

    n_edges = len(
        edges
    )

    n_mask = int(
        round(
            rate
            *
            n_edges
        )
    )

    if n_mask == 0:

        return Aq

    selected = rng.choice(
        n_edges,
        size=n_mask,
        replace=False
    )

    removed = edges[
        selected
    ]

    Aq[
        removed[:,0],
        removed[:,1]
    ] = 0.0

    Aq[
        removed[:,1],
        removed[:,0]
    ] = 0.0

    return Aq


# 5. BUILD INDEPENDENT RUNS


print(
    "\n[2/9] Sampling independent SBM experiments..."
)

A_MEM_RUNS = []

L_MEM_RUNS = []

A_QUERY_RUNS = []

L_QUERY_RUNS = []


for trial in tqdm(
    range(N_TRIALS),
    desc="Independent runs"
):

    A_mem = []

    L_mem = []

    A_query = []

    L_query = []

    for i in range(N_MEM):


        # One graph = one stored associative memory
  

        A = sample_sbm(
            B_LIST[i],
            SEED
            +
            10000000*trial
            +
            1000*i
        )

        L = laplacian(
            A
        )

        # Query = 20%-masked VERSION OF THIS SAME GRAPH
 

        Aq = mask_existing_edges(
            A,
            MASK_RATE,
            SEED
            +
            500000000
            +
            10000000*trial
            +
            i
        )

        Lq = laplacian(
            Aq
        )

        A_mem.append(A)

        L_mem.append(L)

        A_query.append(Aq)

        L_query.append(Lq)

    A_MEM_RUNS.append(
        np.stack(
            A_mem
        )
    )

    L_MEM_RUNS.append(
        np.stack(
            L_mem
        )
    )

    A_QUERY_RUNS.append(
        np.stack(
            A_query
        )
    )

    L_QUERY_RUNS.append(
        np.stack(
            L_query
        )
    )


target = np.arange(
    N_MEM
)

L_DIM = (
    N_NODES
    *
    N_NODES
)


# 6. OPERATOR DISTANCE FOR ONE RUN


def op_d2(
    Q,
    L_mem
):

    batch = len(Q)

    out = np.empty(
        (
            batch,
            N_MEM
        ),
        dtype=float
    )

    for b in range(batch):

        E = (
            Q[b][None,:,:]
            -
            L_mem
        )

        # symmetric matrices
        eig = np.linalg.eigvalsh(
            E
        )

        out[b] = (
            np.max(
                np.abs(eig),
                axis=1
            )**2
        )

    return out



# Eu INNER-PRODUCT SCORE FOR ONE RUN


def eu_score(
    Q,
    L_mem_flat
):

    Q_flat = Q.reshape(
        len(Q),
        L_DIM
    )

    return (
        Q_flat
        @
        L_mem_flat.T
    )


# 7. HARD RETRIEVAL DIAGNOSTIC


print(
    "\n[3/9] Computing hard retrieval..."
)

HARD_OP = []

HARD_EU = []

HARD_OP_PREDS = []

HARD_EU_PREDS = []


for trial in tqdm(
    range(N_TRIALS),
    desc="Hard retrieval"
):

    L_mem = L_MEM_RUNS[
        trial
    ]

    Lq = L_QUERY_RUNS[
        trial
    ]

    L_flat = L_mem.reshape(
        N_MEM,
        L_DIM
    )

    pred_op = np.argmin(
        op_d2(
            Lq,
            L_mem
        ),
        axis=1
    )

    pred_eu = np.argmax(
        eu_score(
            Lq,
            L_flat
        ),
        axis=1
    )

    HARD_OP_PREDS.append(
        pred_op
    )

    HARD_EU_PREDS.append(
        pred_eu
    )

    HARD_OP.append(
        np.mean(
            pred_op == target
        )
    )

    HARD_EU.append(
        np.mean(
            pred_eu == target
        )
    )


HARD_OP = np.array(
    HARD_OP
)

HARD_EU = np.array(
    HARD_EU
)


print("\n" + "="*94)

print(
    "LARGE-beta HARD RETRIEVAL"
)

print("="*94)

print(
    f"Op-DAM      : "
    f"{HARD_OP.mean():.3f} +/- "
    f"{HARD_OP.std(ddof=1)/np.sqrt(N_TRIALS):.3f} SE"
)

print(
    f"Eu-DAM      : "
    f"{HARD_EU.mean():.3f} +/- "
    f"{HARD_EU.std(ddof=1)/np.sqrt(N_TRIALS):.3f} SE"
)

print(
    f"Chance      : "
    f"{1/N_MEM:.3f}"
)



# 8. RUN-SPECIFIC TEMPERATURE SCALES


print(
    "\n[4/9] Computing run-specific temperature scales..."
)

SCALE_OP = []

SCALE_EU = []


for trial in tqdm(
    range(N_TRIALS),
    desc="Temperature scales"
):

    L_mem = L_MEM_RUNS[
        trial
    ]

    L_flat = L_mem.reshape(
        N_MEM,
        L_DIM
    )


    # Operator scale from between-memory distances


    op_pairs = []

    for i in range(N_MEM):

        E = (
            L_mem[i][None,:,:]
            -
            L_mem[
                i+1:
            ]
        )

        if len(E) == 0:
            continue

        eig = np.linalg.eigvalsh(
            E
        )

        op_pairs.extend(
            (
                np.max(
                    np.abs(eig),
                    axis=1
                )**2
            ).tolist()
        )

    SCALE_OP.append(
        max(
            np.median(
                op_pairs
            ),
            EPS
        )
    )

    # Eu-DAM score scale


    S = (
        L_flat
        @
        L_flat.T
    )

    SCALE_EU.append(
        max(
            np.median(
                np.std(
                    S,
                    axis=1
                )
            ),
            EPS
        )
    )


SCALE_OP = np.array(
    SCALE_OP
)

SCALE_EU = np.array(
    SCALE_EU
)


def softmax_rows(Z):

    Z = (
        Z
        -
        np.max(
            Z,
            axis=1,
            keepdims=True
        )
    )

    E = np.exp(Z)

    return (
        E
        /
        np.maximum(
            E.sum(
                axis=1,
                keepdims=True
            ),
            EPS
        )
    )


# 9. Op-DAM RETRIEVAL FOR ONE RUN


def op_retrieve(
    Q0,
    L_mem,
    L_flat,
    beta,
    scale
):

    Q = Q0.copy()

    previous = None

    for _ in range(N_STEPS):

        D = (
            op_d2(
                Q,
                L_mem
            )
            /
            scale
        )

        W = softmax_rows(
            -beta
            *
            D
        )

        Q = (
            W
            @
            L_flat
        ).reshape(
            len(Q),
            N_NODES,
            N_NODES
        )

        pred = np.argmin(
            op_d2(
                Q,
                L_mem
            ),
            axis=1
        )

        if (
            previous is not None
            and
            np.array_equal(
                previous,
                pred
            )
        ):

            break

        previous = pred

    return pred



# 10. Eu-DAM RETRIEVAL FOR ONE RUN


def eu_retrieve(
    Q0,
    L_flat,
    beta,
    scale
):

    Q = Q0.reshape(
        len(Q0),
        L_DIM
    ).copy()

    previous = None

    for _ in range(N_STEPS):

        score = (
            Q
            @
            L_flat.T
        ) / scale

        W = softmax_rows(
            beta
            *
            score
        )

        Q = (
            W
            @
            L_flat
        )

        pred = np.argmax(
            Q
            @
            L_flat.T,
            axis=1
        )

        if (
            previous is not None
            and
            np.array_equal(
                previous,
                pred
            )
        ):

            break

        previous = pred

    return pred



# 11. BETA SWEEP


print(
    "\n[5/9] Running inverse-temperature sweep..."
)

ACC_OP = np.zeros(
    (
        N_TRIALS,
        len(BETA_N)
    )
)

ACC_EU = np.zeros_like(
    ACC_OP
)


bar = tqdm(
    total=
        N_TRIALS
        *
        len(BETA_N),
    desc="Independent runs"
)


start = time.time()


for trial in range(N_TRIALS):

    L_mem = L_MEM_RUNS[
        trial
    ]

    Lq = L_QUERY_RUNS[
        trial
    ]

    L_flat = L_mem.reshape(
        N_MEM,
        L_DIM
    )

    for j,beta in enumerate(
        BETA_N
    ):

        pred_op = op_retrieve(
            Lq,
            L_mem,
            L_flat,
            beta,
            SCALE_OP[trial]
        )

        ACC_OP[
            trial,
            j
        ] = np.mean(
            pred_op == target
        )

        pred_eu = eu_retrieve(
            Lq,
            L_flat,
            beta,
            SCALE_EU[trial]
        )

        ACC_EU[
            trial,
            j
        ] = np.mean(
            pred_eu == target
        )

        bar.update(1)


bar.close()


print(
    f"Sweep completed in "
    f"{time.time()-start:.1f}s"
)




def mean_and_se(X):

    mean = X.mean(
        axis=0
    )

    se = (
        X.std(
            axis=0,
            ddof=1
        )
        /
        np.sqrt(
            X.shape[0]
        )
    )

    return mean,se


op_mean,op_se = mean_and_se(
    ACC_OP
)

eu_mean,eu_se = mean_and_se(
    ACC_EU
)


# 12. ACCURACY FIGURE


print(
    "\n[6/9] Saving retrieval-accuracy figure..."
)

fig,ax = plt.subplots(
    figsize=(7.3,4.9)
)


ax.errorbar(
    BETA_N,
    op_mean,
    yerr=op_se,
    marker="o",
    markersize=5,
    linewidth=2.2,
    elinewidth=1,
    capsize=3,
    label="Graph-DAM"
)


ax.errorbar(
    BETA_N,
    eu_mean,
    yerr=eu_se,
    marker="s",
    markersize=5,
    linewidth=2.0,
    elinewidth=1,
    capsize=3,
    label=r"Eu-DAM"
)


ax.set_xscale(
    "log"
)

ax.set_ylim(
    0,
    1.03
)

ax.set_xlabel(
    r"Inverse temperature $\beta_n$"
)

ax.set_ylabel(
    "Retrieval accuracy"
)

ax.set_title(
    "SBM Community-Network Retrieval"
)

ax.grid(
    alpha=.22
)

ax.legend(
    frameon=True
)

fig.tight_layout()


fig.savefig(
    os.path.join(
        OUTDIR,
        "sbm_retrieval_accuracy.pdf"
    ),
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTDIR,
        "sbm_retrieval_accuracy.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()



# 13. SELECT REPRESENTATIVE EXAMPLE


print(
    "\n[7/9] Selecting illustrative retrieval..."
)

choice = None
best_margin = -np.inf


for trial in range(N_TRIALS):

    pred_op = HARD_OP_PREDS[
        trial
    ]

    pred_eu = HARD_EU_PREDS[
        trial
    ]

    candidates = np.where(
        (
            pred_op == target
        )
        &
        (
            pred_eu != target
        )
    )[0]

    for i in candidates:

        D = op_d2(
            L_QUERY_RUNS[
                trial
            ][i:i+1],
            L_MEM_RUNS[
                trial
            ]
        )[0]

        order = np.argsort(D)

        margin = (
            D[order[1]]
            -
            D[order[0]]
        )

        if margin > best_margin:

            best_margin = margin

            choice = (
                trial,
                int(i)
            )


if choice is None:

    for trial in range(N_TRIALS):

        pred_op = HARD_OP_PREDS[
            trial
        ]

        candidates = np.where(
            pred_op == target
        )[0]

        if len(candidates) > 0:

            choice = (
                trial,
                int(candidates[0])
            )

            break


if choice is None:

    choice = (
        0,
        0
    )


trial_vis,true_idx = choice


op_idx = int(
    HARD_OP_PREDS[
        trial_vis
    ][true_idx]
)


eu_idx = int(
    HARD_EU_PREDS[
        trial_vis
    ][true_idx]
)


print(
    "Run         :",
    trial_vis
)

print(
    "True memory :",
    true_idx
)

print(
    "Graph-DAM      :",
    op_idx
)

print(
    "Eu-DAM      :",
    eu_idx
)


A_MEM_VIS = A_MEM_RUNS[
    trial_vis
]


# 14. FIXED COMMUNITY-AWARE LAYOUT


def fixed_layout():

    pos = np.zeros(
        (
            N_NODES,
            2
        )
    )

    outer_radius = 3.2
    inner_radius = 0.82

    for k in range(K):

        theta = (
            2*np.pi*k/K
        )

        center = np.array([
            outer_radius
            *
            np.cos(theta),

            outer_radius
            *
            np.sin(theta)
        ])

        ids = np.where(
            z == k
        )[0]

        for j,node in enumerate(ids):

            phi = (
                2*np.pi*j
                /
                len(ids)
            )

            pos[node] = (
                center
                +
                inner_radius
                *
                np.array([
                    np.cos(phi),
                    np.sin(phi)
                ])
            )

    return pos


POS = fixed_layout()



# SAVE NETWORK


def save_network(
    A,
    title,
    filename
):

    fig,ax = plt.subplots(
        figsize=(5.8,5.3)
    )

    # all realized edges
    edges = np.argwhere(
        np.triu(
            A > 0,
            1
        )
    )

    for i,j in edges:

        ax.plot(
            [
                POS[i,0],
                POS[j,0]
            ],
            [
                POS[i,1],
                POS[j,1]
            ],
            linewidth=.35,
            alpha=.11,
            zorder=1
        )

    for k in range(K):

        ids = np.where(
            z == k
        )[0]

        ax.scatter(
            POS[ids,0],
            POS[ids,1],
            s=39,
            edgecolors="white",
            linewidths=.45,
            label=f"Community {k+1}",
            zorder=3
        )

    ax.set_title(
        title,
        pad=8
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )

    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTDIR,
            filename + ".pdf"
        ),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            OUTDIR,
            filename + ".png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)



# 15. SAVE TRUE / OP / EU NETWORKS


print(
    "\n[8/9] Saving graph figures..."
)


save_network(
    A_MEM_VIS[
        true_idx
    ],
    f"True Memory: SBM {true_idx}",
    "true_memory_graph"
)


save_network(
    A_MEM_VIS[
        op_idx
    ],
    (
        f"Graph-DAM Retrieval: SBM {op_idx}"
        +
        (
            " (Correct)"
            if op_idx == true_idx
            else " (Incorrect)"
        )
    ),
    "op_dam_graph"
)


save_network(
    A_MEM_VIS[
        eu_idx
    ],
    (
        f"Eu-DAM Retrieval: SBM {eu_idx}"
        +
        (
            " (Correct)"
            if eu_idx == true_idx
            else " (Incorrect)"
        )
    ),
    "eu_dam_graph"
)


# 16. FIEDLER SIGNAL


def fiedler_vector(A):

    L = laplacian(A)

    eigvals,eigvecs = np.linalg.eigh(
        L
    )

    order = np.argsort(
        eigvals
    )

    lambda2 = eigvals[
        order[1]
    ]

    f = eigvecs[
        :,
        order[1]
    ].copy()

    # eigenvector sign is arbitrary
    if np.mean(
        f[:COMM_SIZE]
    ) > 0:

        f *= -1

    return lambda2,f


lam_true,f_true = fiedler_vector(
    A_MEM_VIS[
        true_idx
    ]
)

lam_op,f_op = fiedler_vector(
    A_MEM_VIS[
        op_idx
    ]
)

lam_eu,f_eu = fiedler_vector(
    A_MEM_VIS[
        eu_idx
    ]
)


FMAX = max(
    np.max(
        np.abs(f_true)
    ),
    np.max(
        np.abs(f_op)
    ),
    np.max(
        np.abs(f_eu)
    )
)

FMAX *= 1.12


def save_fiedler(
    f,
    lambda2,
    title,
    filename
):

    fig,ax = plt.subplots(
        figsize=(5.4,3.6)
    )

    x = np.arange(
        N_NODES
    )

    ax.plot(
        x,
        f,
        linewidth=1.8
    )

    for boundary in range(
        COMM_SIZE,
        N_NODES,
        COMM_SIZE
    ):

        ax.axvline(
            boundary-.5,
            linestyle="--",
            linewidth=.8,
            alpha=.55
        )

    ax.axhline(
        0,
        linewidth=.7,
        alpha=.3
    )

    ax.set_xlim(
        0,
        N_NODES-1
    )

    ax.set_ylim(
        -FMAX,
        FMAX
    )

    ax.set_xlabel(
        "Node index"
    )

    ax.set_ylabel(
        "Fiedler coordinate"
    )

    ax.set_title(
        title,
        pad=7
    )

    ax.text(
        .98,
        .05,
        rf"$\lambda_2={lambda2:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10
    )

    ax.grid(
        axis="y",
        alpha=.15
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTDIR,
            filename+".pdf"
        ),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            OUTDIR,
            filename+".png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)


save_fiedler(
    f_true,
    lam_true,
    "Target Fiedler Vector",
    "true_memory_fiedler"
)


save_fiedler(
    f_op,
    lam_op,
    "Graph-DAM Retrieved Fiedler Vector",
    "op_dam_fiedler"
)


save_fiedler(
    f_eu,
    lam_eu,
    "Eu-DAM Retrieved Fiedler Vector",
    "eu_dam_fiedler"
)



# SUMMARY


print(
    "\n[9/9] SUMMARY"
)

print("="*94)

print(
    f"Stored graphs         : {N_MEM}"
)

print(
    f"Nodes                 : {N_NODES}"
)

print(
    f"Communities           : {K}"
)

print(
    f"Masked existing edges : {100*MASK_RATE:.0f}%"
)

print(
    f"Independent runs      : {N_TRIALS}"
)

print()

print(
    "Memory:"
)

print(
    "  one independently sampled SBM graph"
)

print()

print(
    "Query:"
)

print(
    "  the SAME stored graph with 20% of its existing edges removed"
)

print()

print(
    "Methods:"
)

print(
    "  Op-DAM on combinatorial Laplacian L"
)

print(
    "  Eu-DAM on vec(L) using inner-product similarity"
)

print()

print(
    "Hard retrieval:"
)

print(
    f"  Graph-DAM = "
    f"{HARD_OP.mean():.3f} "
    f"+/- "
    f"{HARD_OP.std(ddof=1)/np.sqrt(N_TRIALS):.3f} SE"
)

print(
    f"  Eu-DAM = "
    f"{HARD_EU.mean():.3f} "
    f"+/- "
    f"{HARD_EU.std(ddof=1)/np.sqrt(N_TRIALS):.3f} SE"
)

print(
    f"  Chance = {1/N_MEM:.3f}"
)

print()

print(
    "Output directory:"
)

print(
    OUTDIR
)