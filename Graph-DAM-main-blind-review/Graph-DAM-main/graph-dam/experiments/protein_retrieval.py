# ============================================================
# ONE-CELL REAL PROTEIN ENSEMBLE RETRIEVAL
#
# KEY IDEA:
#
#   Airline:
#       many flights -> one stable airline graph
#
#   Protein:
#       many experimental NMR conformations
#       -> one stable protein contact-network memory
#
# ============================================================
#
# For protein identity i:
#
# MEMORY:
#   first M_MEM real NMR conformations
#
#       L_i^mem = mean_s L(X_is)
#
#   Raw baseline receives SAME amount of data:
#
#       X_i^mem = mean_s X_is
#
# QUERY:
#   a completely HELD-OUT NMR conformation
#   + 20% masking of existing NON-BACKBONE contact edges
#
#   1. Build the full contact graph from the held-out conformation
#   2. Remove 20% of its non-backbone contact edges
#   3. Recompute Lq on the same 80-residue node set
#
# Graph-DAM and Eu-DAM use the SAME corrupted Lq.
# Raw Eu-DAM uses the same held-out C-alpha coordinates directly.
#
# ------------------------------------------------------------
# CONTACT GRAPH
#
# Weighted contacts within 8 Angstrom:
#
#   A_rs = exp[-(d_rs / 8)^2],  if d_rs <= 8 A
#
# Backbone connections retained.
#
# Combinatorial Laplacian:
#
#   L = D - A
#
# ------------------------------------------------------------
#
# No noise.
# No random rotations.
# No normalized Laplacian.
# No pairwise-distance baseline.
# No artificial weakening of Eu baselines.
#
# Saves:
#   protein_retrieval_accuracy.pdf/png
#
#   individual contact graphs:
#       true_memory_contact_graph.*
#       op_dam_contact_graph.*
#       eu_dam_L_contact_graph.*
#       raw_eu_dam_contact_graph.*
#
#   individual 3D protein structures:
#       true_memory_protein_3d.*
#       op_dam_protein_3d.*
#       eu_dam_L_protein_3d.*
#       raw_eu_dam_protein_3d.*
#
# ============================================================

import os
import sys
import subprocess
import pkgutil
import urllib.request
import random
import warnings
import time

warnings.filterwarnings("ignore")

# ============================================================
# DEPENDENCY
# ============================================================

if pkgutil.find_loader("Bio") is None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "biopython"
    ])

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from Bio.PDB import PDBParser

# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
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

OUTDIR = os.path.join("outputs", "protein")
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# 80 tends to retain substantially more NMR proteins than 120.
#
# You can try 100 or 120 afterward if enough proteins survive.
# ------------------------------------------------------------

N_RES = 80

# Number of REAL conformations aggregated into each memory
M_MEM = 6

# held-out model index:
# first M_MEM are memory;
# later models are available as queries
MIN_MODELS = M_MEM + 2

CONTACT_RADIUS = 8.0

MASK_RATE = 0.20

N_TRIALS = 5

N_STEPS = 5

BETA_N = np.logspace(
    -1,
    3,
    25
)

EPS = 1e-10

# ============================================================
# NMR ENSEMBLE CANDIDATES
#
# Code automatically verifies:
#   - enough NMR models
#   - one common chain
#   - >= N_RES common CA residues across ALL selected models
#
# Entries that fail are simply skipped.
# ============================================================

PDB_IDS = [
    "1D3Z",
    "2K39",
    "1XQQ",
    "2K5K",
    "2K6O",
    "2KFW",
    "2KJ3",
    "2KKJ",
    "2KOC",
    "2KRS",
    "2KUB",
    "2KYY",
    "2L3Y",
    "2L7B",
    "2L8O",
    "2LA6",
    "2LCB",
    "2LD5",
    "2LEU",
    "2LF2",
    "2LGW",
    "2LJ3",
    "2LKY",
    "2LMG",
    "2LNL",
    "2LO1",
    "2LP5",
    "2LQ7",
    "2LR1",
    "2LSB",
    "2LTM",
    "2LU1",
    "2LV8",
    "2LWA",
    "2LXY",
    "2M0Y",
    "2M2J",
    "2M3B",
    "2M5M",
    "2M6Q",
    "2M7U",
    "2M8L",
    "2M9F",
    "2MA4",
    "2MB5",
    "2MC5",
    "2MD6",
    "2ME9",
    "2MF9",
    "2MGN",
    "2MHN",
    "2MI6",
    "2MJ9",
    "2MK5",
    "2ML8"
]

PDB_DIR = "/tmp/protein_nmr_ensemble_dam"

os.makedirs(
    PDB_DIR,
    exist_ok=True
)

print("=" * 88)
print("REAL PROTEIN ENSEMBLE CONTACT-NETWORK RETRIEVAL")
print("=" * 88)

# ============================================================
# 1. LOAD PDB ENSEMBLES
# ============================================================

parser = PDBParser(
    QUIET=True
)


def download_pdb(pdb_id):

    path = os.path.join(
        PDB_DIR,
        pdb_id + ".pdb"
    )

    if os.path.exists(path):
        return path

    url = (
        "https://files.rcsb.org/"
        f"download/{pdb_id}.pdb"
    )

    try:

        urllib.request.urlretrieve(
            url,
            path
        )

        return path

    except Exception:

        return None


def model_chain_dict(
    model,
    chain_id
):

    if chain_id not in model:
        return {}

    out = {}

    for residue in model[chain_id]:

        if residue.id[0] != " ":
            continue

        if "CA" not in residue:
            continue

        key = (
            int(residue.id[1]),
            str(residue.id[2]).strip()
        )

        out[key] = (
            residue["CA"]
            .coord
            .astype(float)
        )

    return out


def extract_ensemble(
    pdb_id,
    path
):

    try:

        structure = parser.get_structure(
            pdb_id,
            path
        )

        models = list(
            structure.get_models()
        )

        if len(models) < MIN_MODELS:
            return None

        # ----------------------------------------------------
        # Find a chain existing in all models.
        # ----------------------------------------------------

        common_chains = None

        for model in models:

            ids = {
                chain.id
                for chain in model
            }

            if common_chains is None:
                common_chains = ids

            else:
                common_chains &= ids

        if not common_chains:
            return None

        best = None

        for chain_id in common_chains:

            dicts = [
                model_chain_dict(
                    model,
                    chain_id
                )
                for model in models
            ]

            common_res = None

            for D in dicts:

                keys = set(D.keys())

                if common_res is None:
                    common_res = keys

                else:
                    common_res &= keys

            common_res = sorted(
                common_res
            )

            if (
                best is None
                or
                len(common_res) > len(best[0])
            ):

                best = (
                    common_res,
                    chain_id,
                    dicts
                )

        if best is None:
            return None

        keys, chain_id, dicts = best

        if len(keys) < N_RES:
            return None

        keys = keys[:N_RES]

        X_models = []

        for D in dicts:

            X = np.stack([
                D[k]
                for k in keys
            ])

            X_models.append(
                X
            )

        return (
            chain_id,
            np.stack(X_models)
        )

    except Exception:

        return None


print(
    "\n[1/10] Loading experimental NMR ensembles..."
)

protein_ids = []

ENSEMBLES = []

for pdb_id in tqdm(
    PDB_IDS,
    desc="NMR ensembles"
):

    path = download_pdb(
        pdb_id
    )

    if path is None:
        continue

    result = extract_ensemble(
        pdb_id,
        path
    )

    if result is None:
        continue

    chain_id, X_models = result

    protein_ids.append(
        f"{pdb_id}:{chain_id}"
    )

    ENSEMBLES.append(
        X_models
    )


N = len(
    protein_ids
)

if N < 5:

    raise RuntimeError(
        f"Only {N} usable ensembles with N_RES={N_RES}. "
        "Try N_RES=60."
    )


print(
    "\nProtein memories:",
    N
)

print(
    "Residues per protein:",
    N_RES
)

print(
    "Conformations per memory:",
    M_MEM
)

print()

for i in range(N):

    print(
        f"{i:2d} | "
        f"{protein_ids[i]:8s} | "
        f"models={len(ENSEMBLES[i])}"
    )


target = np.arange(
    N
)

# ============================================================
# 2. CENTER EVERY REAL CONFORMATION
#
# Translation only.
# No rotation / Procrustes.
# ============================================================

def center_structure(X):

    return (
        X
        -
        X.mean(
            axis=0,
            keepdims=True
        )
    )


for i in range(N):

    ENSEMBLES[i] = np.stack([
        center_structure(X)
        for X in ENSEMBLES[i]
    ])

# ============================================================
# 3. 8-A WEIGHTED CONTACT NETWORK
#
# Stable but informative:
#
# A_rs = exp[-(d/8)^2]  for d <= 8 A
#
# Backbone edges retained.
# ============================================================

def pairwise_distance(X):

    diff = (
        X[:, None, :]
        -
        X[None, :, :]
    )

    return np.sqrt(
        np.sum(
            diff * diff,
            axis=-1
        )
    )


def adjacency_from_coords(X):

    R = pairwise_distance(
        X
    )

    A = np.zeros(
        (
            N_RES,
            N_RES
        ),
        dtype=np.float64
    )

    contact = (
        (R <= CONTACT_RADIUS)
        &
        (R > EPS)
    )

    A[contact] = np.exp(
        -(
            R[contact]
            /
            CONTACT_RADIUS
        )**2
    )

    # --------------------------------------------------------
    # Stable physical chain structure
    # --------------------------------------------------------

    for r in range(
        N_RES - 1
    ):

        A[r, r+1] = max(
            A[r, r+1],
            1.0
        )

        A[r+1, r] = max(
            A[r+1, r],
            1.0
        )

    np.fill_diagonal(
        A,
        0.0
    )

    return A


def laplacian_from_A(A):

    d = A.sum(
        axis=1
    )

    return (
        np.diag(d)
        -
        A
    )

# ============================================================
# 4. AGGREGATED MEMORIES
#
# CRUCIAL CHANGE:
#
# L_MEMORY[i] =
#       mean of contact Laplacians from M_MEM real conformations
#
# Raw baseline receives identical number of conformations:
#
# X_MEMORY[i] =
#       mean of corresponding coordinate matrices
#
# ============================================================

print(
    "\n[2/10] Aggregating real conformations into memories..."
)

A_MEM = []

L_MEM = []

X_MEMORY = []

for i in range(N):

    models = ENSEMBLES[i][
        :M_MEM
    ]

    A_models = np.stack([
        adjacency_from_coords(X)
        for X in models
    ])

    L_models = np.stack([
        laplacian_from_A(A)
        for A in A_models
    ])

    # --------------------------------------------------------
    # Graph memory
    # --------------------------------------------------------

    Lbar = L_models.mean(
        axis=0
    )

    # equivalent average adjacency for visualization
    Abar = A_models.mean(
        axis=0
    )

    # --------------------------------------------------------
    # Raw memory gets SAME number of conformations
    # --------------------------------------------------------

    Xbar = models.mean(
        axis=0
    )

    Xbar = center_structure(
        Xbar
    )

    A_MEM.append(
        Abar
    )

    L_MEM.append(
        Lbar
    )

    X_MEMORY.append(
        Xbar
    )


A_MEM = np.stack(
    A_MEM
)

L_MEM = np.stack(
    L_MEM
)

X_MEMORY = np.stack(
    X_MEMORY
)


L_DIM = (
    N_RES
    *
    N_RES
)

X_DIM = (
    3
    *
    N_RES
)


L_FLAT = L_MEM.reshape(
    N,
    L_DIM
)

X_FLAT = X_MEMORY.reshape(
    N,
    X_DIM
)

# ============================================================
# DIAGNOSTIC:
# average graph density
# ============================================================

print(
    "\nAggregated graph diagnostics:"
)

for i in range(N):

    degree = A_MEM[i].sum(
        axis=1
    )

    print(
        f"{protein_ids[i]:8s} | "
        f"mean degree={degree.mean():6.2f} | "
        f"max degree={degree.max():6.2f}"
    )

# ============================================================
# 5. HELD-OUT QUERY WITH 20% CONTACT-EDGE MASKING
#
# Query model is NEVER used in memory construction.
# Different held-out experimental conformations are selected
# across trials where available.
#
# Corruption acts only on existing non-backbone contacts.
# The complete 80-residue node set and sequential backbone
# connections are retained, so the query remains 80 x 80.
# ============================================================

def mask_contact_edges(A, rate, seed):

    rng = np.random.default_rng(seed)
    Aq = A.copy()

    # Existing unordered edges.
    edges = np.argwhere(np.triu(A > 0, 1))

    # Sequential (r,r+1) edges encode the known protein backbone
    # and are therefore not corrupted.
    contact_edges = np.array([
        [r, s]
        for r, s in edges
        if abs(int(r) - int(s)) > 1
    ], dtype=int)

    if len(contact_edges) == 0:
        return Aq

    n_mask = max(
        1,
        int(round(rate * len(contact_edges)))
    )
    n_mask = min(n_mask, len(contact_edges))

    chosen = rng.choice(
        len(contact_edges),
        size=n_mask,
        replace=False
    )
    removed = contact_edges[chosen]

    Aq[removed[:, 0], removed[:, 1]] = 0.0
    Aq[removed[:, 1], removed[:, 0]] = 0.0

    return Aq


print(
    f"\n[3/10] Building held-out queries + "
    f"{100*MASK_RATE:.0f}% non-backbone contact-edge masking..."
)

Q_L = []
Q_X = []
Q_A = []
QUERY_MODEL_INDEX = []

for trial in range(N_TRIALS):

    trial_L = []
    trial_X = []
    trial_A = []
    trial_model_idx = []

    for i in range(N):

        ensemble = ENSEMBLES[i]

        # Choose among experimental models NEVER used in memory.
        available = np.arange(M_MEM, len(ensemble))
        q_idx = available[trial % len(available)]
        Xclean = ensemble[q_idx].copy()

        # Build the genuine held-out contact graph, then mask
        # 20% of existing non-backbone contacts.
        Aclean = adjacency_from_coords(Xclean)
        Aq = mask_contact_edges(
            Aclean,
            MASK_RATE,
            SEED + 10000 * trial + i
        )
        Lq = laplacian_from_A(Aq)

        # Graph-DAM and Eu-DAM on L use the same corrupted query.
        trial_A.append(Aq)
        trial_L.append(Lq)

        # Raw Eu-DAM uses the same held-out experimental
        # conformation directly, without constructing a graph.
        trial_X.append(Xclean.reshape(-1))
        trial_model_idx.append(q_idx)

    Q_A.append(np.stack(trial_A))
    Q_L.append(np.stack(trial_L))
    Q_X.append(np.stack(trial_X))
    QUERY_MODEL_INDEX.append(trial_model_idx)

# ============================================================
# 6. OPERATOR DISTANCE
# ============================================================

def op_d2(Q):

    B = len(Q)

    out = np.empty(
        (
            B,
            N
        ),
        dtype=np.float64
    )

    for b in range(B):

        E = (
            Q[b][None, :, :]
            -
            L_MEM
        )

        eig = np.linalg.eigvalsh(
            E
        )

        op = np.max(
            np.abs(eig),
            axis=1
        )

        out[b] = (
            op * op
        )

    return out

# ============================================================
# INNER PRODUCTS
# ============================================================

def graph_score_flat(Q):

    return (
        Q
        @
        L_FLAT.T
    )


def graph_score(Q):

    return graph_score_flat(
        Q.reshape(
            len(Q),
            L_DIM
        )
    )


def raw_score_flat(Q):

    return (
        Q
        @
        X_FLAT.T
    )


def raw_score(Q):

    return raw_score_flat(
        Q.reshape(
            len(Q),
            X_DIM
        )
    )

# ============================================================
# 7. TEMPERATURE SCALES
# ============================================================

print(
    "\n[4/10] Computing inverse-temperature scales..."
)

op_pairs = []

for i in range(N):

    for j in range(
        i+1,
        N
    ):

        E = (
            L_MEM[i]
            -
            L_MEM[j]
        )

        eig = np.linalg.eigvalsh(
            E
        )

        d = np.max(
            np.abs(eig)
        )

        op_pairs.append(
            d*d
        )


SCALE_OP = max(
    np.median(
        op_pairs
    ),
    EPS
)


S_LL = (
    L_FLAT
    @
    L_FLAT.T
)

SCALE_EU_L = max(
    np.median(
        np.std(
            S_LL,
            axis=1
        )
    ),
    EPS
)


S_XX = (
    X_FLAT
    @
    X_FLAT.T
)

SCALE_RAW = max(
    np.median(
        np.std(
            S_XX,
            axis=1
        )
    ),
    EPS
)

# ============================================================
# 8. HARD RETRIEVAL
# ============================================================

hard_op_pred = []

hard_eu_pred = []

hard_raw_pred = []


for trial in range(
    N_TRIALS
):

    hard_op_pred.append(
        np.argmin(
            op_d2(
                Q_L[trial]
            ),
            axis=1
        )
    )

    hard_eu_pred.append(
        np.argmax(
            graph_score(
                Q_L[trial]
            ),
            axis=1
        )
    )

    hard_raw_pred.append(
        np.argmax(
            raw_score(
                Q_X[trial]
            ),
            axis=1
        )
    )


def hard_accuracy(P):

    return np.array([
        np.mean(
            pred == target
        )
        for pred in P
    ])


HA_OP = hard_accuracy(
    hard_op_pred
)

HA_EU = hard_accuracy(
    hard_eu_pred
)

HA_RAW = hard_accuracy(
    hard_raw_pred
)


print(
    "\n" + "="*88
)

print(
    "LARGE-beta HARD RETRIEVAL ACCURACY"
)

print(
    "="*88
)

print(
    f"Op-DAM                : "
    f"{HA_OP.mean():.3f} +/- "
    f"{HA_OP.std():.3f}"
)

print(
    f"Eu-DAM on L           : "
    f"{HA_EU.mean():.3f} +/- "
    f"{HA_EU.std():.3f}"
)

print(
    f"Eu-DAM on original X  : "
    f"{HA_RAW.mean():.3f} +/- "
    f"{HA_RAW.std():.3f}"
)

# ============================================================
# SOFTMAX
# ============================================================

def softmax_rows(Z):

    Z = (
        Z
        -
        Z.max(
            axis=1,
            keepdims=True
        )
    )

    E = np.exp(
        Z
    )

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

# ============================================================
# 9A. OP-DAM
# ============================================================

def op_retrieve(
    Q0,
    beta
):

    Q = Q0.copy()

    last = None


    for _ in range(
        N_STEPS
    ):

        D = (
            op_d2(Q)
            /
            SCALE_OP
        )

        W = softmax_rows(
            -beta * D
        )

        Q = (
            W
            @
            L_FLAT
        ).reshape(
            len(Q),
            N_RES,
            N_RES
        )

        pred = np.argmin(
            op_d2(Q),
            axis=1
        )


        if (
            last is not None
            and
            np.array_equal(
                pred,
                last
            )
        ):
            break


        last = pred


    return pred

# ============================================================
# 9B. Eu-DAM ON EXACT SAME L
# ============================================================

def eu_l_retrieve(
    Q0,
    beta
):

    Q = Q0.reshape(
        len(Q0),
        L_DIM
    ).copy()

    last = None


    for _ in range(
        N_STEPS
    ):

        score = (
            graph_score_flat(Q)
            /
            SCALE_EU_L
        )

        W = softmax_rows(
            beta * score
        )

        Q = (
            W
            @
            L_FLAT
        )

        pred = np.argmax(
            graph_score_flat(
                Q
            ),
            axis=1
        )


        if (
            last is not None
            and
            np.array_equal(
                pred,
                last
            )
        ):
            break


        last = pred


    return pred

# ============================================================
# 9C. Eu-DAM ON ORIGINAL COORDINATES
# ============================================================

def raw_eu_retrieve(
    Q0,
    beta
):

    Q = Q0.reshape(
        len(Q0),
        X_DIM
    ).copy()

    last = None


    for _ in range(
        N_STEPS
    ):

        score = (
            raw_score_flat(Q)
            /
            SCALE_RAW
        )

        W = softmax_rows(
            beta * score
        )

        Q = (
            W
            @
            X_FLAT
        )

        pred = np.argmax(
            raw_score_flat(
                Q
            ),
            axis=1
        )


        if (
            last is not None
            and
            np.array_equal(
                pred,
                last
            )
        ):
            break


        last = pred


    return pred

# ============================================================
# 10. TEMPERATURE SWEEP
# ============================================================

print(
    "\n[5/10] Running inverse-temperature sweep..."
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


ACC_RAW = np.zeros_like(
    ACC_OP
)


bar = tqdm(
    total=
        N_TRIALS
        *
        len(BETA_N)
        *
        3,
    desc="Protein DAM"
)


start = time.time()


for trial in range(
    N_TRIALS
):

    for j, beta in enumerate(
        BETA_N
    ):

        # OP
        pred = op_retrieve(
            Q_L[trial],
            beta
        )

        ACC_OP[
            trial,
            j
        ] = np.mean(
            pred == target
        )

        bar.update(1)


        # Eu-L
        pred = eu_l_retrieve(
            Q_L[trial],
            beta
        )

        ACC_EU[
            trial,
            j
        ] = np.mean(
            pred == target
        )

        bar.update(1)


        # Raw Eu
        pred = raw_eu_retrieve(
            Q_X[trial],
            beta
        )

        ACC_RAW[
            trial,
            j
        ] = np.mean(
            pred == target
        )

        bar.update(1)


bar.close()


print(
    f"Sweep completed in "
    f"{time.time()-start:.1f}s"
)

# ============================================================
# MEAN + STANDARD ERROR
# ============================================================

op_mean = ACC_OP.mean(
    axis=0
)

eu_mean = ACC_EU.mean(
    axis=0
)

raw_mean = ACC_RAW.mean(
    axis=0
)


def standard_error(X):

    return (
        X.std(
            axis=0,
            ddof=1
        )
        /
        np.sqrt(
            X.shape[0]
        )
    )


op_se = standard_error(
    ACC_OP
)

eu_se = standard_error(
    ACC_EU
)

raw_se = standard_error(
    ACC_RAW
)

# ============================================================
# PUBLICATION ACCURACY FIGURE
# ============================================================

print(
    "\n[6/10] Saving accuracy figure..."
)


fig, ax = plt.subplots(
    figsize=(7.4,4.9)
)


ax.errorbar(
    BETA_N,
    op_mean,
    yerr=op_se,
    marker="o",
    markersize=5,
    linewidth=2.2,
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
    capsize=3,
    label=r"Eu-DAM on $L$"
)


ax.errorbar(
    BETA_N,
    raw_mean,
    yerr=raw_se,
    marker="^",
    markersize=5,
    linewidth=2.0,
    capsize=3,
    label="Eu-DAM on original coordinates"
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
    "Protein Ensemble Retrieval"
)

ax.legend(
    frameon=True
)

ax.grid(
    alpha=.22
)

fig.tight_layout()


fig.savefig(
    os.path.join(
        OUTDIR,
        "protein_retrieval_accuracy.pdf"
    ),
    bbox_inches="tight"
)


fig.savefig(
    os.path.join(
        OUTDIR,
        "protein_retrieval_accuracy.png"
    ),
    dpi=600,
    bbox_inches="tight"
)


plt.show()

# ============================================================
# SELECT ILLUSTRATIVE CASE
#
# Prefer:
#     Op correct
#     Eu-L wrong
#     Raw Eu wrong
#
# But never fabricate this.
# ============================================================

print(
    "\n[7/10] Selecting illustrative retrieval..."
)


choice = None
best_score = -1


for trial in range(
    N_TRIALS
):

    candidate = np.where(
        (
            hard_op_pred[trial]
            == target
        )
        &
        (
            hard_eu_pred[trial]
            != target
        )
        &
        (
            hard_raw_pred[trial]
            != target
        )
    )[0]


    for i in candidate:

        structural_score = (
            A_MEM[i].sum()
        )

        if structural_score > best_score:

            best_score = structural_score

            choice = (
                trial,
                int(i)
            )


# fallback:
# Op correct / Eu-L wrong
if choice is None:

    for trial in range(
        N_TRIALS
    ):

        candidate = np.where(
            (
                hard_op_pred[trial]
                == target
            )
            &
            (
                hard_eu_pred[trial]
                != target
            )
        )[0]

        if len(candidate) > 0:

            choice = (
                trial,
                int(candidate[0])
            )

            break


# final honest fallback
if choice is None:

    choice = (
        0,
        0
    )


trial_vis, true_idx = choice


op_idx = int(
    hard_op_pred[
        trial_vis
    ][true_idx]
)


eu_idx = int(
    hard_eu_pred[
        trial_vis
    ][true_idx]
)


raw_idx = int(
    hard_raw_pred[
        trial_vis
    ][true_idx]
)


print(
    "True       :",
    protein_ids[true_idx]
)

print(
    "Graph-DAM  :",
    protein_ids[op_idx]
)

print(
    "Eu-DAM on L:",
    protein_ids[eu_idx]
)

print(
    "Raw Eu-DAM :",
    protein_ids[raw_idx]
)


print(
    "Held-out model:",
    QUERY_MODEL_INDEX[
        trial_vis
    ][true_idx]
)


# ============================================================
# OUTPUT LIST
# ============================================================

show_idx = [
    true_idx,
    op_idx,
    eu_idx,
    raw_idx
]


show_names = [
    "true_memory",
    "op_dam",
    "eu_dam_L",
    "raw_eu_dam"
]


def status(idx):

    return (
        "Correct"
        if idx == true_idx
        else "Incorrect"
    )


show_titles = [

    (
        "True Memory\n"
        +
        protein_ids[true_idx]
    ),

    (
        "Graph-DAM\n"
        +
        protein_ids[op_idx]
        +
        f" ({status(op_idx)})"
    ),

    (
        r"Eu-DAM on $L$"
        +
        "\n"
        +
        protein_ids[eu_idx]
        +
        f" ({status(eu_idx)})"
    ),

    (
        "Raw Eu-DAM\n"
        +
        protein_ids[raw_idx]
        +
        f" ({status(raw_idx)})"
    )
]

# ============================================================
# 8. SAVE EACH AGGREGATED CONTACT GRAPH INDEPENDENTLY
# ============================================================

print(
    "\n[8/10] Saving contact-network memories..."
)


VMAX = max(
    A_MEM[idx].max()
    for idx in show_idx
)


for idx, name, title in zip(
    show_idx,
    show_names,
    show_titles
):

    fig, ax = plt.subplots(
        figsize=(5.2,4.8)
    )


    im = ax.imshow(
        A_MEM[idx],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        vmin=0,
        vmax=VMAX
    )


    ax.set_xlabel(
        "Residue index"
    )

    ax.set_ylabel(
        "Residue index"
    )

    ax.set_title(
        title,
        pad=8
    )


    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=.046,
        pad=.04
    )


    cb.set_label(
        "Mean contact weight"
    )


    fig.tight_layout()


    fig.savefig(
        os.path.join(
            OUTDIR,
            name + "_contact_graph.pdf"
        ),
        bbox_inches="tight"
    )


    fig.savefig(
        os.path.join(
            OUTDIR,
            name + "_contact_graph.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )


    plt.show()


    plt.close(
        fig
    )

# ============================================================
# 9. SAVE ACTUAL REPRESENTATIVE REAL 3D PROTEINS
#
# For visualization, use model 1 of each retrieved protein,
# NOT the coordinate average.
#
# This therefore remains an actual experimental structure.
# ============================================================

print(
    "\n[9/10] Saving actual real 3D structures..."
)


XYZ_SHOW = []


for idx in show_idx:

    # genuine first experimental conformation
    X = ENSEMBLES[
        idx
    ][0].copy()

    X = center_structure(
        X
    )

    XYZ_SHOW.append(
        X
    )


GLOBAL_RADIUS = max(
    np.max(
        np.linalg.norm(
            X,
            axis=1
        )
    )
    for X in XYZ_SHOW
)


GLOBAL_RADIUS *= 1.08


for X, name, title in zip(
    XYZ_SHOW,
    show_names,
    show_titles
):

    fig = plt.figure(
        figsize=(5.4,5.1)
    )


    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    ax.plot(
        X[:,0],
        X[:,1],
        X[:,2],
        linewidth=2.2,
        alpha=.92
    )


    ax.scatter(
        X[:,0],
        X[:,1],
        X[:,2],
        s=15,
        depthshade=True
    )


    ax.scatter(
        X[0,0],
        X[0,1],
        X[0,2],
        s=65,
        marker="o",
        label="N terminus"
    )


    ax.scatter(
        X[-1,0],
        X[-1,1],
        X[-1,2],
        s=65,
        marker="^",
        label="C terminus"
    )


    ax.set_xlim(
        -GLOBAL_RADIUS,
        GLOBAL_RADIUS
    )

    ax.set_ylim(
        -GLOBAL_RADIUS,
        GLOBAL_RADIUS
    )

    ax.set_zlim(
        -GLOBAL_RADIUS,
        GLOBAL_RADIUS
    )


    ax.set_box_aspect(
        (1,1,1)
    )


    ax.view_init(
        elev=20,
        azim=-55
    )


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


    ax.set_title(
        title,
        pad=10
    )


    ax.legend(
        frameon=False,
        fontsize=8
    )


    fig.tight_layout()


    fig.savefig(
        os.path.join(
            OUTDIR,
            name + "_protein_3d.pdf"
        ),
        bbox_inches="tight"
    )


    fig.savefig(
        os.path.join(
            OUTDIR,
            name + "_protein_3d.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )


    plt.show()


    plt.close(
        fig
    )

# ============================================================
# 10. SUMMARY
# ============================================================

print(
    "\n[10/10] SUMMARY"
)

print(
    "=" * 88
)

print(
    f"Number of identities      : {N}"
)

print(
    f"Residues                  : {N_RES}"
)

print(
    f"Memory conformations      : {M_MEM}"
)

print(
    f"Contact radius            : {CONTACT_RADIUS} A"
)

print(
    f"Query edge masking        : {100*MASK_RATE:.0f}%"
)

print()

print(
    "Graph memory:"
)

print(
    "  mean Laplacian across real NMR conformations"
)

print()

print(
    "Raw memory:"
)

print(
    "  mean coordinate matrix across the SAME conformations"
)

print()

print(
    "Query:"
)

print(
    "  completely held-out real NMR conformation"
)

print(
    "  + 20% masking of existing non-backbone contact edges"
)

print(
    "  full residue set and backbone connectivity retained"
)

print()

print(
    "Graph methods:"
)

print(
    "  same L memory + same L query"
)

print(
    "  Graph-DAM (operator norm) versus vectorized Eu-DAM"
)

print()

print(
    "Raw baseline:"
)

print(
    "  same original coordinate observations"
)

print(
    "  represented directly as vec(X)"
)

print()

print(
    f"True       : {protein_ids[true_idx]}"
)

print(
    f"Graph-DAM  : {protein_ids[op_idx]}"
)

print(
    f"Eu-DAM on L: {protein_ids[eu_idx]}"
)

print(
    f"Raw Eu-DAM : {protein_ids[raw_idx]}"
)

print()

print(
    "Saved to:"
)

print(
    OUTDIR
)