import os, urllib.request, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm.auto import tqdm


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

OUTDIR = os.path.join("outputs", "airline")
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MASK_RATE = 0.20

MIN_FLIGHTS_PER_HALF = 1200

N_AIRPORTS = 45

RAW_ROWS = 300

N_TRIALS = 5

N_STEPS = 5

BETA_N = np.logspace(
    -1,
    3,
    25
)

EPS = 1e-10

print("=" * 80)
print("AIRLINE NETWORK RETRIEVAL ACROSS TIME")
print("=" * 80)

# ============================================================
# 1. LOAD REAL FLIGHT DATA
# ============================================================

print("\n[1/10] Loading nycflights13...")

flights_url = (
    "https://vincentarelbundock.github.io/"
    "Rdatasets/csv/nycflights13/flights.csv"
)

airports_url = (
    "https://vincentarelbundock.github.io/"
    "Rdatasets/csv/nycflights13/airports.csv"
)

flights_path = "/tmp/nycflights13_flights.csv"
airports_path = "/tmp/nycflights13_airports.csv"

if not os.path.exists(flights_path):
    urllib.request.urlretrieve(
        flights_url,
        flights_path
    )

if not os.path.exists(airports_path):
    urllib.request.urlretrieve(
        airports_url,
        airports_path
    )

df = pd.read_csv(flights_path)

airport_df = pd.read_csv(airports_path)

df = df[
    [
        "month",
        "day",
        "carrier",
        "origin",
        "dest",
        "distance",
        "air_time",
        "dep_delay",
        "arr_delay"
    ]
].copy()

df = df[
    df["carrier"].notna()
    &
    df["origin"].notna()
    &
    df["dest"].notna()
].reset_index(drop=True)

print(
    "Individual flight rows:",
    len(df)
)

# ============================================================
# 2. TEMPORAL SPLIT
# ============================================================

MEM_DF = df[
    df["month"] <= 6
].copy()

QUE_DF = df[
    df["month"] >= 7
].copy()

mem_counts = (
    MEM_DF
    .groupby("carrier")
    .size()
)

que_counts = (
    QUE_DF
    .groupby("carrier")
    .size()
)

carriers = sorted(
    list(
        set(
            mem_counts[
                mem_counts >= MIN_FLIGHTS_PER_HALF
            ].index
        )
        &
        set(
            que_counts[
                que_counts >= MIN_FLIGHTS_PER_HALF
            ].index
        )
    )
)

N = len(carriers)

target = np.arange(N)

print(
    "\n[2/10] Carrier memories:",
    N
)

print(
    "Carriers:",
    carriers
)

# ============================================================
# 3. FIX ONE COMMON AIRPORT SYSTEM
# ============================================================

airport_freq = pd.concat(
    [
        df["origin"],
        df["dest"]
    ]
).value_counts()

airports = (
    airport_freq
    .head(N_AIRPORTS)
    .index
    .tolist()
)

airport_set = set(airports)

airport_to_idx = {
    a: i
    for i, a in enumerate(airports)
}

MEM_DF = MEM_DF[
    MEM_DF["origin"].isin(airport_set)
    &
    MEM_DF["dest"].isin(airport_set)
].copy()

QUE_DF = QUE_DF[
    QUE_DF["origin"].isin(airport_set)
    &
    QUE_DF["dest"].isin(airport_set)
].copy()

# ============================================================
# AIRPORT COORDINATES
# ============================================================

airport_coords = {}

for _, row in airport_df.iterrows():

    faa = row["faa"]

    if faa in airport_set:

        airport_coords[faa] = (
            float(row["lon"]),
            float(row["lat"])
        )

# safety fallback
for k, a in enumerate(airports):

    if a not in airport_coords:

        angle = (
            2*np.pi*k
            /
            len(airports)
        )

        airport_coords[a] = (
            -95 + 20*np.cos(angle),
            37 + 10*np.sin(angle)
        )

# ============================================================
# 4. GRAPH / LAPLACIAN CONSTRUCTION
# ============================================================

def adjacency_from_flights(frame):

    A = np.zeros(
        (
            N_AIRPORTS,
            N_AIRPORTS
        ),
        dtype=np.float64
    )

    for o, d in zip(
        frame["origin"],
        frame["dest"]
    ):

        if (
            o not in airport_to_idx
            or
            d not in airport_to_idx
        ):
            continue

        i = airport_to_idx[o]
        j = airport_to_idx[d]

        if i == j:
            continue

        A[i,j] += 1.0
        A[j,i] += 1.0

    return A


def laplacian_from_adjacency(A):

    degree = A.sum(axis=1)

    return (
        np.diag(degree)
        -
        A
    )


MEM_FRAMES = []
QUERY_FRAMES = []

for carrier in carriers:

    M = (
        MEM_DF[
            MEM_DF["carrier"] == carrier
        ]
        .copy()
        .reset_index(drop=True)
    )

    Q = (
        QUE_DF[
            QUE_DF["carrier"] == carrier
        ]
        .copy()
        .reset_index(drop=True)
    )

    MEM_FRAMES.append(M)
    QUERY_FRAMES.append(Q)

print(
    "\n[3/10] Building memory graphs..."
)

A_MEM = np.stack([
    adjacency_from_flights(M)
    for M in MEM_FRAMES
])

L_MEM = np.stack([
    laplacian_from_adjacency(A)
    for A in A_MEM
])

L_DIM = (
    N_AIRPORTS
    *
    N_AIRPORTS
)

L_FLAT = L_MEM.reshape(
    N,
    L_DIM
)

# ============================================================
# 5. RAW-DATA BASELINE
# ============================================================

CONT_COLS = [
    "distance",
    "air_time",
    "dep_delay",
    "arr_delay"
]

cont_mean = (
    df[CONT_COLS]
    .mean()
    .fillna(0)
    .to_numpy(dtype=float)
)

cont_std = (
    df[CONT_COLS]
    .std()
    .replace(0,1)
    .fillna(1)
    .to_numpy(dtype=float)
)

RAW_ROW_DIM = (
    2*N_AIRPORTS
    +
    len(CONT_COLS)
)

RAW_DIM = (
    RAW_ROWS
    *
    RAW_ROW_DIM
)


def encode_raw_rows(
    frame,
    seed
):

    rr = np.random.default_rng(seed)

    if len(frame) >= RAW_ROWS:

        idx = rr.choice(
            len(frame),
            size=RAW_ROWS,
            replace=False
        )

    else:

        idx = rr.choice(
            len(frame),
            size=RAW_ROWS,
            replace=True
        )

    S = (
        frame.iloc[idx]
        .reset_index(drop=True)
    )

    X = np.zeros(
        (
            RAW_ROWS,
            RAW_ROW_DIM
        ),
        dtype=np.float64
    )

    for r, row in S.iterrows():

        o = row["origin"]
        d = row["dest"]

        if o in airport_to_idx:

            X[
                r,
                airport_to_idx[o]
            ] = 1.0

        if d in airport_to_idx:

            X[
                r,
                N_AIRPORTS
                +
                airport_to_idx[d]
            ] = 1.0

        vals = (
            row[CONT_COLS]
            .astype(float)
            .to_numpy()
        )

        vals = np.where(
            np.isfinite(vals),
            vals,
            cont_mean
        )

        vals = (
            vals
            -
            cont_mean
        ) / cont_std

        X[
            r,
            2*N_AIRPORTS:
        ] = vals

    return X


RAW_MEM_MATRIX = np.stack([
    encode_raw_rows(
        MEM_FRAMES[i],
        SEED + 100*i
    )
    for i in range(N)
])

RAW_MEM = RAW_MEM_MATRIX.reshape(
    N,
    RAW_DIM
)

# ============================================================
# 6. HELD-OUT + MASKED QUERIES
# ============================================================

print(
    f"\n[4/10] Building Jul-Dec queries + "
    f"{int(100*MASK_RATE)}% masking..."
)

Q_L = []
Q_RAW = []

for trial in range(
    N_TRIALS
):

    rr = np.random.default_rng(
        SEED
        +
        10000
        +
        trial
    )

    trial_L = []
    trial_raw = []

    for i, Qfull in enumerate(
        QUERY_FRAMES
    ):

        # --------------------------------------------
        # GRAPH QUERY
        # --------------------------------------------

        n = len(Qfull)

        n_mask = max(
            1,
            int(
                round(
                    MASK_RATE*n
                )
            )
        )

        drop = rr.choice(
            n,
            size=n_mask,
            replace=False
        )

        keep = np.ones(
            n,
            dtype=bool
        )

        keep[drop] = False

        Qmasked = (
            Qfull
            .iloc[
                np.where(keep)[0]
            ]
            .reset_index(drop=True)
        )

        Aq = adjacency_from_flights(
            Qmasked
        )

        trial_L.append(
            laplacian_from_adjacency(
                Aq
            )
        )

        # --------------------------------------------
        # RAW QUERY
        # --------------------------------------------

        Xraw = encode_raw_rows(
            Qfull,
            SEED
            +
            500000
            +
            1000*trial
            +
            i
        )

        m = max(
            1,
            int(
                round(
                    MASK_RATE
                    *
                    RAW_ROWS
                )
            )
        )

        drop_raw = rr.choice(
            RAW_ROWS,
            size=m,
            replace=False
        )

        Xraw[
            drop_raw,
            :
        ] = 0.0

        trial_raw.append(
            Xraw
        )

    Q_L.append(
        np.stack(
            trial_L
        )
    )

    Q_RAW.append(
        np.stack(
            trial_raw
        )
    )

# ============================================================
# 7. DISTANCES / SCORES
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
            Q[b][None,:,:]
            -
            L_MEM
        )

        eig = np.linalg.eigvalsh(E)

        op = np.max(
            np.abs(eig),
            axis=1
        )

        out[b] = (
            op*op
        )

    return out


def graph_score_flat(QF):

    return (
        QF
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


def raw_score_flat(QF):

    return (
        QF
        @
        RAW_MEM.T
    )


def raw_score(Q):

    return raw_score_flat(
        Q.reshape(
            len(Q),
            RAW_DIM
        )
    )

# ============================================================
# 8. TEMPERATURE SCALES
# ============================================================

print(
    "\n[5/10] Computing inverse-temperature scales..."
)

op_pair = []

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

        eig = np.linalg.eigvalsh(E)

        d = np.max(
            np.abs(eig)
        )

        op_pair.append(
            d*d
        )

SCALE_OP = max(
    np.median(op_pair),
    EPS
)

S_LL = (
    L_FLAT
    @
    L_FLAT.T
)

SCALE_GRAPH_EU = max(
    np.median(
        np.std(
            S_LL,
            axis=1
        )
    ),
    EPS
)

S_RAW = (
    RAW_MEM
    @
    RAW_MEM.T
)

SCALE_RAW_EU = max(
    np.median(
        np.std(
            S_RAW,
            axis=1
        )
    ),
    EPS
)

# ============================================================
# HARD PREDICTIONS
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
                Q_RAW[trial]
            ),
            axis=1
        )
    )

# ============================================================
# SOFTMAX
# ============================================================

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

# ============================================================
# RETRIEVAL FUNCTIONS
# ============================================================

def op_retrieve(
    Q0,
    beta_n
):

    Q = Q0.copy()

    previous = None

    for _ in range(
        N_STEPS
    ):

        d2 = (
            op_d2(Q)
            /
            SCALE_OP
        )

        W = softmax_rows(
            -beta_n*d2
        )

        Q = (
            W
            @
            L_FLAT
        ).reshape(
            len(Q),
            N_AIRPORTS,
            N_AIRPORTS
        )

        pred = np.argmin(
            op_d2(Q),
            axis=1
        )

        if (
            previous is not None
            and
            np.array_equal(
                pred,
                previous
            )
        ):
            break

        previous = pred

    return pred


def graph_eu_retrieve(
    Q0,
    beta_n
):

    QF = Q0.reshape(
        len(Q0),
        L_DIM
    ).copy()

    previous = None

    for _ in range(
        N_STEPS
    ):

        score = (
            graph_score_flat(
                QF
            )
            /
            SCALE_GRAPH_EU
        )

        W = softmax_rows(
            beta_n*score
        )

        QF = (
            W
            @
            L_FLAT
        )

        pred = np.argmax(
            graph_score_flat(
                QF
            ),
            axis=1
        )

        if (
            previous is not None
            and
            np.array_equal(
                pred,
                previous
            )
        ):
            break

        previous = pred

    return pred


def raw_eu_retrieve(
    Q0,
    beta_n
):

    QF = Q0.reshape(
        len(Q0),
        RAW_DIM
    ).copy()

    previous = None

    for _ in range(
        N_STEPS
    ):

        score = (
            raw_score_flat(
                QF
            )
            /
            SCALE_RAW_EU
        )

        W = softmax_rows(
            beta_n*score
        )

        QF = (
            W
            @
            RAW_MEM
        )

        pred = np.argmax(
            raw_score_flat(
                QF
            ),
            axis=1
        )

        if (
            previous is not None
            and
            np.array_equal(
                pred,
                previous
            )
        ):
            break

        previous = pred

    return pred

# ============================================================
# 9. TEMPERATURE SWEEP
# ============================================================

print(
    "\n[6/10] Running inverse-temperature sweep..."
)

ACC_OP = np.zeros(
    (
        N_TRIALS,
        len(BETA_N)
    )
)

ACC_GE = np.zeros_like(
    ACC_OP
)

ACC_RAW = np.zeros_like(
    ACC_OP
)

progress = tqdm(
    total=
        N_TRIALS
        *
        len(BETA_N)
        *
        3,
    desc="DAM sweep",
    unit="run"
)

start = time.time()

for trial in range(
    N_TRIALS
):

    for j, beta_n in enumerate(
        BETA_N
    ):

        pred = op_retrieve(
            Q_L[trial],
            beta_n
        )

        ACC_OP[
            trial,
            j
        ] = np.mean(
            pred == target
        )

        progress.update(1)

        pred = graph_eu_retrieve(
            Q_L[trial],
            beta_n
        )

        ACC_GE[
            trial,
            j
        ] = np.mean(
            pred == target
        )

        progress.update(1)

        pred = raw_eu_retrieve(
            Q_RAW[trial],
            beta_n
        )

        ACC_RAW[
            trial,
            j
        ] = np.mean(
            pred == target
        )

        progress.update(1)

progress.close()

print(
    f"Sweep completed in "
    f"{time.time()-start:.1f}s"
)

# ============================================================
# MEAN + STANDARD ERROR
# ============================================================

op_mean = ACC_OP.mean(axis=0)
ge_mean = ACC_GE.mean(axis=0)
raw_mean = ACC_RAW.mean(axis=0)

op_se = (
    ACC_OP.std(
        axis=0,
        ddof=1
    )
    /
    np.sqrt(N_TRIALS)
)

ge_se = (
    ACC_GE.std(
        axis=0,
        ddof=1
    )
    /
    np.sqrt(N_TRIALS)
)

raw_se = (
    ACC_RAW.std(
        axis=0,
        ddof=1
    )
    /
    np.sqrt(N_TRIALS)
)

# ============================================================
# FIGURE 1:
# RETRIEVAL ACCURACY + ERROR BARS
# ============================================================

print(
    "\n[7/10] Plotting accuracy..."
)

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

ax.errorbar(
    BETA_N,
    op_mean,
    yerr=op_se,
    marker="o",
    markersize=5,
    linewidth=2.2,
    elinewidth=1.0,
    capsize=3,
    label="Graph-DAM"
)

ax.errorbar(
    BETA_N,
    ge_mean,
    yerr=ge_se,
    marker="s",
    markersize=5,
    linewidth=2.0,
    elinewidth=1.0,
    capsize=3,
    label=r"Eu-DAM"
)

ax.errorbar(
    BETA_N,
    raw_mean,
    yerr=raw_se,
    marker="^",
    markersize=5,
    linewidth=2.0,
    elinewidth=1.0,
    capsize=3,
    label="Raw Eu-DAM"
)

ax.set_xscale("log")

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
    "Airline Network Retrieval Across Time"
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
        "airline_retrieval_accuracy.pdf"
    ),
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTDIR,
        "airline_retrieval_accuracy.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()


print(
    "\n[8/10] Selecting illustrative retrieval..."
)

best_choice = None
best_score = -np.inf

for trial in range(
    N_TRIALS
):

    op_pred = hard_op_pred[trial]
    eu_pred = hard_eu_pred[trial]
    raw_pred = hard_raw_pred[trial]

    candidate = np.where(
        (op_pred == target)
        &
        (eu_pred != target)
        &
        (raw_pred != target)
    )[0]

    for i in candidate:

        A = A_MEM[i]

        degree = A.sum(axis=1)

        n_active = np.sum(
            degree > 0
        )

        n_edges = np.sum(
            np.triu(
                A > 0,
                1
            )
        )

        # favor visually rich graphs
        # while avoiding trivial 3-4 node examples
        visual_score = (
            4*n_active
            +
            n_edges
        )

        if (
            n_active >= 10
            and
            n_edges >= 10
            and
            visual_score > best_score
        ):

            best_score = visual_score

            best_choice = (
                trial,
                int(i)
            )

# fallback: any triple-disagreement case
if best_choice is None:

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

        if len(candidate) > 0:

            best_choice = (
                trial,
                int(candidate[0])
            )

            break

if best_choice is None:

    raise RuntimeError(
        "No trial/example found where Op-DAM is correct "
        "and both Eu baselines are incorrect."
    )

trial_vis, true_idx = best_choice

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
    "True memory       :",
    carriers[true_idx]
)

print(
    "Graph-DAM retrieval  :",
    carriers[op_idx]
)

print(
    "Eu-DAM       :",
    carriers[eu_idx]
)

print(
    "Raw Eu-DAM        :",
    carriers[raw_idx]
)

# ============================================================
# FOUR GRAPHS TO DISPLAY
# ============================================================


A_SHOW = [
    A_MEM[true_idx],
    A_MEM[op_idx],
    A_MEM[eu_idx],
    A_MEM[raw_idx]
]

titles = [
    f"True Memory: {carriers[true_idx]}",
    f"Graph-DAM: {carriers[op_idx]}\n(Correct)",
    rf"Eu-DAM: {carriers[eu_idx]}" + "\n(Incorrect)",
    f"Raw Eu-DAM: {carriers[raw_idx]}\n(Incorrect)"
]

# ------------------------------------------------------------
# COMMON SCALE ACROSS ALL FOUR PANELS
# ------------------------------------------------------------

MAX_DEG = max(
    A.sum(axis=1).max()
    for A in A_SHOW
)

MAX_EDGE = max(
    A.max()
    for A in A_SHOW
)

# ------------------------------------------------------------
# Coordinates
# ------------------------------------------------------------

lon = np.array([
    airport_coords[a][0]
    for a in airports
])

lat = np.array([
    airport_coords[a][1]
    for a in airports
])

# ------------------------------------------------------------
# ACTIVE AIRPORT UNION
# ------------------------------------------------------------

ACTIVE_UNION = np.zeros(
    N_AIRPORTS,
    dtype=bool
)

for A in A_SHOW:
    ACTIVE_UNION |= (
        A.sum(axis=1) > 0
    )

LABEL_IDX = np.where(
    ACTIVE_UNION
)[0]


# TIGHT COMMON GEOGRAPHIC WINDOW


active_lon = lon[
    ACTIVE_UNION
]

active_lat = lat[
    ACTIVE_UNION
]

xrange = (
    active_lon.max()
    -
    active_lon.min()
)

yrange = (
    active_lat.max()
    -
    active_lat.min()
)

xpad = max(
    1.2,
    0.05*xrange
)

ypad = max(
    0.8,
    0.08*yrange
)

GLOBAL_XLIM = (
    active_lon.min()-xpad,
    active_lon.max()+xpad
)

GLOBAL_YLIM = (
    active_lat.min()-ypad,
    active_lat.max()+ypad
)


# LABEL OFFSET HELPER


def label_offset(idx):

    # alternating directions
    patterns = [
        (4, 4),
        (4, -8),
        (-18, 4),
        (-18, -8),
        (7, 1),
        (-22, 1)
    ]

    return patterns[
        idx % len(patterns)
    ]

# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_memory_graph(
    ax,
    A,
    title
):

    degree = A.sum(
        axis=1
    )

    active = (
        degree > 0
    )

    # --------------------------------------------------------
    # faint locations for all airports in the displayed union
    # --------------------------------------------------------

    ax.scatter(
        lon[ACTIVE_UNION],
        lat[ACTIVE_UNION],
        s=12,
        facecolors="0.88",
        edgecolors="none",
        zorder=1
    )

    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    segments = []
    widths = []

    for i in range(
        N_AIRPORTS
    ):

        for j in range(
            i+1,
            N_AIRPORTS
        ):

            w = A[i,j]

            if w <= 0:
                continue

            segments.append(
                [
                    (lon[i], lat[i]),
                    (lon[j], lat[j])
                ]
            )

            width = (
                0.35
                +
                3.0
                *
                np.log1p(w)
                /
                np.log1p(
                    max(
                        MAX_EDGE,
                        1
                    )
                )
            )

            widths.append(
                width
            )

    if len(segments) > 0:

        lc = LineCollection(
            segments,
            linewidths=widths,
            alpha=0.38,
            zorder=2
        )

        ax.add_collection(
            lc
        )

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    node_size = (
        22
        +
        260
        *
        np.sqrt(
            degree
            /
            max(
                MAX_DEG,
                EPS
            )
        )
    )

    ax.scatter(
        lon[active],
        lat[active],
        s=node_size[active],
        edgecolors="white",
        linewidths=0.65,
        zorder=3
    )

  
    # LABEL ALL ACTIVE AIRPORTS IN UNION
 

    for order, idx in enumerate(
        LABEL_IDX
    ):

        dx, dy = label_offset(
            order
        )

        is_active = (
            degree[idx] > 0
        )

        ax.annotate(
            airports[idx],
            (
                lon[idx],
                lat[idx]
            ),
            xytext=(
                dx,
                dy
            ),
            textcoords="offset points",
            fontsize=7.2,
            color=(
                "black"
                if is_active
                else "0.55"
            ),
            alpha=(
                1.0
                if is_active
                else 0.65
            ),
            zorder=4
        )

    # --------------------------------------------------------
    # SAME TIGHT LIMITS
    # --------------------------------------------------------

    ax.set_xlim(
        *GLOBAL_XLIM
    )

    ax.set_ylim(
        *GLOBAL_YLIM
    )

    ax.set_aspect(
        "auto"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        title,
        fontsize=12.5,
        weight="bold",
        pad=7
    )

    for spine in ax.spines.values():

        spine.set_linewidth(
            0.7
        )

        spine.set_alpha(
            0.4
        )

# ============================================================
# FOUR-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16.8, 5.2)
)

for ax, A, title in zip(
    axes,
    A_SHOW,
    titles
):

    plot_memory_graph(
        ax,
        A,
        title
    )

fig.suptitle(
    "Airline Network Retrieval Across Time",
    fontsize=16,
    y=0.975
)

fig.subplots_adjust(
    left=0.015,
    right=0.995,
    bottom=0.045,
    top=0.84,
    wspace=0.045
)

graphs_pdf = os.path.join(
    OUTDIR,
    "airline_retrieved_networks.pdf"
)

graphs_png = os.path.join(
    OUTDIR,
    "airline_retrieved_networks.png"
)

fig.savefig(
    graphs_pdf,
    bbox_inches="tight"
)

fig.savefig(
    graphs_png,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# SAVE EACH NETWORK INDIVIDUALLY
# ============================================================

individual_names = [
    "true_memory",
    "op_dam_retrieval",
    "eu_dam_L_retrieval",
    "raw_eu_dam_retrieval"
]

for A, title, fname in zip(
    A_SHOW,
    titles,
    individual_names
):

    fig, ax = plt.subplots(
        figsize=(5.3,4.5)
    )

    plot_memory_graph(
        ax,
        A,
        title
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTDIR,
            fname + ".pdf"
        ),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            OUTDIR,
            fname + ".png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n[10/10] RESULTS"
)

print(
    "="*80
)

for name, mean_, se_ in [

    (
        "Graph-DAM",
        op_mean,
        op_se
    ),

    (
        "Eu-DAM",
        ge_mean,
        ge_se
    ),

    (
        "Raw Eu-DAM",
        raw_mean,
        raw_se
    )
]:

    k = np.argmax(
        mean_
    )

    print(
        f"{name:14s}: "
        f"{mean_[k]:.3f} +/- "
        f"{se_[k]:.3f} SE, "
        f"beta_n={BETA_N[k]:.4g}"
    )

print(
    "\nIllustrative retrieval:"
)

print(
    f"  True       : {carriers[true_idx]}"
)

print(
    f"  Graph-DAM     : {carriers[op_idx]}  <-- correct"
)

print(
    f"  Eu-DAM: {carriers[eu_idx]}  <-- incorrect"
)

print(
    f"  Raw Eu-DAM : {carriers[raw_idx]}  <-- incorrect"
)

print(
    "\nPublication files saved to:"
)

print(
    OUTDIR
)

print(
    "\nMain figures:"
)

print(
    "  airline_retrieval_accuracy.pdf"
)

print(
    "  airline_retrieval_accuracy.png"
)

print(
    "  airline_retrieved_networks.pdf"
)

print(
    "  airline_retrieved_networks.png"
)