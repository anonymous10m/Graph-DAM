# ================================================================
# UCI RAW HAR × GRAPH-DAM
# PAPER/REPOSITORY-ALIGNED + FAIR NON-GRAPH RAW BASELINE
#
# Graph side:
#   - real UCI HAR activity interval
#   - 60-node temporal-sensor graph
#   - ordinary combinatorial Laplacian L = D - A
#   - NO graph-mass normalization
#   - query = same realized graph with 20% EXISTING edges removed
#
# Eu-DAM:
#   - EXACT same corrupted Laplacian query
#   - classical Euclidean LSE on vec(L)
#
# Raw Eu-DAM:
#   - NON-GRAPH sensor summary features
#   - NOT the complete resampled trajectory
#   - features: time-domain + spectral summaries from 6 raw channels
#   - query uses a corrupted version of the same underlying interval
#   - 20% temporal samples are removed/interpolated before features
#
# Raw features per channel:
#   mean, std, RMS, median, IQR, MAD,
#   mean |dx|, std(dx), zero-crossing rate,
#   spectral centroid, spectral entropy,
#   dominant frequency,
#   low/mid/high spectral energy
#
# Additional magnitude features for accelerometer and gyroscope.
#
# DAM mechanics:
#   - beta = 25 log-spaced values from 1e-1 to 1e3
#   - 5 iterations
#   - 5 independent corruption runs
#   - repository-style temperature scales
#
# ================================================================

import os
import glob
import shutil
import zipfile
import urllib.request
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ================================================================
# SETTINGS
# ================================================================

SEED = 42
EPS = 1e-12

N_MEM = 30
N_RUNS = 5
N_STEPS = 5

BETAS = np.logspace(-1, 3, 25)

# ------------------------------------------------
# Graph construction
# ------------------------------------------------

N_BLOCKS = 10
N_AXES = 6
N_NODES = N_BLOCKS * N_AXES

KNN = 10

# ------------------------------------------------
# Graph corruption
# ------------------------------------------------

EDGE_REMOVE_FRAC = 0.20

# ------------------------------------------------
# Raw query corruption
# ------------------------------------------------

RAW_SAMPLE_REMOVE_FRAC = 0.20

# ------------------------------------------------
# Minimum interval length
# ------------------------------------------------

MIN_SAMPLES = 300

rng = np.random.default_rng(SEED)

print("=" * 92)
print("UCI RAW HAR × GRAPH-DAM — FAIR NON-GRAPH RAW BASELINE")
print("=" * 92)

# ================================================================
# 1. DOWNLOAD
# ================================================================

print("\n[1/10] Downloading UCI HAR raw data...")

URLS = [
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",

    "https://archive.ics.uci.edu/ml/machine-learning-databases/00341/"
    "UCI%20HAR%20Dataset.zip",
]

ZIP_PATH = "/tmp/uci_raw_har.zip"
ROOT = "/tmp/uci_raw_har"


def valid_zip(path):

    try:

        return (
            os.path.exists(path)
            and
            os.path.getsize(path) > 1_000_000
            and
            zipfile.is_zipfile(path)
        )

    except Exception:

        return False


if not valid_zip(ZIP_PATH):

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    success = False

    for url in URLS:

        try:

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            with urllib.request.urlopen(
                req,
                timeout=120
            ) as response:

                data = response.read()

            with open(
                ZIP_PATH,
                "wb"
            ) as f:

                f.write(data)

            if valid_zip(ZIP_PATH):

                success = True
                break

        except Exception:

            pass

    if not success:

        raise RuntimeError(
            "Could not download UCI HAR."
        )


if os.path.exists(ROOT):

    shutil.rmtree(ROOT)


os.makedirs(
    ROOT,
    exist_ok=True
)


with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as z:

    z.extractall(ROOT)


# ------------------------------------------------
# Extract nested archives
# ------------------------------------------------

for _ in range(3):

    nested = []

    for root, _, files in os.walk(ROOT):

        for f in files:

            if f.lower().endswith(".zip"):

                nested.append(
                    os.path.join(
                        root,
                        f
                    )
                )

    changed = False

    for zp in nested:

        dest = (
            zp
            +
            "_extract"
        )

        if os.path.exists(dest):
            continue

        try:

            os.makedirs(
                dest,
                exist_ok=True
            )

            with zipfile.ZipFile(
                zp,
                "r"
            ) as z:

                z.extractall(dest)

            changed = True

        except Exception:

            pass

    if not changed:
        break


def find_file(filename):

    for root, _, files in os.walk(ROOT):

        if filename in files:

            return os.path.join(
                root,
                filename
            )

    return None


LABEL_PATH = find_file(
    "labels.txt"
)


if LABEL_PATH is None:

    raise RuntimeError(
        "labels.txt not found."
    )


RAW_DIR = os.path.dirname(
    LABEL_PATH
)


print(
    "RawData directory:",
    RAW_DIR
)

# ================================================================
# 2. LOAD CONTINUOUS RECORDINGS
# ================================================================

print("\n[2/10] Loading continuous recordings...")

labels = np.loadtxt(
    LABEL_PATH,
    dtype=int
)


if labels.ndim == 1:

    labels = labels[
        None,
        :
    ]


print(
    "Label intervals:",
    len(labels)
)


recording_cache = {}


def locate_raw_file(
    prefix,
    exp,
    user
):

    names = [
        f"{prefix}_exp{exp:02d}_user{user:02d}.txt",
        f"{prefix}_exp{exp}_user{user}.txt",
    ]

    for name in names:

        path = os.path.join(
            RAW_DIR,
            name
        )

        if os.path.exists(path):

            return path

    pattern = os.path.join(
        RAW_DIR,
        f"{prefix}_exp*{exp}*_user*{user}*.txt"
    )

    hits = glob.glob(
        pattern
    )

    if hits:

        return hits[0]

    return None


def get_recording(
    exp,
    user
):

    key = (
        int(exp),
        int(user)
    )

    if key in recording_cache:

        return recording_cache[key]

    acc_path = locate_raw_file(
        "acc",
        exp,
        user
    )

    gyro_path = locate_raw_file(
        "gyro",
        exp,
        user
    )

    if (
        acc_path is None
        or
        gyro_path is None
    ):

        return None

    acc = np.loadtxt(
        acc_path,
        dtype=np.float64
    )

    gyro = np.loadtxt(
        gyro_path,
        dtype=np.float64
    )

    n = min(
        len(acc),
        len(gyro)
    )

    Z = np.concatenate(
        [
            acc[:n, :3],
            gyro[:n, :3]
        ],
        axis=1
    )

    recording_cache[
        key
    ] = Z

    return Z


# ================================================================
# 3. BASIC SIGNAL FUNCTIONS
# ================================================================

print("\n[3/10] Defining graph and raw representations...")


def robust_standardize(Z):

    Z = np.asarray(
        Z,
        dtype=np.float64
    )

    med = np.median(
        Z,
        axis=0,
        keepdims=True
    )

    mad = np.median(
        np.abs(
            Z - med
        ),
        axis=0,
        keepdims=True
    )

    scale = (
        1.4826
        *
        mad
    )

    sd = np.std(
        Z,
        axis=0,
        keepdims=True
    )

    bad = (
        scale < 1e-6
    )

    scale[
        bad
    ] = sd[
        bad
    ]

    scale[
        scale < 1e-6
    ] = 1.0

    return (
        Z - med
    ) / scale


def resample_1d(
    x,
    m
):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    old = np.linspace(
        0.0,
        1.0,
        len(x)
    )

    new = np.linspace(
        0.0,
        1.0,
        m
    )

    return np.interp(
        new,
        old,
        x
    )


# ================================================================
# 4. GRAPH REPRESENTATION
#
# 60 aligned nodes:
#     10 temporal blocks × 6 sensor axes
#
# Each node contains its local waveform.
# Edges = absolute waveform correlations.
# ================================================================

def node_signals(Z):

    Z = robust_standardize(
        Z
    )

    blocks = np.array_split(
        np.arange(
            len(Z)
        ),
        N_BLOCKS
    )

    X = []

    for ids in blocks:

        for axis in range(
            N_AXES
        ):

            x = resample_1d(
                Z[
                    ids,
                    axis
                ],
                64
            )

            x = (
                x
                -
                np.mean(x)
            )

            sd = np.std(x)

            if sd > EPS:

                x = (
                    x
                    /
                    sd
                )

            X.append(x)

    return np.stack(
        X,
        axis=0
    )


def adjacency_from_interval(Z):

    X = node_signals(
        Z
    )

    C = np.abs(
        np.corrcoef(
            X
        )
    )

    C = np.nan_to_num(
        C,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    np.fill_diagonal(
        C,
        0.0
    )

    # ------------------------------------------------------------
    # kNN sparsification
    # ------------------------------------------------------------

    A0 = np.zeros_like(
        C
    )

    for i in range(
        N_NODES
    ):

        idx = np.argpartition(
            C[i],
            -KNN
        )[
            -KNN:
        ]

        A0[
            i,
            idx
        ] = C[
            i,
            idx
        ]

    A = np.maximum(
        A0,
        A0.T
    )

    np.fill_diagonal(
        A,
        0.0
    )

    return A


# ================================================================
# ORDINARY COMBINATORIAL LAPLACIAN
#
#       L = D - A
#
# NO graph normalization.
# ================================================================

def laplacian(A):

    A = np.asarray(
        A,
        dtype=np.float64
    )

    A = (
        A
        +
        A.T
    ) / 2.0

    np.fill_diagonal(
        A,
        0.0
    )

    D = np.diag(
        A.sum(
            axis=1
        )
    )

    L = (
        D
        -
        A
    )

    return (
        L
        +
        L.T
    ) / 2.0


# ================================================================
# 5. NON-GRAPH RAW FEATURE REPRESENTATION
#
# This intentionally does NOT preserve the entire raw trajectory.
#
# It is a standard compact vector representation of sensor data.
# ================================================================

def safe_entropy(p):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    p = np.maximum(
        p,
        EPS
    )

    p = (
        p
        /
        p.sum()
    )

    H = (
        -np.sum(
            p
            *
            np.log(p)
        )
    )

    if len(p) > 1:

        H = (
            H
            /
            np.log(
                len(p)
            )
        )

    return H


def channel_features(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    if len(x) < 4:

        return np.zeros(
            15,
            dtype=np.float64
        )

    # ------------------------------------------------------------
    # Time-domain
    # ------------------------------------------------------------

    mean = np.mean(x)

    std = np.std(x)

    rms = np.sqrt(
        np.mean(
            x ** 2
        )
    )

    median = np.median(x)

    q25 = np.percentile(
        x,
        25
    )

    q75 = np.percentile(
        x,
        75
    )

    iqr = (
        q75
        -
        q25
    )

    mad = np.median(
        np.abs(
            x
            -
            median
        )
    )

    dx = np.diff(x)

    mean_abs_dx = np.mean(
        np.abs(dx)
    )

    std_dx = np.std(
        dx
    )

    xc = (
        x
        -
        mean
    )

    zcr = np.mean(
        xc[:-1]
        *
        xc[1:]
        <
        0
    )

    # ------------------------------------------------------------
    # Spectral
    # ------------------------------------------------------------

    xs = (
        x
        -
        np.mean(x)
    )

    spec = np.abs(
        np.fft.rfft(
            xs
        )
    ) ** 2

    freq = np.fft.rfftfreq(
        len(xs),
        d=1.0
    )

    if len(spec) > 1:

        spec[
            0
        ] = 0.0

    total = np.sum(
        spec
    )

    if total <= EPS:

        centroid = 0.0
        entropy = 0.0
        dom_freq = 0.0
        low = 0.0
        mid = 0.0
        high = 0.0

    else:

        p = (
            spec
            /
            total
        )

        centroid = np.sum(
            freq
            *
            p
        )

        entropy = safe_entropy(
            p
        )

        dom_freq = freq[
            np.argmax(
                spec
            )
        ]

        # normalized-frequency bands
        low_mask = (
            freq <= 0.10
        )

        mid_mask = (
            (freq > 0.10)
            &
            (freq <= 0.25)
        )

        high_mask = (
            freq > 0.25
        )

        low = np.sum(
            spec[
                low_mask
            ]
        ) / total

        mid = np.sum(
            spec[
                mid_mask
            ]
        ) / total

        high = np.sum(
            spec[
                high_mask
            ]
        ) / total

    return np.array(
        [
            mean,
            std,
            rms,
            median,
            iqr,
            mad,
            mean_abs_dx,
            std_dx,
            zcr,
            centroid,
            entropy,
            dom_freq,
            low,
            mid,
            high
        ],
        dtype=np.float64
    )


def raw_feature_vector(Z):

    Z = np.asarray(
        Z,
        dtype=np.float64
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # do NOT robust-standardize each interval before these features.
    #
    # Raw Eu-DAM should retain ordinary non-graph sensor information.
    # ------------------------------------------------------------

    feats = []

    # six original channels
    for axis in range(
        N_AXES
    ):

        feats.extend(
            channel_features(
                Z[
                    :,
                    axis
                ]
            )
        )

    # ------------------------------------------------------------
    # Accelerometer magnitude
    # ------------------------------------------------------------

    acc_mag = np.sqrt(
        np.sum(
            Z[
                :,
                0:3
            ] ** 2,
            axis=1
        )
    )

    feats.extend(
        channel_features(
            acc_mag
        )
    )

    # ------------------------------------------------------------
    # Gyroscope magnitude
    # ------------------------------------------------------------

    gyro_mag = np.sqrt(
        np.sum(
            Z[
                :,
                3:6
            ] ** 2,
            axis=1
        )
    )

    feats.extend(
        channel_features(
            gyro_mag
        )
    )

    return np.asarray(
        feats,
        dtype=np.float64
    )


# ================================================================
# RAW QUERY CORRUPTION
#
# Remove 20% of temporal observations and reconstruct by
# interpolation before extracting raw features.
# ================================================================

def corrupt_raw_signal(
    Z,
    rr,
    frac
):

    Z = np.asarray(
        Z,
        dtype=np.float64
    )

    T = len(Z)

    n_remove = int(
        np.floor(
            frac
            *
            T
        )
    )

    if n_remove <= 0:

        return Z.copy()

    # ------------------------------------------------------------
    # Keep endpoints so interpolation is well-defined.
    # ------------------------------------------------------------

    eligible = np.arange(
        1,
        T - 1
    )

    n_remove = min(
        n_remove,
        len(eligible)
    )

    removed = rr.choice(
        eligible,
        size=n_remove,
        replace=False
    )

    keep = np.ones(
        T,
        dtype=bool
    )

    keep[
        removed
    ] = False

    t = np.arange(
        T
    )

    tk = t[
        keep
    ]

    Q = np.empty_like(
        Z
    )

    for axis in range(
        N_AXES
    ):

        Q[
            :,
            axis
        ] = np.interp(
            t,
            tk,
            Z[
                keep,
                axis
            ]
        )

    return Q


# ================================================================
# 6. BUILD REAL ACTIVITY INTERVALS
# ================================================================

print("\n[4/10] Building real activity intervals...")

candidates = []


for row in labels:

    exp = int(
        row[0]
    )

    user = int(
        row[1]
    )

    activity = int(
        row[2]
    )

    start = int(
        row[3]
    )

    end = int(
        row[4]
    )

    if (
        end
        -
        start
        <
        MIN_SAMPLES
    ):

        continue

    Zall = get_recording(
        exp,
        user
    )

    if Zall is None:

        continue

    start = max(
        0,
        start
    )

    end = min(
        len(Zall),
        end
    )

    if (
        end
        -
        start
        <
        MIN_SAMPLES
    ):

        continue

    Z = Zall[
        start:end
    ].copy()

    try:

        A = adjacency_from_interval(
            Z
        )

        L = laplacian(
            A
        )

        raw = raw_feature_vector(
            Z
        )

        if (
            np.all(
                np.isfinite(A)
            )
            and
            np.all(
                np.isfinite(L)
            )
            and
            np.all(
                np.isfinite(raw)
            )
        ):

            candidates.append(
                {
                    "exp": exp,
                    "user": user,
                    "activity": activity,
                    "start": start,
                    "end": end,
                    "signal": Z,
                    "A": A,
                    "L": L,
                    "raw": raw
                }
            )

    except Exception:

        pass


print(
    "Usable intervals:",
    len(candidates)
)


if len(candidates) < N_MEM:

    raise RuntimeError(
        "Not enough usable intervals."
    )


# ================================================================
# 7. MEMORY SELECTION
#
# Same operator-separated memory selection as the previous
# high-Graph-DAM experiment.
#
# Selection uses CLEAN MEMORIES ONLY.
# No corrupted query is used.
# ================================================================

print("\n[5/10] Selecting operator-separated memories...")


CAND_L = np.stack(
    [
        c["L"]
        for c in candidates
    ]
)


def operator_distance(
    A,
    B
):

    eig = np.linalg.eigvalsh(
        A - B
    )

    return float(
        np.max(
            np.abs(
                eig
            )
        )
    )


# ------------------------------------------------------------
# deterministic starting memory
# ------------------------------------------------------------

mean_L = np.mean(
    CAND_L,
    axis=0
)


center_dist = np.array(
    [
        operator_distance(
            L,
            mean_L
        )
        for L in CAND_L
    ]
)


first = int(
    np.argmax(
        center_dist
    )
)


selected = [
    first
]


min_dist = np.array(
    [
        operator_distance(
            CAND_L[i],
            CAND_L[first]
        )
        for i in range(
            len(CAND_L)
        )
    ]
)


min_dist[
    first
] = -np.inf


while len(selected) < N_MEM:

    nxt = int(
        np.argmax(
            min_dist
        )
    )

    selected.append(
        nxt
    )

    for i in range(
        len(CAND_L)
    ):

        if min_dist[i] == -np.inf:

            continue

        d = operator_distance(
            CAND_L[i],
            CAND_L[nxt]
        )

        if d < min_dist[i]:

            min_dist[
                i
            ] = d

    min_dist[
        nxt
    ] = -np.inf


selected = np.asarray(
    selected,
    dtype=int
)


MEMORIES = [
    candidates[i]
    for i in selected
]


MEM_A = np.stack(
    [
        m["A"]
        for m in MEMORIES
    ]
)


MEM_L = np.stack(
    [
        m["L"]
        for m in MEMORIES
    ]
)


MEM_RAW_ORIGINAL = np.stack(
    [
        m["raw"]
        for m in MEMORIES
    ]
)


# ================================================================
# STANDARDIZE RAW FEATURES USING MEMORY SET
#
# This is feature-wise standardization, not per-vector normalization.
# ================================================================

RAW_MEAN = MEM_RAW_ORIGINAL.mean(
    axis=0,
    keepdims=True
)


RAW_STD = MEM_RAW_ORIGINAL.std(
    axis=0,
    keepdims=True
)


RAW_STD[
    RAW_STD < 1e-8
] = 1.0


MEM_RAW = (
    MEM_RAW_ORIGINAL
    -
    RAW_MEAN
) / RAW_STD


RAW_DIM = MEM_RAW.shape[
    1
]


L_DIM = (
    N_NODES
    *
    N_NODES
)


L_FLAT = MEM_L.reshape(
    N_MEM,
    L_DIM
)


TARGET = np.arange(
    N_MEM
)


CHANCE = (
    1.0
    /
    N_MEM
)


# ================================================================
# MEMORY SEPARATION
# ================================================================

pair_op = []


for i in range(
    N_MEM
):

    for j in range(
        i + 1,
        N_MEM
    ):

        pair_op.append(
            operator_distance(
                MEM_L[i],
                MEM_L[j]
            )
        )


pair_op = np.asarray(
    pair_op
)


print(
    "Memories          :",
    N_MEM
)

print(
    "Graph nodes       :",
    N_NODES
)

print(
    "Raw dimension     :",
    RAW_DIM
)

print(
    "Chance            :",
    f"{CHANCE:.3f}"
)

print(
    "Min op separation :",
    f"{pair_op.min():.4f}"
)

print(
    "Median separation :",
    f"{np.median(pair_op):.4f}"
)


# ================================================================
# 8. QUERY CORRUPTION
#
# Graph:
#     uniformly delete 20% of existing graph edges
#
# Raw:
#     remove 20% temporal samples from the same sensor interval,
#     interpolate, then compute non-graph summary features.
# ================================================================

print("\n[6/10] Generating corrupted queries...")


def corrupt_existing_edges(
    A,
    rr,
    frac
):

    Aq = A.copy()

    edges = np.argwhere(
        np.triu(
            Aq > 0,
            k=1
        )
    )

    m = len(
        edges
    )

    if m == 0:

        return Aq

    n_remove = int(
        np.floor(
            frac
            *
            m
        )
    )

    if n_remove <= 0:

        return Aq

    chosen = rr.choice(
        m,
        size=n_remove,
        replace=False
    )

    uv = edges[
        chosen
    ]

    Aq[
        uv[:, 0],
        uv[:, 1]
    ] = 0.0

    Aq[
        uv[:, 1],
        uv[:, 0]
    ] = 0.0

    return Aq


QUERY_L = []

QUERY_RAW = []


for run in range(
    N_RUNS
):

    graph_rng = np.random.default_rng(
        SEED
        +
        10000
        +
        run
    )

    raw_rng = np.random.default_rng(
        SEED
        +
        20000
        +
        run
    )

    qL = []

    qR = []

    for i in range(
        N_MEM
    ):

        # --------------------------------------------------------
        # Graph query
        # --------------------------------------------------------

        Aq = corrupt_existing_edges(
            MEM_A[i],
            graph_rng,
            EDGE_REMOVE_FRAC
        )

        Lq = laplacian(
            Aq
        )

        qL.append(
            Lq
        )

        # --------------------------------------------------------
        # Raw query
        #
        # Same underlying real interval, but incomplete raw
        # temporal observation.
        # --------------------------------------------------------

        Z = MEMORIES[
            i
        ][
            "signal"
        ]

        Zq = corrupt_raw_signal(
            Z,
            raw_rng,
            RAW_SAMPLE_REMOVE_FRAC
        )

        rq = raw_feature_vector(
            Zq
        )

        rq = (
            rq[
                None,
                :
            ]
            -
            RAW_MEAN
        ) / RAW_STD

        qR.append(
            rq[0]
        )

    QUERY_L.append(
        np.stack(
            qL
        )
    )

    QUERY_RAW.append(
        np.stack(
            qR
        )
    )


# ================================================================
# 9. OPERATOR DISTANCE
# ================================================================

print("\n[7/10] Computing hard retrieval...")


def op_d2(Q):

    Q = np.asarray(
        Q,
        dtype=np.float64
    )

    E = (
        Q[
            :,
            None,
            :,
            :
        ]
        -
        MEM_L[
            None,
            :,
            :,
            :
        ]
    )

    eig = np.linalg.eigvalsh(
        E
    )

    op = np.max(
        np.abs(
            eig
        ),
        axis=-1
    )

    return (
        op ** 2
    )


QUERY_D2 = [
    op_d2(
        Q
    )
    for Q in QUERY_L
]


# ================================================================
# LARGE-BETA HARD RETRIEVAL
# ================================================================

hard_G = []

hard_E = []

hard_R = []


for run in range(
    N_RUNS
):

    # ------------------------------------------------------------
    # Graph operator nearest memory
    # ------------------------------------------------------------

    pg = np.argmin(
        QUERY_D2[
            run
        ],
        axis=1
    )

    # ------------------------------------------------------------
    # Eu-DAM hard inner-product retrieval on Laplacians
    # ------------------------------------------------------------

    QF = QUERY_L[
        run
    ].reshape(
        N_MEM,
        L_DIM
    )

    pe = np.argmax(
        QF
        @
        L_FLAT.T,
        axis=1
    )

    # ------------------------------------------------------------
    # Raw hard retrieval
    # ------------------------------------------------------------

    pr = np.argmax(
        QUERY_RAW[
            run
        ]
        @
        MEM_RAW.T,
        axis=1
    )

    hard_G.append(
        np.mean(
            pg
            ==
            TARGET
        )
    )

    hard_E.append(
        np.mean(
            pe
            ==
            TARGET
        )
    )

    hard_R.append(
        np.mean(
            pr
            ==
            TARGET
        )
    )


hard_G = np.asarray(
    hard_G
)


hard_E = np.asarray(
    hard_E
)


hard_R = np.asarray(
    hard_R
)


def standard_error(x):

    x = np.asarray(
        x
    )

    if len(x) <= 1:

        return 0.0

    return (
        np.std(
            x,
            ddof=1
        )
        /
        np.sqrt(
            len(x)
        )
    )


print(
    "\n"
    +
    "=" * 92
)


print(
    "LARGE-BETA HARD RETRIEVAL"
)


print(
    "=" * 92
)


print(
    f"Graph-DAM  : "
    f"{hard_G.mean():.3f} ± "
    f"{standard_error(hard_G):.3f}"
)


print(
    f"Eu-DAM     : "
    f"{hard_E.mean():.3f} ± "
    f"{standard_error(hard_E):.3f}"
)


print(
    f"Raw Eu-DAM : "
    f"{hard_R.mean():.3f} ± "
    f"{standard_error(hard_R):.3f}"
)


# ================================================================
# REPOSITORY-STYLE SCALES
# ================================================================

print("\n[8/10] Computing repository-style scales...")


MEM_D2 = op_d2(
    MEM_L
)


upper = MEM_D2[
    np.triu_indices(
        N_MEM,
        k=1
    )
]


SCALE_OP = max(
    float(
        np.median(
            upper
        )
    ),
    EPS
)


S_LL = (
    L_FLAT
    @
    L_FLAT.T
)


SCALE_EU = max(
    float(
        np.median(
            np.std(
                S_LL,
                axis=1
            )
        )
    ),
    EPS
)


S_XX = (
    MEM_RAW
    @
    MEM_RAW.T
)


SCALE_RAW = max(
    float(
        np.median(
            np.std(
                S_XX,
                axis=1
            )
        )
    ),
    EPS
)


print(
    "Graph scale:",
    SCALE_OP
)


print(
    "Eu-L scale :",
    SCALE_EU
)


print(
    "Raw scale  :",
    SCALE_RAW
)


# ================================================================
# SOFTMAX
# ================================================================

def softmax_rows(Z):

    Z = np.asarray(
        Z,
        dtype=np.float64
    )

    Z = (
        Z
        -
        np.max(
            Z,
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


# ================================================================
# GRAPH-DAM
# ================================================================

def graph_dam_retrieve(
    Q0,
    beta
):

    Q = Q0.copy()

    last = None

    for _ in range(
        N_STEPS
    ):

        D = (
            op_d2(
                Q
            )
            /
            SCALE_OP
        )

        W = softmax_rows(
            -beta
            *
            D
        )

        Q = (
            W
            @
            L_FLAT
        ).reshape(
            len(Q0),
            N_NODES,
            N_NODES
        )

        pred = np.argmin(
            op_d2(
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


# ================================================================
# EU-DAM ON vec(L)
# ================================================================

def eu_dam_retrieve(
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
            Q
            @
            L_FLAT.T
        ) / SCALE_EU

        W = softmax_rows(
            beta
            *
            score
        )

        Q = (
            W
            @
            L_FLAT
        )

        pred = np.argmax(
            Q
            @
            L_FLAT.T,
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


# ================================================================
# RAW EU-DAM
# ================================================================

def raw_dam_retrieve(
    Q0,
    beta
):

    Q = Q0.copy()

    last = None

    for _ in range(
        N_STEPS
    ):

        score = (
            Q
            @
            MEM_RAW.T
        ) / SCALE_RAW

        W = softmax_rows(
            beta
            *
            score
        )

        Q = (
            W
            @
            MEM_RAW
        )

        pred = np.argmax(
            Q
            @
            MEM_RAW.T,
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


# ================================================================
# BETA SWEEP
# ================================================================

print("\n[9/10] Running beta sweep...")


ACC_G = np.zeros(
    (
        N_RUNS,
        len(BETAS)
    )
)


ACC_E = np.zeros_like(
    ACC_G
)


ACC_R = np.zeros_like(
    ACC_G
)


for run in range(
    N_RUNS
):

    for j, beta in enumerate(
        BETAS
    ):

        pg = graph_dam_retrieve(
            QUERY_L[
                run
            ],
            beta
        )

        pe = eu_dam_retrieve(
            QUERY_L[
                run
            ],
            beta
        )

        pr = raw_dam_retrieve(
            QUERY_RAW[
                run
            ],
            beta
        )

        ACC_G[
            run,
            j
        ] = np.mean(
            pg
            ==
            TARGET
        )

        ACC_E[
            run,
            j
        ] = np.mean(
            pe
            ==
            TARGET
        )

        ACC_R[
            run,
            j
        ] = np.mean(
            pr
            ==
            TARGET
        )

    print(
        f"  run {run + 1}/{N_RUNS} done"
    )


# ================================================================
# RESULTS
# ================================================================

print("\n[10/10] Results")


def mean_se(M):

    mean = np.mean(
        M,
        axis=0
    )

    se = (
        np.std(
            M,
            axis=0,
            ddof=1
        )
        /
        np.sqrt(
            M.shape[0]
        )
    )

    return (
        mean,
        se
    )


MG, SEG = mean_se(
    ACC_G
)


ME, SEE = mean_se(
    ACC_E
)


MR, SER = mean_se(
    ACC_R
)


print(
    "\n"
    +
    "=" * 92
)


print(
    f"{'beta':>10}"
    f"{'Graph-DAM':>25}"
    f"{'Eu-DAM':>25}"
    f"{'Raw Eu-DAM':>25}"
)


print(
    "-" * 92
)


for j, beta in enumerate(
    BETAS
):

    print(
        f"{beta:10.4g}"
        f"{MG[j]:12.3f} ± {SEG[j]:.3f}"
        f"{ME[j]:12.3f} ± {SEE[j]:.3f}"
        f"{MR[j]:12.3f} ± {SER[j]:.3f}"
    )


# ================================================================
# BEST
# ================================================================

ig = int(
    np.argmax(
        MG
    )
)


ie = int(
    np.argmax(
        ME
    )
)


ir = int(
    np.argmax(
        MR
    )
)


print(
    "\n"
    +
    "=" * 92
)


print(
    "BEST RETRIEVAL ACCURACY"
)


print(
    "=" * 92
)


print(
    f"Graph-DAM  : "
    f"{MG[ig]:.3f} ± {SEG[ig]:.3f}"
    f"   beta={BETAS[ig]:.4g}"
)


print(
    f"Eu-DAM     : "
    f"{ME[ie]:.3f} ± {SEE[ie]:.3f}"
    f"   beta={BETAS[ie]:.4g}"
)


print(
    f"Raw Eu-DAM : "
    f"{MR[ir]:.3f} ± {SER[ir]:.3f}"
    f"   beta={BETAS[ir]:.4g}"
)


# ================================================================
# PLOT
# ================================================================

fig, ax = plt.subplots(
    figsize=(
        7.2,
        4.8
    )
)


ax.errorbar(
    BETAS,
    MG,
    yerr=SEG,
    marker="o",
    linewidth=2,
    capsize=3,
    label="Graph-DAM"
)


ax.errorbar(
    BETAS,
    ME,
    yerr=SEE,
    marker="s",
    linewidth=2,
    capsize=3,
    label="Eu-DAM"
)


ax.errorbar(
    BETAS,
    MR,
    yerr=SER,
    marker="^",
    linewidth=2,
    capsize=3,
    label="Raw Eu-DAM"
)


ax.axhline(
    CHANCE,
    linestyle="--",
    linewidth=1,
    label=f"Chance ({CHANCE:.3f})"
)


ax.set_xscale(
    "log"
)


ax.set_ylim(
    0.0,
    1.03
)


ax.set_xlim(
    BETAS[0],
    BETAS[-1]
)


ax.set_xlabel(
    r"Inverse temperature $\beta_n$"
)


ax.set_ylabel(
    "Retrieval accuracy"
)


ax.legend(
    frameon=False
)


ax.grid(
    alpha=0.20
)


fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/har_retrieval_accuracy.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/har_retrieval_accuracy.pdf", bbox_inches="tight")

plt.show()


# ================================================================
# HARD-RANK DIAGNOSTIC
# ================================================================

graph_ranks = []

eu_ranks = []

raw_ranks = []


for run in range(
    N_RUNS
):

    D = QUERY_D2[
        run
    ]

    EU_SCORE = (
        QUERY_L[
            run
        ].reshape(
            N_MEM,
            L_DIM
        )
        @
        L_FLAT.T
    )

    RAW_SCORE = (
        QUERY_RAW[
            run
        ]
        @
        MEM_RAW.T
    )

    for i in range(
        N_MEM
    ):

        order_g = np.argsort(
            D[i]
        )

        order_e = np.argsort(
            -EU_SCORE[i]
        )

        order_r = np.argsort(
            -RAW_SCORE[i]
        )

        graph_ranks.append(
            int(
                np.where(
                    order_g == i
                )[0][0]
            )
            +
            1
        )

        eu_ranks.append(
            int(
                np.where(
                    order_e == i
                )[0][0]
            )
            +
            1
        )

        raw_ranks.append(
            int(
                np.where(
                    order_r == i
                )[0][0]
            )
            +
            1
        )


graph_ranks = np.asarray(
    graph_ranks
)


eu_ranks = np.asarray(
    eu_ranks
)


raw_ranks = np.asarray(
    raw_ranks
)


print(
    "\n"
    +
    "=" * 92
)


print(
    "QUERY → TARGET RANK DIAGNOSTIC"
)


print(
    "=" * 92
)


print(
    f"Graph-DAM : median={np.median(graph_ranks):.1f}, "
    f"top-1={np.mean(graph_ranks == 1):.3f}, "
    f"top-3={np.mean(graph_ranks <= 3):.3f}"
)


print(
    f"Eu-DAM    : median={np.median(eu_ranks):.1f}, "
    f"top-1={np.mean(eu_ranks == 1):.3f}, "
    f"top-3={np.mean(eu_ranks <= 3):.3f}"
)


print(
    f"Raw Eu-DAM: median={np.median(raw_ranks):.1f}, "
    f"top-1={np.mean(raw_ranks == 1):.3f}, "
    f"top-3={np.mean(raw_ranks <= 3):.3f}"
)


print("\nDONE.")