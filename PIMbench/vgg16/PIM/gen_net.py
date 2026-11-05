# usage: python3 gen_net.py PROFILE SEED QUANT
#   PROFILE : A  = 3 conv layers, small values: all activations stay below 32767 (requantization must do nothing)
#             B  = A + a 4th conv (stride 2), pixel-like inputs (0..255) and weights -7..7: requantization MUST kick in
#   SEED    : integer, makes the random data reproducible
#   QUANT   : new = percentile requantizer (ONCNN_CALIB_PCT, default 0.9999, round-to-nearest, saturate)
#             old = max-based floor shift (your original oncnnRequantize)
import sys, os, math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

profile, seed, quant = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert profile in ("A", "B") and quant in ("old", "new")
PCT = float(os.environ.get("ONCNN_CALIB_PCT", "0.9999"))
PACK = os.environ.get("ONCNN_PACK", "flat").lower()
assert PACK in ("flat", "pixel")
VARIANT = os.environ.get("ONCNN_VARIANT", "SS").upper()
assert VARIANT in ("SS", "SP")
TOTAL_PES = int(os.environ.get("ONCNN_TOTAL_PES", "4096"))
LANES = 16
COLS_C = 16
DELTA_M = int(os.environ.get("ONCNN_DELTA_M", "2" if VARIANT == "SP" else "3"))
DELTA_A = int(os.environ.get("ONCNN_DELTA_A", "2"))
PA_BITS = int(os.environ.get("ONCNN_PA", "16"))
TREE_LVL = int(math.ceil(math.log2(LANES)))
C_SOP = DELTA_M + DELTA_A * TREE_LVL + 2 * PA_BITS + TREE_LVL
INT16_MAX = 32767
rng = np.random.default_rng(seed)

def ceil_div(a, b):
    return (a + b - 1) // b

def ref_chunks_and_calls(cout, hout, wout, cin, kh, kw, num_pes=TOTAL_PES, pack=PACK):
    cols = min(COLS_C, num_pes)
    rows_f = max(1, num_pes // cols)
    outer = ceil_div(cout, rows_f) * ceil_div(hout * wout, cols)
    if pack == "pixel":
        chunks = kh * kw * ceil_div(cin, LANES)
        cycles = outer * (C_SOP + 1) * chunks
    else:
        chunks = ceil_div(cin * kh * kw, LANES)
        cycles = outer * ceil_div((C_SOP + 1) * cin * kh * kw, LANES)
    return chunks, outer * chunks, cycles

# ---- network description: (type, params...) ----------------------------------------------
C, H, W = 3, 12, 16
net = [("conv", 8, 3, 3, 1, 1, "4d"), ("relu",), ("pool", 2, 2),
       ("conv", 16, 3, 3, 1, 1, "3d"), ("relu",), ("pool", 2, 2),
       ("conv", 20, 3, 3, 1, 1, "4d"), ("relu",)]
if profile == "B":
    net += [("conv", 8, 3, 3, 2, 1, "3d")]           # stride 2, signed output, no ReLU

# ---- reference operators (independent of the C++ code) -----------------------------------
def requant(x):
    """what conv2ONCNN does to its input before convolving; returns (xq, shift, clipped, zeroed)"""
    x = x.astype(np.int64)
    a = np.abs(x)
    if quant == "old":
        mx = int(a.max())
        s = 0
        while (mx >> s) > INT16_MAX: s += 1
        return (x >> s), s, 0, int(((a != 0) & ((x >> s) == 0)).sum())     # arithmetic (floor) shift
    mags = np.sort(a[a != 0])
    if mags.size == 0: return x, 0, 0, 0
    k = min(mags.size - 1, int(PCT * mags.size))
    ref = int(mags[k]); s = 0
    while (ref >> s) > INT16_MAX: s += 1
    if s == 0: return x, 0, 0, 0
    half = 1 << (s - 1)
    q = (a + half) >> s
    clipped = int((q > INT16_MAX).sum()); q = np.minimum(q, INT16_MAX)
    return np.where(x < 0, -q, q), s, clipped, int(((a != 0) & (q == 0)).sum())

def conv(x, w4, stride, pad):
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = swv(xp, w4.shape[2:], axis=(1, 2))[:, ::stride, ::stride]      # cin,oh,ow,kh,kw
    return np.einsum("cijkl,fckl->fij", win, w4)

def pool(x, k, s):
    c, h, w = x.shape
    oh, ow = (h - k) // s + 1, (w - k) // s + 1
    y = np.empty((c, oh, ow), np.int64)
    for i in range(oh):
        for j in range(ow):
            y[:, i, j] = x[:, i * s:i * s + k, j * s:j * s + k].reshape(c, -1).max(1)
    return y

# ---- run the reference ---------------------------------------------------------------------
XMAX, WMAX = (4, 1) if profile == "A" else (256, 7)                 # B: pixel-like inputs and bigger weights
x = rng.integers(0, XMAX, size=(C, H, W)).astype(np.int64)
meta, weights, expected = [f"input {C} {H} {W}"], [], []
cur = x
total_calls = 0
total_cycles = 0
for li, st in enumerate(net):
    if st[0] == "conv":
        _, cout, kh, kw, stride, pad, mode = st
        cin = cur.shape[0]
        if mode == "4d":
            w4 = rng.integers(-WMAX, WMAX + 1, size=(cout, cin, kh, kw)).astype(np.int64); weights.append(w4)
        else:                                                      # shared 2D kernel per filter
            w3 = rng.integers(-WMAX, WMAX + 1, size=(cout, kh, kw)).astype(np.int64); weights.append(w3)
            w4 = np.repeat(w3[:, None], cin, axis=1)               # same kernel on every channel
        xq, s, clipped, zeroed = requant(cur)
        if profile == "A": assert s == 0, "profile A must not need requantization"
        cur = conv(xq, w4, stride, pad)
        assert np.abs(cur).max() < 2**31
        chunks, calls, cyc = ref_chunks_and_calls(cout, cur.shape[1], cur.shape[2], cin, kh, kw)
        total_calls += calls
        total_cycles += cyc
        print(f"stage {li}: conv {mode} -> {cur.shape}  input shift={s} clipped={clipped} nonzero->0={zeroed}  chunks={chunks} calls={calls} eq10_cycles={cyc}")
        meta.append(f"conv {cout} {kh} {kw} {stride} {pad} {mode}")
    elif st[0] == "relu":
        cur = np.maximum(cur, 0); meta.append("relu")
        print(f"stage {li}: relu -> {cur.shape}")
    else:
        cur = pool(cur, st[1], st[2]); meta.append(f"pool {st[1]} {st[2]}")
        print(f"stage {li}: pool -> {cur.shape}")
    expected.append(cur.flatten())
print(f"ref_summary: pack={PACK} variant={VARIANT} total_calls={total_calls} total_eq10_cycles={total_cycles}")
sv = lambda arr, fn: open(fn, "w").write(" ".join(map(str, arr.tolist())))
sv(x.flatten(), "net_input.txt")
sv(np.concatenate([w.flatten() for w in weights]), "net_weights.txt")
sv(np.concatenate(expected), "net_expected.txt")
open("net_meta.txt", "w").write("\n".join(meta) + "\n")