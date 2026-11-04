"""Phase B/E analysis: source keys, exact-dup groups, perceptual near-dup pairs, atomic leakage units.
Reads only cache files produced by 01_extract_inventory.py.
"""
import os, re
import numpy as np, pandas as pd

C = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\04_recovery_investigation\analysis_logs\cache"
inv = pd.read_csv(os.path.join(C, "inventory_raw.csv"))
inv["source_prefix"] = inv.filename.str.split(".rf.").str[0].str.lower()

RULES = [
    ("VIDEO_FRAME_SUFFIX", re.compile(r"^(.*?_(?:mp4|mov|avi|mkv|webm|gif|m4v|3gp))-\d+_(?:jpg|png|jpeg)$")),
    ("MARITIME_MVI", re.compile(r"^(mvi_\d+_(?:vis|nir)(?:_haze)?)_frame\d+_(?:jpg|png)$")),
    ("VISDRONE_SEQ", re.compile(r"^(\d{7})_\d{5}_d_\d{7}_(?:jpg|png)$")),
    ("KAKAOTALK_BATCH", re.compile(r"^(kakaotalk_\d{8}_\d+)(?:_\d+)?_(?:jpg|jpeg|png)$")),
    ("FRAME_NUMBER_SUFFIX", re.compile(r"^(.*?(?:frame|output|scene|clip|img_)[_-]?)\d+_(?:jpg|png|jpeg)$")),
]

def vkey(p):
    for name, rx in RULES:
        m = rx.match(p)
        if m:
            k = m.group(1)
            return ("visdrone_" + k if name == "VISDRONE_SEQ" else k), name
    return p, "PREFIX_ONLY"

vk = inv.source_prefix.map(vkey)
inv["video_key"] = [a for a, b in vk]
inv["video_key_rule"] = [b for a, b in vk]

# exact duplicate groups
shas = inv.image_sha256.unique()
sid = {s: i for i, s in enumerate(sorted(shas))}
inv["sha_group"] = inv.image_sha256.map(sid)
inv["sha_group_size"] = inv.groupby("sha_group").image_id.transform("size")
inv["sha_group_splits"] = inv.groupby("sha_group").split.transform(lambda x: ",".join(sorted(set(x))))
inv["sha_group_label_variants"] = inv.groupby("sha_group").label_sha256.transform("nunique")

# perceptual hashing on unique images
rep = inv.sort_values("image_id").drop_duplicates("sha_group")[["sha_group", "phash", "phash_flip", "dhash"]]
ph = np.array([int(x, 16) for x in rep.phash], dtype=np.uint64)
pf = np.array([int(x, 16) for x in rep.phash_flip], dtype=np.uint64)
dh = np.array([int(x, 16) for x in rep.dhash], dtype=np.uint64)
gids = rep.sha_group.values
N = len(ph)
pairs = []
for i0 in range(0, N, 400):
    blk = ph[i0:i0 + 400, None]
    d1 = np.bitwise_count(blk ^ ph[None, :]).astype(np.int16)
    d2 = np.bitwise_count(blk ^ pf[None, :]).astype(np.int16)
    dd = np.bitwise_count(dh[i0:i0 + 400, None] ^ dh[None, :]).astype(np.int16)
    d = np.minimum(d1, d2)
    ii, jj = np.nonzero(d <= 12)
    for a, b in zip(ii, jj):
        ga, gb = i0 + a, b
        if ga < gb:
            pairs.append((gids[ga], gids[gb], int(d1[a, b]), int(d2[a, b]), int(dd[a, b])))
pp = pd.DataFrame(pairs, columns=["sha_group_a", "sha_group_b", "phash_dist", "phash_flip_dist", "dhash_dist"])
pp["min_dist"] = pp[["phash_dist", "phash_flip_dist"]].min(1)
pp.to_csv(os.path.join(C, "phash_pairs_le12.csv"), index=False)
print("unique images", N, "pairs<=12", len(pp), pp.min_dist.value_counts().sort_index().to_dict())

# union-find for atomic leakage units: exact sha + same source_prefix + same video_key + phash<=8 (incl. flip, dhash<=16 confirm)
par = list(range(len(inv)))
def f(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
def u(a, b):
    ra, rb = f(a), f(b)
    if ra != rb:
        par[rb] = ra
first_of = {}
for col in ["sha_group", "source_prefix", "video_key"]:
    for idx, key in zip(inv.index, inv[col]):
        if key in first_of.setdefault(col, {}):
            u(first_of[col][key], idx)
        else:
            first_of[col][key] = idx
sg_first = inv.drop_duplicates("sha_group").set_index("sha_group").index.to_series()
sg_row = {g: i for i, g in zip(inv.index, inv.sha_group) if g not in locals().get("_seen", set())}
sg_row = inv.drop_duplicates("sha_group").set_index("sha_group")["image_id"].to_dict()
strong = pp[(pp.min_dist <= 8) & (pp.dhash_dist <= 20)]
for a, b in zip(strong.sha_group_a, strong.sha_group_b):
    u(sg_row[a], sg_row[b])
inv["leak_unit_raw"] = [f(i) for i in inv.index]
lu = {r: k for k, r in enumerate(sorted(inv.leak_unit_raw.unique()))}
inv["leak_unit"] = inv.leak_unit_raw.map(lu); inv.drop(columns="leak_unit_raw", inplace=True)
sz = inv.groupby("leak_unit").size()
print("leak units", len(sz), "largest", sz.sort_values().tail(10).to_dict())
print("unit size dist", sz.describe().to_dict())
inv.to_csv(os.path.join(C, "inventory_keys.csv"), index=False)
print(inv.video_key_rule.value_counts().to_dict())
print("video keys", inv.video_key.nunique(), "prefixes", inv.source_prefix.nunique())

