#%%
import pandas as pd
import re
from pathlib import Path
import numpy as np
#%%
RAW_FILE = Path('farsnet_facts.tsv')
CLEAN_FILE = Path('farsnet_facts_cleaned.parquet')
#%% md
# ## 1- Load And Normalizing Farsnet Facts File
#%%
ARABIC_CHARS = re.compile(r"[\u064B-\u065F]")
def normalise(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = ARABIC_CHARS.sub("", text)
    text = text.replace("ي", "ی").replace("ك", "ک")
    return text.strip()

print("Reading raw triples …")
df = pd.read_csv(
        RAW_FILE,
        sep="\t",
        names=["sid", "subject_words", "relation", "tid"],
        dtype=str,
        keep_default_na=False,
        quoting=3
)

print("Normalising text …")
for col in ("subject_words", "relation"):
    df[col] = df[col].map(normalise)

df = df.drop_duplicates(subset=["sid", "relation", "tid"]).reset_index(drop=True)
print("rows after de-dup:", len(df))

#Save the file
df.to_parquet(CLEAN_FILE, index=False)
print(f"Clean DataFrame written to {CLEAN_FILE}")
print(df.head())
#%% md
# ## 2- Triple Vectorization
# #### For each cleaned row, we will produce sid | subject_words | relation | tid. We will have a 301-dimensional vector.
#%%
from gensim.models import KeyedVectors
import tqdm
import pyarrow as pa,pyarrow.parquet as pq
# File paths
FT_PATH      = "cc.fa.300.vec.gz"   # <-- update if different
OUT_FILE_VEC = "farsnet_vectors.parquet"
#%%
# First load ft for efficiency (We just run it once)
ft = KeyedVectors.load_word2vec_format("cc.fa.300.vec.gz", binary=False)
ft.save("cc.fa.300.kv")
#%%
# for regular usage, we read the extracted model
ft = KeyedVectors.load("cc.fa.300.kv")
#%%
# loading cleaned triplets
df = pd.read_parquet(CLEAN_FILE, columns=["sid","subject_words","relation",'tid'])

# relation -> integer id
rel2id = {rel:i for i,rel in enumerate(sorted(df["relation"].unique()))}
df["rel_id"] = df["relation"].map(rel2id).astype("uint16")   # ≤ 65 535 relations

# subject embedding
def mean_ft(tokens):
    vecs = [ft[w] for w in tokens.split() if w in ft]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300, dtype="float32")

print("Vectorising …")
subs = [mean_ft(s) for s in tqdm.tqdm(df["subject_words"], total=len(df))]
sub_mat = np.vstack(subs).astype("float32")     # shape (N, 300)

# assemble (rel_id | emb)
rel_vec = df["rel_id"].to_numpy(dtype="uint16")[:,None]       # (N, 1)
feat_mat = np.hstack([rel_vec.astype("float32"), sub_mat])        # (N, 301)

# Persist
# relation-id column → float32 (N,1)
rel_vec_f32 = df["rel_id"].to_numpy(dtype=np.float32)[:, None]

# (N, 1 + 300)  float32
feat_mat = np.hstack([rel_vec_f32, sub_mat])

table = pa.table(
    {
        "sid":    pa.array(df["sid"]),
        "tid":    pa.array(df["tid"]),
        "rel_id": pa.array(df["rel_id"], type=pa.uint16()),
        "vec":    pa.FixedSizeListArray.from_arrays(feat_mat.ravel(), 301)
    })
pq.write_table(table, OUT_FILE_VEC)
print("Saved vectors to:", OUT_FILE_VEC)
#%% md
# ## 3- Clustering
# #### Clusters the 301-d feature vectors
# #### Then computes cluster_rank = average number of distinct objects (tids) per (sid, rel) pair inside each cluster
# #### Writes an enriched Parquet file
#%%
# set the paths and parameters
VEC_FILE   = r"farsnet_vectors.parquet"
OUT_FILE   = r"farsnet_vectors_clustered.parquet"
N_CLUSTERS = 20
RANDOM_SEED = 42
#%%
from sklearn.cluster import MiniBatchKMeans

#%%
# load vectors
table = pq.read_table(VEC_FILE)
df    = table.to_pandas()
mat   = np.vstack(df["vec"].to_numpy())  # (N, 301) float32

# clustering
print(f"Clustering with k={N_CLUSTERS} …")
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS,
                         random_state=RANDOM_SEED,
                         batch_size=10_000,
                         n_init="auto")
df["cluster"] = kmeans.fit_predict(mat).astype("uint8")

# compute cluster ranks
pair_size = df.groupby(["sid", "rel_id"])["tid"].nunique().rename("obj_cnt")
df = df.join(pair_size, on=["sid", "rel_id"])

# average within each cluster
rank = df.groupby("cluster")["obj_cnt"].mean().rename("cluster_rank")
df   = df.join(rank, on="cluster")
df.drop(columns="obj_cnt", inplace=True)
df["cluster_rank"] = df["cluster_rank"].astype("float32")

# save the results
cols = ["sid", "tid", "rel_id", "cluster", "cluster_rank", "vec"]
pq.write_table(pa.Table.from_pandas(df[cols]), OUT_FILE)
print("✓ Saved to: ", OUT_FILE)
#%% md
# ## 4- Question Formats Assignment
# #### Load the file produced in Step 3
# #### Map cluster rank to the question format
# #### Write a new Parquet file for step 5
#%%
# set the paths and parameters
CLUSTERED_FILE = r"farsnet_vectors_clustered.parquet"
OUT_FILE_QFMT  = r"farsnet_vectors_qfmt.parquet"
#%%
df = pd.read_parquet(CLUSTERED_FILE)
# question format mapping
def choose_format(rank: float) -> str:
    if rank <= 1.5:
        return "true_false"          # single answer
    elif rank <= 5:
        return "short_list"          # enumerate N objects
    elif rank <= 10:
        return "select_all_mcq"      # multi-choice
    else:
        return "hybrid_mcq"              # how-many

df["q_format"] = df["cluster_rank"].apply(choose_format).astype("category")

# save the results
cols = ["sid", "tid", "rel_id", "cluster", "cluster_rank", "q_format", "vec"]
df[cols].to_parquet(OUT_FILE_QFMT, index=False)
print(" Question formats assigned to:", OUT_FILE_QFMT)

#%% md
# ## 5- Template Library Creation
# #### Loads the file from Step 4
# #### Builds lightweight lookup tables:
# #####   . sid → subject_words and tid → object_words
# #####   . rel_id → relation_text
# #### Implements four reusable renderers – one per q_format
# #### Return a new Dataframe with one row per (sid,rel) containing:
# #####   . the question string
# #####   . the list of gold objects (for scoring)
# #####   . a list of distractors if the format needs them
#%%
# config and paths
VEC_QFMT_FILE = r"farsnet_vectors_qfmt.parquet"
CLEAN_FILE    = r"farsnet_facts_cleaned.parquet"
PROMPTS_RAW = r'prompts_raw.parquet'
MAX_ENUM      = 5     # cap list length in short_list
NEG_RATIO     = (2,1)
#%%
import random,json
from collections import defaultdict
import re
from pathlib import Path

#%%
TOKEN_RGX = re.compile(r"[ \t\u200c|\\/()\[\]{}]+")
PUNCT_RGX = re.compile(r"^[\W_]+|[\W_]+$")

# look up table
clean      = pd.read_parquet(CLEAN_FILE,
                              columns=["sid","subject_words","relation","tid"])
sid2text   = clean.drop_duplicates("sid").set_index("sid")["subject_words"].to_dict()
tid2text   = clean.drop_duplicates("tid").set_index("tid")["subject_words"].to_dict()
rel2id = {r:i for i,r in enumerate(sorted(clean["relation"].unique()))}
id2rel = {v:k for k,v in rel2id.items()}

#%%
df_fmt = pd.read_parquet(VEC_QFMT_FILE)



REL_MAP_FILE = "relation_text_map.json"
if Path(REL_MAP_FILE).is_file():
    rel_phrase = json.load(open(REL_MAP_FILE, encoding="utf-8"))
else:
    rel_phrase = defaultdict(str)

# crude mapping for demo
relid2txt = {
    rid: rel_phrase.get(rel, rel)
    for rid, rel in id2rel.items()
}

def get_rel_txt(rid: int) -> str:
    return relid2txt.get(rid, id2rel.get(rid, str(rid)))


def pretty_obj(txt: str) -> str:
    first = txt.split("|")[0]
    return PUNCT_RGX.sub("", first).strip()

slice_df = (
    df_fmt.groupby(["sid", "rel_id", "q_format"], observed=True)
          .agg(objs=("tid", list), cluster_rank=("cluster_rank", "first"))
          .reset_index()
)



MAX_ENUM = 4
def true_false(subj, rel_id, obj_texts, negs=None):
    """50 % chance of being True or False; False sample uses KGLens method."""
    rel_txt = get_rel_txt(rel_id)

    is_true = random.random() < 0.5
    if is_true:
        obj_txt = obj_texts[0]
        gold    = ["True"]
    else:
        cand_ids = [
            t for t in tid2text
            if t not in obj_texts
        ]
        fake_tid = random.choice(cand_ids) if cand_ids else None
        obj_txt  = pretty_obj(tid2text.get(fake_tid, "…"))
        gold     = ["False"]

    prompt = f"«آیا «{subj}» {rel_txt} «{obj_txt}» است؟ فقط «True» یا «False» پاسخ دهید.»"
    return prompt, gold


def short_list(subj, rel_id, obj_texts):
    rel_txt = get_rel_txt(rel_id)
    n = min(len(obj_texts), MAX_ENUM)
    return f"«{n} مورد از «{rel_txt}» برای «{subj}» را نام ببرید.»", obj_texts[:n]

def select_all_mcq(subj, rel_id, obj_texts, negs):
    rel_txt = get_rel_txt(rel_id)
    items   = obj_texts + negs
    random.shuffle(items)
    opts = "  ".join(f"[ ] {x}" for x in items)
    return f"«کدام گزینه‌ها {rel_txt} «{subj}» هستند؟»\n{opts}", obj_texts

def hybrid_mcq(subj, rel_id, obj_texts, negs):
    """count  +  select-all in one prompt"""
    rel_txt = get_rel_txt(rel_id)

    cnt_prompt = f"«چند {rel_txt} برای «{subj}» تعریف شده است؟»"
    items = obj_texts + negs
    random.shuffle(items)
    opts = "  ".join(f"[ ] {x}" for x in items)
    sel_prompt = f"«کدام گزینه‌ها {rel_txt} «{subj}» هستند؟»\n{opts}"

    full_prompt = cnt_prompt + "\n\n" + sel_prompt
    gold = [len(obj_texts)] + obj_texts
    return full_prompt, gold

RENDERER = {
    "true_false"      : lambda s,r,objs,negs=None: true_false(s,r,objs),
    "short_list"     : lambda s,r,objs,negs=None: short_list(s,r,objs),
    "select_all_mcq" : select_all_mcq,
    "hybrid_mcq"     : lambda s,r,objs,negs: hybrid_mcq(s, r, objs, negs),
}


all_obj_texts = [pretty_obj(t) for t in tid2text.values()]
pick_easy_negs = lambda k: random.sample(all_obj_texts, k)

#  prompt records
records = []

for _, row in slice_df.iterrows():
    sid      = row["sid"]
    rel_id   = row["rel_id"]
    qfmt     = row["q_format"]
    obj_tids = row["objs"]

    # skip rows without objects (paranoia)
    if not obj_tids:
        continue

    subj_txt  = sid2text.get(sid, f"id{sid}")
    obj_texts = [pretty_obj(tid2text[t]) for t in obj_tids]

    # hybrid_mcq → 3 prompts
    if qfmt == "hybrid_mcq":
        fmt_cycle = ["select_all_mcq", "short_list", "select_all_mcq"]
        used_opts = set()

        for fmt_var in fmt_cycle:
            negs = (
                random.sample(all_obj_texts, max(1, len(obj_texts) // 2))
                if fmt_var in ("select_all_mcq", "hybrid_mcq") else []
            )
            negs = [n for n in negs if n not in used_opts]     #
            used_opts.update(negs)

            prompt, gold_texts = RENDERER[fmt_var](subj_txt, rel_id, obj_texts, negs)

            records.append({
                "sid"        : sid,
                "rel_id"     : rel_id,
                "q_format"   : fmt_var,
                "question"   : prompt,
                "gold"       : gold_texts,
                "distractors": negs,
                "objs"       : obj_tids,
            })
        continue

    negs = (
        random.sample(all_obj_texts, max(1, len(obj_texts) // 2))
        if qfmt == "select_all_mcq" else []
    )

    prompt, gold_texts = RENDERER[qfmt](subj_txt, rel_id, obj_texts, negs)

    records.append({
        "sid"        : sid,
        "rel_id"     : rel_id,
        "q_format"   : qfmt,
        "question"   : prompt,
        "gold"       : gold_texts,
        "distractors": negs,          # FIXED: unified name
        "objs"       : obj_tids,
    })

prompts_df = pd.DataFrame(records)
prompts_df["gold"]        = prompts_df["gold"].apply(lambda lst: json.dumps(lst, ensure_ascii=False))
prompts_df["distractors"] = prompts_df["distractors"].apply(lambda lst: json.dumps(lst, ensure_ascii=False))

prompts_df.to_parquet(PROMPTS_RAW, index=False)
print("Built", len(prompts_df), "prompts")

#%%
df = pd.read_parquet("prompts_raw.parquet")
missing = sorted(set(range(11,101)) - set(df["sid"].unique()))
print("Missing SIDs:", missing)  # should be []

#%%
import pandas as pd

prompts_df = pd.read_parquet("prompts_raw.parquet")

# safe numeric conversion ─ non-digits → NaN
sid_nums = pd.to_numeric(prompts_df["sid"], errors="coerce")

# keep only the numeric sids and cast to int
present = set(sid_nums.dropna().astype(int))

# check the 11-100 range
missing = [i for i in range(11, 101) if i not in present]
print("Missing SIDs:", missing)          # → []

#%%
df_copy = prompts_df.copy()
df_copy["sid"] = df_copy["sid"].astype(int)
df.sortby('sid', inplace=True)
#%% md
# ## 6- Semantic Distractor / Negative Generation
# #### Enrich each prompt with hard-and easy negatives so the MCQ formats become truly challenging
#%%

import json, re, random
import numpy as np  # FIX
import pandas as pd  # FIX
import pyarrow as pa
import faiss
from gensim.models import KeyedVectors  # FIX
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)
np.random.seed(42)

#%%
# Load resources & raw prompts
ft = KeyedVectors.load("cc.fa.300.kv")
PROMPTS_RAW = "prompts_raw.parquet"
CLEAN_FACTS_PARQ = "farsnet_facts_cleaned.parquet"  # FIX: explicit path
PROMPTS_WITH_NEGS = "prompts_with_negs.parquet"

# regex for extract words
TOKEN_RGX = re.compile(r"[ \t\u200c|\\/()\[\]{}]+")
PUNCT_RGX = re.compile(r"^[\W_]+|[\W_]+$")

prompts_df = pd.read_parquet(PROMPTS_RAW)

# rebuild tidy tid → text  &  prettifier
clean = pd.read_parquet(CLEAN_FACTS_PARQ,
                        columns=["tid", "subject_words"])

tid2text = (
    clean.drop_duplicates("tid")
         .set_index("tid")["subject_words"]
         .to_dict()
)

def pretty_obj(txt: str) -> str:
    return txt.split("|")[0].strip()


all_obj_texts = [pretty_obj(t) for t in tid2text.values()]  # FIX

#%%
# Build vector index for hard / easy negatives

tid_vec = {}
for tid, txt in tid2text.items():
    words = [PUNCT_RGX.sub("", w) for w in TOKEN_RGX.split(txt) if w]
    vecs = [ft[w] for w in words if w in ft]
    if vecs:
        v = np.mean(vecs, axis=0, dtype=np.float32)
        tid_vec[tid] = v / (np.linalg.norm(v) + 1e-9)

ids, mat = zip(*tid_vec.items())
mat = np.vstack(mat).astype("float32")
index = faiss.IndexFlatIP(mat.shape[1])
index.add(mat)
tid2row = {t: i for i, t in enumerate(ids)}

#%% md
# #### Function to get k hard / k easy negatives
#%%
def negatives_for(gold_ids, hard_k, easy_k):
    hard, easy = [], []

    # HARD: nearest unrelated tails
    for gid in gold_ids:
        if gid in tid2row:
            v = mat[tid2row[gid]].reshape(1, -1)
            D, I = index.search(v, hard_k + 5)  # fetch a few extra
            for tid in (ids[i] for i in I[0] if ids[i] not in gold_ids):
                hard.append(tid)
                if len(hard) >= hard_k:
                    break
        if len(hard) >= hard_k:
            break

    # EASY: random tails
    pool = [t for t in tid2text if t not in gold_ids and t not in hard]
    easy = random.sample(pool, easy_k) if len(pool) >= easy_k else pool[:easy_k]
    return hard + easy

#%% md
# #### Augment prompts_df
#%%
HARD_EASY = {
    "select_all_mcq": (2, 1),
    "hybrid_mcq"   : (3, 0)
}

distractor_col = []
for _, row in prompts_df.iterrows():
    fmt = row["q_format"]
    if fmt in HARD_EASY:
        h, e   = HARD_EASY[fmt]
        gold_ids = set(row["objs"])
        neg_ids  = negatives_for(gold_ids, h, e)
        distractor_col.append([pretty_obj(tid2text[t]) for t in neg_ids])
    else:
        distractor_col.append([])

prompts_df["distractors"] = distractor_col

# serialise lists as JSON -------------------------------------------------
prompts_df["gold"]        = prompts_df["gold"].apply(
    lambda lst: json.dumps(lst, ensure_ascii=False)
)
prompts_df["distractors"] = prompts_df["distractors"].apply(
    lambda lst: json.dumps(lst, ensure_ascii=False)
)

prompts_df.to_parquet(PROMPTS_WITH_NEGS, index=False)
print("Step-6 saved to", PROMPTS_WITH_NEGS, "(rows:", len(prompts_df), ")")
#%% md
# #### Delete rows with zero distractors
#%%
mcq_mask = prompts_df["q_format"].isin(["select_all_mcq", "hybrid_mcq"])

empty_mask = prompts_df.loc[mcq_mask, "distractors"] \
    .apply(lambda j: len(json.loads(j)) == 0)

to_drop = prompts_df.index[mcq_mask & empty_mask]
print("MCQ rows with ZERO distractors:", len(to_drop))

prompts_df = prompts_df.drop(to_drop).reset_index(drop=True)
prompts_df.to_parquet(PROMPTS_WITH_NEGS, index=False)  # overwrite
print("Cleaned file written to:", PROMPTS_WITH_NEGS)

#%% md
# #### Fix duplicates
#%%
TARGET = {"select_all_mcq": 3, "hybrid_mcq": 3}  # desired min distractors

PROMPTS_FINAL = "prompts_final.parquet"

p = pd.read_parquet(PROMPTS_WITH_NEGS)


def json2list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, pa.lib.ListScalar):
        return x.as_py()
    if pd.isna(x) or x == "":
        return []
    return json.loads(x)


p["gold"] = p["gold"].apply(json2list)
p["distractors"] = p["distractors"].apply(json2list)

fixed_gold, fixed_negs = [], []
for _, row in p.iterrows():
    qfmt = row.q_format
    gold = list(dict.fromkeys(row.gold))
    negs = [n for n in dict.fromkeys(row.distractors) if n not in gold]

    # ensure min distractors for target formats --------------------------
    need = max(0, TARGET.get(qfmt, 0) - len(negs))
    if need:
        pool = [w for w in all_obj_texts if w not in gold and w not in negs]
        negs.extend(random.sample(pool, min(need, len(pool))))

    fixed_gold.append(gold)
    fixed_negs.append(negs)

p["gold"] = fixed_gold
p["distractors"] = fixed_negs

# ----------------------------------------------------------------------  # FIX: build options column
p["options"] = p.apply(
    lambda r: r["gold"] + r["distractors"] if r["distractors"] else [],
    axis=1
)

# rename columns to final schema ----------------------------------------  # FIX
p = p.rename(columns={"question": "prompt", "gold": "answer"})

cols = ["prompt", "answer", "options", "q_format",
        "sid", "rel_id", "objs", "distractors"]  # FIX order
p[cols].to_parquet(PROMPTS_FINAL, index=False)
print("✓", PROMPTS_FINAL, "written (rows:", len(p), ")")

#%% md
# #### Fix the gold column
#%%
import pyarrow.parquet as pq

tbl = pq.read_table(PROMPTS_FINAL)
PROMPTS = 'prompts.parquet'

def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, pa.lib.ListScalar):
        return x.as_py()
    if isinstance(x, str):
        return json.loads(x)
    return []


# map tid list → surface strings ----------------------------------------
objs_py = [as_list(x) for x in tbl.column("objs").to_pylist()]

new_answer = []
for tids in objs_py:
    texts = [tid2text.get(t, str(t)) for t in tids]
    uniq = list(dict.fromkeys(texts))
    new_answer.append(uniq)

answer_idx = tbl.schema.get_field_index("answer")  # FIX: column renamed
tbl = tbl.set_column(answer_idx, "answer",
                     pa.array([json.dumps(a, ensure_ascii=False)
                               for a in new_answer], pa.string()))

pq.write_table(tbl, PROMPTS, compression="zstd")
print("Gold/answer column deduplicated; file updated:", PROMPTS)

#%% md
#  ## 7-Determine Number of Evaluation Loops
# $$
# \begin{align*}
# n &= \left(\frac{Z\,\sigma}{E}\right)^2, & Z &= 1.96, & E &= 0.03, \\
# \sigma &= \sqrt{\frac{\sum_{i=1}^N (x_i - \bar{x})^2}{\,N - 1\,}}
# \end{align*}
# $$
# 
#%%
import json, random, numpy as np, pandas as pd, pyarrow as pa

#%%
PILOT_FRAC    = 0.05
PILOT_LOOPS   = 3
CONF_LEVEL    = 0.95
MARGIN_ERR    = 0.02

Z = 1.96
E = MARGIN_ERR
#%%
prompts = pd.read_parquet("prompts_final.parquet")

clusters = prompts[["sid", "rel_id"]].drop_duplicates()

n_sample = max(1, int(len(clusters) * PILOT_FRAC))
pilot_clusters = clusters.sample(n=n_sample, random_state=42)
pilot_prompts  = prompts.merge(pilot_clusters, on=["sid", "rel_id"])


def evaluate_stub(df):
    return np.random.normal(0.75, 0.02, size=len(df))

f1_scores = []
for i in range(PILOT_LOOPS):
    scores = evaluate_stub(pilot_prompts)
    f1     = float(np.mean(scores))
    f1_scores.append(f1)
    print(f"Pilot loop {i+1}: mean F1 = {f1:.4f}")

sigma = float(np.std(f1_scores, ddof=1))
print(f"\nPilot F1 SD (σ): {sigma:.4f}")

if sigma == 0 or np.isnan(sigma):
    n_req = max(PILOT_LOOPS, 10)
    print(f"σ ≈ 0 ⇒ defaulting to {n_req} loops")
else:
    n_est = (Z * sigma / E) ** 2
    n_req = int(np.ceil(n_est))
    print(f"Estimated total loops needed (n): {n_req}  (≈{n_est:.2f})")
#%% md
# ## 8- Making Questions
# ####  Load required libraries and define input/output file path
#%%
import json, random, pandas as pd, pyarrow as pa, numpy as np

PROMPTS_FILE = "prompts_final.parquet"
FACTS_FILE = "farsnet_facts_cleaned.parquet"
OUT_PARQ = "quiz_all.parquet"



#%% md
# #### Helper functions for parse Json/list-like columns and deduplicate
#%%
def j2list(x):
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    if isinstance(x, pa.lib.ListScalar):         return x.as_py()
    if isinstance(x, str):                      return json.loads(x)
    return []


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

#%% md
# #### Making questions with proper answers and options
#%%
facts = pd.read_parquet(FACTS_FILE, columns=["sid", "tid", "subject_words"])
sid2surf = facts.drop_duplicates("sid").set_index("sid")["subject_words"].to_dict()
tid2surf = facts.drop_duplicates("tid").set_index("tid")["subject_words"].to_dict()


def id2surf(_id):
    return sid2surf.get(_id) or tid2surf.get(_id, str(_id))


# load  prompts_final
use_cols = ["rel_id","sid", "prompt", "q_format", "objs", "distractors", "answer"]
raw = pd.read_parquet(PROMPTS_FILE, columns=use_cols)
raw["objs"] = raw["objs"].apply(j2list)
raw["distractors"] = raw["distractors"].apply(j2list)

random.seed(42)

# assemble records
records = []
for idx, row in raw.iterrows():
    qid = idx + 1
    fmt = row.q_format
    qtxt = row.prompt.splitlines()[0].strip()
    subj = id2surf(row.sid)

    # define options and answers
    if fmt == "true_false":
        answerm = ["True"] if "t" in str(row.answer).lower() else ["False"]
        options = ["True", "False"]

    elif fmt == "short_list":
        answerm = [id2surf(row.objs[0])] if row.objs else [""]
        options = []

    else:
        correct = [id2surf(x) for x in row.objs]
        options = dedup(correct + row.distractors)
        random.shuffle(options)
        num_map = {opt: str(i + 1) for i, opt in enumerate(options)}
        answerm = [num_map[o] for o in correct]  # list of numbers(str)

    records.append(
        dict(
            qid=qid,
            sid=row.sid,
            rel_id=row.rel_id,
            subject_words=subj,
            q_format=fmt,
            question=qtxt,
            options=options,
            answerm=answerm,
        )
    )

quiz_df = pd.DataFrame(records)
quiz_df.to_parquet(OUT_PARQ, index=False, compression="zstd")
print(f"{OUT_PARQ} written with {len(quiz_df)} questions")

#%% md
# ## 9- Build the Parameterised Knowledge Graph (PKG)
# #### Initialise the PKG with a Beta(1, 1) prior (`α = 1`, `β = 1`)
# #### compute the initial failure‐probability estimate `θ = α / (α + β) = 0.5`
#%%
import pandas as pd
import pyarrow.parquet as pq
#%%
QUIZ_FILE      = "quiz_all.parquet"
CLUSTERED_FILE = "farsnet_vectors_clustered.parquet"
PKG_FILE       = "pkg_v0.parquet"
#%%

quiz_keys = pd.read_parquet(
    QUIZ_FILE,engine='fastparquet', columns=["sid", "rel_id"]
).drop_duplicates(["sid", "rel_id"])

# attach cluster metadata computed earlier
cluster_meta = pd.read_parquet(
    CLUSTERED_FILE,engine='fastparquet', columns=["sid", "rel_id", "cluster", "cluster_rank"]
).drop_duplicates(["sid", "rel_id"])

pkg = (
    quiz_keys.merge(cluster_meta, on=["sid", "rel_id"], how="left")
             .reset_index(drop=True)
)

# initialise Beta parameters
pkg["alpha"] = 1
pkg["beta"]  = 1
pkg["theta"] = 0.5                # α / (α + β)

# build dataframe
pkg.to_parquet(PKG_FILE, index=False)
print(f"PKG initialised with {len(pkg):,} edges : {PKG_FILE}")
#%%

#%% md
# ## 10- Thompson Sampling-Based Question Selection
# #### For every edge in pkg draw γ ∼ Beta(α, β)
# #### Return the _k_ edges with the **highest** γ – those give the greatest expected information gain
#%%
import pandas as pd
import numpy as np

PKG_FILE   = "pkg_v0.parquet"
BATCH_SIZE = 64
#%%

def thompson_sample(k: int = BATCH_SIZE,
                    pkg_file: str = PKG_FILE,
                    seed: int | None = None) -> pd.DataFrame:
    """
    Thompson-sample the Parameterised KG and return the top-k edges.
    """
    pkg = pd.read_parquet(
        pkg_file,
        columns=["sid", "rel_id",
                 "cluster", "cluster_rank",
                 "alpha", "beta", "theta"]
    )

    rng = np.random.default_rng(seed)
    gamma = rng.beta(pkg["alpha"].to_numpy(),
                     pkg["beta"].to_numpy())
    # indices of k highest draws
    top_idx = np.argpartition(-gamma, k)[:k]
    batch   = pkg.iloc[top_idx].copy()

    batch["gamma"] = gamma[top_idx]
    return batch.reset_index(drop=True)

#%% md
# ## 11- Batched Querying to GPT-4o mini
# #### 1. Join the 64 edges returned by `thompson_sample()` with **quiz_all.parquet**
#    to fetch the concrete `prompt` + `qid`.
# #### 2. Chunk those 64 prompts into **10-prompt** API calls.
# #### 3. Send each chunk to GPT-4o-mini, log `(qid, sid, rel_id, prompt, response, latency)`.
#%%
import pandas as pd
import numpy as np
import time,os
from openai import OpenAI
from Config import get_openai_api_key
import pyarrow.parquet as pq
#%%
QUIZ_FILE        = "quiz_all.parquet"
PKG_FILE         = "pkg_v0.parquet"
GPT_LOG_FILE     = "gpt_log.parquet"

PROMPTS_PER_CALL = 10               
MODEL_NAME       = "gpt-4o"
client = OpenAI(api_key=get_openai_api_key())
#%% md
# #### Prompt initialization for the getting accurate answers
#%%
PROMPT_SUFFIX = {
    "true_false"      : "\nفقط True یا False.",
    "short_list"      : "\nفقط یک کلمه؛ توضیح ننویسید.",
    "select_all_mcq"  : "\nپاسخ را با «,» و بدون فاصله بنویسید؛همه سوال ها دست کم یک جواب دارند و فقط شماره یا شماره‌های درست انگلیسی (مثلاً «1,3»).",
}
#%% md
# #### Functions for making prompt more accurate
#%%
def _coerce_prompt_cols(df: pd.DataFrame) -> pd.DataFrame:

    if "prompt" not in df.columns:
        if "question" in df.columns:
            df = df.copy()
            df["prompt"] = df["question"]
        else:
            raise KeyError("Neither 'prompt' nor 'question' found.")
    return df

def add_user_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """
   make user prompts for the quiz questions.
    """
    df = _coerce_prompt_cols(df).copy()

    def _build(row: pd.Series) -> str:
        q_txt  = str(row["prompt"]).strip()
        q_type = str(row.get("q_format", "free")).lower()
        suffix = PROMPT_SUFFIX.get(q_type, "")

        # mcq questions
        if q_type == "select_all_mcq":
            opts_raw = row.get("options", "")

            if isinstance(opts_raw, (list, tuple, pd.Series, np.ndarray)):
                opts_seq = [str(o).strip() for o in opts_raw if str(o).strip()]
            else:                             # strِ کاما‌جدا
                opts_seq = [o.strip() for o in str(opts_raw).split(",") if o.strip()]

            opts_txt = "  ".join(f"{i}) {opt}" for i, opt in enumerate(opts_seq, 1))
            return f"{q_txt}\n{opts_txt}\n{suffix}"

        # true/false
        if q_type == "true_false":
            return f"{q_txt}{suffix}"

        # short_list
        if q_type == "short_list":
            return f"{q_txt}{suffix}"

        return q_txt

    df["user_prompt"] = df.apply(_build, axis=1)
    return df
#%%
df_quiz = pd.read_parquet(QUIZ_FILE)
df = add_user_prompts(df_quiz)
df.to_parquet(QUIZ_FILE, index=False)
df.head(10)
#%% md
# #### Helper functions
#%%
def as_text(x) -> str:
    """Return a plain string for the OpenAI chat API."""
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x)

def fetch_prompts_for_edges(
    edges: pd.DataFrame,
    quiz_file: str = QUIZ_FILE,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
        qid | sid | rel_id | q_format | prompt | options | gamma
    """
    import pyarrow.parquet as pq
    import pandas as pd

    schema   = pq.read_schema(quiz_file)
    text_col = "prompt"  if "prompt"  in schema.names else "question"
    opts_col = "options" if "options" in schema.names else None

    quiz_cols = ["qid", "sid", "rel_id", "q_format", text_col]
    if opts_col:
        quiz_cols.append(opts_col)

    quiz = pd.read_parquet(quiz_file, columns=quiz_cols)

    df = (
        edges
        .merge(quiz, on=["sid", "rel_id"], how="left")
        .dropna(subset=[text_col])
    )
    df.rename(columns={text_col: "prompt"}, inplace=True)

    # keep an `options` column even when the file has none – code downstream relies on it
    if "options" not in df.columns:
        df["options"] = None

    return df[["qid", "sid", "rel_id", "q_format", "prompt", "options", "gamma"]]

def gpt_batch_call(df_batch: pd.DataFrame,
                   model: str = MODEL_NAME,
                   temperature: float = 0.0,
                   max_tokens: int = 64) -> list[str]:

    df_batch = add_user_prompts(df_batch)
    replies  = []

    for q in df_batch["user_prompt"]:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": q}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        replies.append(resp.choices[0].message.content.strip())

    return replies


def query_llm_and_log(edges: pd.DataFrame,
                      prompts_per_call: int = PROMPTS_PER_CALL,
                      log_file: str      = GPT_LOG_FILE) -> None:
    """
    • Slice *edges* into batches, build user_prompt, call the model,
      and append results to gpt_log.parquet.
    • Each logged row now contains:
        qid, sid, rel_id, user_prompt, prompt, answer, response, latency, ts
    """
    # ── load quiz once for gold answers ───────────────────────────────────────
    schema    = pq.read_schema(QUIZ_FILE)
    text_col  = "prompt" if "prompt" in schema.names else "question"
    ans_col   = next(c for c in ["answer", "answerm", "gold",
                                 "gold_answer", "target"] if c in schema.names)

    quiz_gold = pd.read_parquet(QUIZ_FILE,
                                columns=["qid", "sid", "rel_id", ans_col])

    records = []

    for i in range(0, len(edges), prompts_per_call):
        batch_df = edges.iloc[i:i+prompts_per_call].reset_index(drop=True)

        # add user_prompt and merge gold answer
        batch_df = add_user_prompts(batch_df)
        batch_df = batch_df.merge(quiz_gold, on=["qid", "sid", "rel_id"],
                                  how="left")

        # ── call the model ────────────────────────────────────────────────────
        t0        = time.time()
        responses = gpt_batch_call(batch_df)          # uses .user_prompt
        latency   = time.time() - t0

        # ── assemble log rows ────────────────────────────────────────────────
        for j, r in enumerate(responses):
            row = batch_df.iloc[j]
            records.append({
                "qid":         row.qid,
                "sid":         row.sid,
                "rel_id":      row.rel_id,
                "user_prompt": row.user_prompt,
                "prompt":      row.prompt,
                "answer":      row[ans_col],
                "response":    r,
                "latency":     latency,
                "timestamp":   pd.Timestamp.utcnow()
            })

    log_df = pd.DataFrame.from_records(records)

    # ── append (or create) parquet log ───────────────────────────────────────
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        log_df.to_parquet(log_file, index=False, compression="snappy")
    else:
        pd.concat([pd.read_parquet(log_file), log_df],
                  ignore_index=True).to_parquet(
            log_file, index=False, compression="snappy"
        )

    print(f"logged {len(records)} responses to : {log_file}")

#%% md
# #### run the first loop
#%%
def run_one_loop(k: int = BATCH_SIZE,
                 pkg_file: str = PKG_FILE,
                 quiz_file: str = QUIZ_FILE,
                 prompts_per_call: int = PROMPTS_PER_CALL):
    """
    Query GPT-4o + log the answers
    """
    edges      = thompson_sample(k=k, pkg_file=pkg_file)
    edges_prom = fetch_prompts_for_edges(edges, quiz_file=quiz_file)
    query_llm_and_log(edges_prom, prompts_per_call=prompts_per_call)
#%%
run_one_loop()
#%% md
# ## 12- Parsing & Advanced Metrics Calculation
#%% md
# #### Parse every raw GPT response
# #### Compare with the gold answers in **quiz_all.parquet**
# #### Compute KGQuiz-style metric → F1
# #### Join cluster-id so we can roll-up stats per cluster
# #### Append the per-question results to **eval_log.parquet**
#%%
import os, re, json, unicodedata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
#%%
QUIZ_FILE      = "quiz_all.parquet"
PKG_FILE       = "pkg_v0.parquet"
GPT_LOG_FILE   = "gpt_log.parquet"
EVAL_LOG_FILE  = "eval_log.parquet"
CLUSTERED_FILE = "farsnet_vectors_clustered.parquet"
METRICS_LOG_FILE = ("gpt_log_with_metrics.parquet")  # NEW

#%%
# log = pd.read_parquet(GPT_LOG_FILE, engine='fastparquet')
# log.head(10)
#%%
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def normalise_digits(s: str) -> str:
    """Convert Persian ↦ Latin digits and strip whitespace."""
    return str(s).translate(PERSIAN_DIGITS).strip()

def str_to_set_numbers(txt: str) -> set[int]:
    """Return the set of 1-based option-indices contained in *txt*."""
    nums = re.findall(r"\d+", normalise_digits(txt))
    return {int(n) for n in nums}
#%%
schema   = pq.read_schema(QUIZ_FILE,  )
text_col = "prompt" if "prompt" in schema.names else "question"
col_need = ["qid", "sid", "rel_id", "q_format", "answer", "cluster_id"]
quiz_gt  = pd.read_parquet(QUIZ_FILE,engine='fastparquet', columns=[c for c in col_need if c in schema.names])

#%%
LOOPS_TOTAL   = 10          # number of Thompson iterations
BATCH_SIZE    = 10           # prompts per iteration
F1_SUCCESS    = 0.80         # MCQ success threshold
LR_GAMMA      = 0.05
#%%
def evaluate_last_batch(log_file: str = GPT_LOG_FILE,
                        batch_n : int = BATCH_SIZE,          # kept for API compat; not used
                        clustered_file: str = CLUSTERED_FILE
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    • Detect rows in gpt_log that do NOT yet have metrics (tp/fp/fn/f1 is NA)
    • Join with ground-truth + cluster labels
    • Compute TP / FP / FN (+ Precision / Recall / F1) row-wise
    • Return (row_metrics_for_new_rows, cluster_metrics_over_new_rows)
    """
    import numpy as np, pandas as pd, json, re, unicodedata

    df_all = (pd.read_parquet(log_file, engine="fastparquet")
                .sort_values("timestamp")
                .reset_index(drop=True))
    for c in ("tp","fp","fn","prec","recall","f1"):
        if c not in df_all.columns:
            df_all[c] = pd.NA

    new_mask = df_all[["tp","fp","fn","f1"]].isna().any(axis=1)
    if not new_mask.any():
        return (df_all.iloc[0:0].copy(), df_all.iloc[0:0].copy())

    df_new = df_all.loc[new_mask].copy()

    # ──────────────────────────────────────────────────────────────────────────
    # CHANGE: Enrich gpt_log with user_prompt and answer from quiz_all.parquet
    #         Falls back to `prompt` or `question` → user_prompt; `gold` → answer.
    try:
        quiz = pd.read_parquet("quiz_all.parquet", engine="fastparquet")

        # prefer qid if present in BOTH; else (sid, rel_id)
        if "qid" in df_new.columns and "qid" in quiz.columns:
            join_cols = ["qid"]
        elif {"sid","rel_id"}.issubset(df_new.columns) and {"sid","rel_id"}.issubset(quiz.columns):
            join_cols = ["sid","rel_id"]
        else:
            join_cols = [c for c in ("sid","rel_id","qid") if c in df_new.columns and c in quiz.columns]
            if not join_cols:
                raise KeyError("No common join key among ('qid', 'sid','rel_id').")

        # build user_prompt if missing in quiz
        if "user_prompt" not in quiz.columns:
            if "prompt" in quiz.columns:
                quiz = quiz.assign(user_prompt=quiz["prompt"])
            elif "question" in quiz.columns:
                quiz = quiz.assign(user_prompt=quiz["question"])
            else:
                quiz = quiz.assign(user_prompt=pd.NA)

        # build answer if missing; map from gold
        if "answer" not in quiz.columns and "gold" in quiz.columns:
            quiz = quiz.assign(answer=quiz["gold"])

        keep_cols = [c for c in ("user_prompt","answer","subject_words") if c in quiz.columns]  # subject_words kept if present
        if keep_cols:
            df_add = quiz[join_cols + keep_cols].drop_duplicates(join_cols)
            df_new = df_new.merge(df_add, on=join_cols, how="left")

            # ensure columns exist in df_all, then persist values for just-updated rows
            for c in ("user_prompt","answer","subject_words"):
                if c in df_new.columns:
                    if c not in df_all.columns:
                        df_all[c] = pd.NA
                    df_all.loc[new_mask, c] = df_new[c].values
    except Exception as e:
        print(f"[evaluate_last_batch] Skipped enrichment of user_prompt/answer: {e}")
    # ──────────────────────────────────────────────────────────────────────────

    # ───── Your existing parsing + metric computation goes here (unchanged) ───
    def _to_list(x):
        if isinstance(x, list): return x
        if pd.isna(x): return []
        try:
            return json.loads(x) if isinstance(x, str) and x.strip().startswith(("[", "{")) else [str(x)]
        except Exception:
            return [str(x)]

    def _norm(s):
        if s is None or (isinstance(s, float) and pd.isna(s)): return ""
        s = str(s).strip()
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("ي","ی").replace("ك","ک").replace("\u200c"," ").replace("\u0640","")
        s = re.sub(r"[«»\"'“”]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _get_opts(row):
        opts = row.get("options", None)
        if isinstance(opts, list): return [_norm(o) for o in opts]
        if isinstance(opts, str) and opts.strip().startswith("["):
            try:
                arr = json.loads(opts)
                if isinstance(arr, list):
                    return [_norm(o) for o in arr]
            except Exception:
                return []
        return []

    def _parse_gold(row):
        gold_col = "gold" if "gold" in row.index else ("answer" if "answer" in row.index else None)
        g = row.get(gold_col, None)
        if isinstance(g, list): vals = g
        elif isinstance(g, str) and g.strip().startswith("["):
            try: vals = json.loads(g)
            except Exception: vals = [g]
        else:
            vals = [] if pd.isna(g) else [g]
        vals = [_norm(v) for v in vals if v is not None]
        opts = _get_opts(row)
        out = []
        import re as _re
        for v in vals:
            if _re.fullmatch(r"\d+", v) and opts:
                i = int(v) - 1
                if 0 <= i < len(opts): out.append(opts[i])
            else:
                out.append(v)
        seen=set(); res=[]
        for v in out:
            if v and v not in seen:
                seen.add(v); res.append(v)
        return res

    def _parse_pred(row):
        for c in ("pred","parsed_answer","choices","model_answer","response"):
            if c in row and pd.notna(row[c]):
                raw = row[c]; break
        else:
            return []
        vals = []
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                arr = json.loads(raw)
                if isinstance(arr, list):
                    vals = [_norm(v) for v in arr]
            except Exception:
                pass
        if not vals:
            s = _norm(raw)
            opts = _get_opts(row)
            import re as _re
            if opts:
                letters = _re.findall(r"\b([A-H])\b", s) + _re.findall(r"\b([a-h])\b", s)
                numbers = _re.findall(r"\b([1-9]|10)\b", s)
                map_letter = {chr(65+i): opts[i] for i in range(min(26, len(opts)))}
                map_letter.update({chr(97+i): opts[i] for i in range(min(26, len(opts)))})
                map_number = {str(i+1): opts[i] for i in range(len(opts))}
                for L in letters:
                    if L in map_letter: vals.append(map_letter[L])
                for n in numbers:
                    if n in map_number: vals.append(map_number[n])
                for o in opts:
                    if o and o in s:
                        vals.append(o)
            if not vals:
                parts = [t for t in re.split(r"[,\|؛،/؛/\n]+", s) if t.strip()]
                vals = [_norm(p) for p in parts]
        seen=set(); out=[]
        for v in vals:
            if v and v not in seen:
                seen.add(v); out.append(v)
        return out

    tp_list, fp_list, fn_list, prec_list, rec_list, f1_list = [], [], [], [], [], []
    for _, row in df_new.iterrows():
        gold = set(_parse_gold(row))
        pred = set(_parse_pred(row))
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        tp_list.append(tp); fp_list.append(fp); fn_list.append(fn)
        prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)

    df_new["tp"]    = tp_list
    df_new["fp"]    = fp_list
    df_new["fn"]    = fn_list
    df_new["prec"]  = prec_list
    df_new["recall"]= rec_list
    df_new["f1"]    = f1_list

    # persist back to log (includes the new user_prompt/answer columns)
    df_all.loc[new_mask, ["tp","fp","fn","prec","recall","f1"]] = df_new[["tp","fp","fn","prec","recall","f1"]].values
    df_all.to_parquet(log_file, engine="fastparquet", index=False)

    # cluster metrics (unchanged)
    if "cluster" in df_new.columns:
        cluster_metrics = (df_new.groupby("cluster", dropna=False)
                                 .agg(tp=("tp","sum"), fp=("fp","sum"), fn=("fn","sum"))
                                 .reset_index())
        cluster_metrics["prec"]   = cluster_metrics["tp"] / (cluster_metrics["tp"] + cluster_metrics["fp"]).replace(0, np.nan)
        cluster_metrics["recall"] = cluster_metrics["tp"] / (cluster_metrics["tp"] + cluster_metrics["fn"]).replace(0, np.nan)
        cluster_metrics["f1"]     = (2*cluster_metrics["prec"]*cluster_metrics["recall"]/
                                     (cluster_metrics["prec"]+cluster_metrics["recall"])).replace({np.nan:0.0})
    else:
        cluster_metrics = df_new.iloc[0:0].copy()

    return df_new[["timestamp","sid","rel_id","tp","fp","fn","prec","recall","f1"]], cluster_metrics

#%%

#%%
# row_metrics, cluster_metrics = evaluate_last_batch()
# display(row_metrics.head(20))
# row_metrics.head(50).to_csv("metrics_log.csv", index=False, encoding="utf-8-sig")
# display(cluster_metrics.head())
#%% md
# ## 13-  Dynamic PKG update  (α for failures, β for wins)
#%%
import pandas as pd, numpy as np, os, time
import pyarrow.parquet as pq
from openai import OpenAI
from Config import get_openai_api_key
import re
#%%
client = OpenAI(api_key=get_openai_api_key())
#%%
PKG_FILE      = "pkg_v0.parquet"
GPT_LOG_FILE  = "gpt_log.parquet"
QUIZ_FILE    = "quiz_all.parquet"
CLUSTERED_FILE = "farsnet_vectors_clustered.parquet"
METRICS_LOG_FILE = "gpt_log_with_metrics.parquet"
#%%
LOOPS_TOTAL   = 100     # number of Thompson iterations
BATCH_SIZE    = 1300          # prompts per iteration
F1_SUCCESS    = 0.80         # MCQ success threshold
LR_GAMMA      = 0.05
MODEL_NAME = 'gpt-4o'
#%%
def random_sample_pct(pct: float,
                      pkg_file: str = PKG_FILE,
                      seed: int | None = None) -> pd.DataFrame:
    """
    Randomly sample ~pct% of PKG rows; each row has independent pct% chance.
    Shape / columns match thompson_sample() output to avoid downstream changes.
    """
    import numpy as np, pandas as pd
    assert 0 < pct <= 100, "pct must be in (0, 100]"
    pkg = pd.read_parquet(
        pkg_file, engine="fastparquet",
        columns=["sid","rel_id","cluster","cluster_rank","alpha","beta","theta"]
    )
    rng = np.random.default_rng(seed)
    mask = rng.random(len(pkg)) < (pct / 100.0)
    batch = pkg.loc[mask].copy()

    # Keep 'gamma' column to avoid changing downstream code that expects it
    # CHANGE: gamma is a throwaway uniform draw here (purely for compatibility)
    if len(batch):
        batch["gamma"] = rng.random(len(batch))
    else:
        batch["gamma"] = []
    return batch.reset_index(drop=True)

#%%
import os
import pandas as pd

# CHANGED: initialize gpt_log with metrics columns present from the start
def ensure_stage13_initialized(pkg_src_file: str = CLUSTERED_FILE,
                               pkg_file: str     = PKG_FILE,
                               log_file: str     = GPT_LOG_FILE,
                               force: bool = False) -> None:
    import os, pandas as pd
    if force or not os.path.exists(pkg_file):
        base = pd.read_parquet(pkg_src_file, engine='fastparquet')
        keep = [c for c in ["sid","rel_id","cluster","cluster_rank"] if c in base.columns]
        pkg  = base[keep].drop_duplicates(["sid","rel_id"] + (["cluster"] if "cluster" in keep else [])).copy()
        if "alpha" not in pkg.columns: pkg["alpha"] = 1
        if "beta"  not in pkg.columns: pkg["beta"]  = 1
        if "theta" not in pkg.columns: pkg["theta"] = pkg["alpha"] / (pkg["alpha"] + pkg["beta"])
        pkg.to_parquet(pkg_file, engine='fastparquet', compression='snappy', index=False)

    if force or not os.path.exists(log_file):
        cols = ["timestamp","qid","sid","rel_id","response","answer","tp","fp","fn","f1"]  # CHANGED
        pd.DataFrame(columns=cols).to_parquet(log_file, engine='fastparquet', compression='snappy', index=False)



#%% md
# 
#%%
def update_pkg_with_eval(row_metrics: pd.DataFrame,
                         pkg_file: str = PKG_FILE) -> None:
    """
    Update PKG α/β from evaluation results (success→β+=1, failure→α+=1).
    Minimal + local (no neighbor propagation). Keep as-is elsewhere.
    """
    import pandas as pd
    if row_metrics is None or row_metrics.empty:
        return

    # Success criterion: f1 > 0 (keep identical to your current policy)
    rm = row_metrics.copy()
    rm["success"] = (rm["f1"] > 0).astype(int)
    rm["fail"]    = (rm["success"] == 0).astype(int)

    # load pkg and update by (sid, rel_id)
    pkg = pd.read_parquet(pkg_file, engine="fastparquet")
    key_cols = [c for c in ("sid","rel_id") if c in pkg.columns and c in rm.columns]
    if not key_cols:
        raise KeyError("Cannot align row_metrics with PKG; need matching 'sid' and/or 'rel_id' columns.")

    updates = (rm.groupby(key_cols, dropna=False)
                 .agg(dalpha=("fail","sum"), dbeta=("success","sum"))
                 .reset_index())

    pkg = pkg.merge(updates, on=key_cols, how="left")
    pkg["dalpha"] = pkg["dalpha"].fillna(0).astype(int)
    pkg["dbeta"]  = pkg["dbeta"].fillna(0).astype(int)

    pkg["alpha"] = pkg["alpha"] + pkg["dalpha"]
    pkg["beta"]  = pkg["beta"]  + pkg["dbeta"]

    # refresh theta = alpha / (alpha+beta) (or your exact estimator)
    denom = (pkg["alpha"] + pkg["beta"]).clip(lower=1)
    pkg["theta"] = (pkg["alpha"] / denom).astype(float)

    # drop temp columns, save back
    pkg = pkg.drop(columns=["dalpha","dbeta"])
    pkg.to_parquet(pkg_file, engine="fastparquet", index=False)

#%%
# ──────────────────────────  helpers (re-written)  ──────────────────────────
import os, time, pyarrow.parquet as pq, pandas as pd
from typing import List

def as_text(x) -> str:
    """Return a plain string for the OpenAI chat API."""
    return " ".join(map(str, x)) if isinstance(x, list) else str(x)

def fetch_prompts_for_edges(
        edges: pd.DataFrame,
        quiz_file: str = QUIZ_FILE
) -> pd.DataFrame:
    """
    Return a tidy DF with **user_prompt already present**.

    Columns: qid | sid | rel_id | q_format | prompt | options | user_prompt | gamma
    """
    schema   = pq.read_schema(quiz_file)
    text_col = "prompt"   if "prompt"   in schema.names else "question"
    opts_col = "options"  if "options"  in schema.names else None
    up_col   = "user_prompt"                                  # <- already built
    gamma_ok = "gamma" in edges.columns

    quiz_cols = ["qid", "sid", "rel_id", "q_format", text_col, up_col]
    if opts_col: quiz_cols.append(opts_col)

    quiz = pd.read_parquet(quiz_file, engine='fastparquet', columns=quiz_cols)

    # CHANGED: join on BOTH keys to avoid fan-out on subjects with many rels/templates
    df = (edges.merge(quiz, on=["sid", "rel_id"], how="left")  # CHANGED
               .dropna(subset=[up_col]))                       # ensure prompt exists

    # CHANGED: enforce ONE prompt per (sid, rel_id) and a stable preference
    prio_cols = [c for c in ["cluster_rank", "qid", "rel_id", "sid"] if c in df.columns]  # CHANGED
    if prio_cols:
        df = df.sort_values(prio_cols)                                                    # CHANGED
    df = df.drop_duplicates(["sid", "rel_id"], keep="first")                              # CHANGED

    # CHANGED: hard-cap to exactly the number of sampled unique edges (≈ BATCH_SIZE)
    expected_k = len(edges.drop_duplicates(["sid", "rel_id"])) if {"sid","rel_id"}.issubset(edges.columns) else len(df)  # CHANGED
    df = df.head(expected_k)                                                              # CHANGED

    df.rename(columns={text_col: "prompt"}, inplace=True)
    if opts_col and "options" not in df.columns:
        df["options"] = None                       # downstream safety
    if not gamma_ok:
        df["gamma"] = edges.get("gamma", 0.5)      # preserve gamma col

    return df[["qid","sid","rel_id","q_format",
               "prompt","options","user_prompt","gamma"]]


def gpt_batch_call(df_batch: pd.DataFrame,
                   model: str = MODEL_NAME,
                   temperature: float = 0.0,
                   max_tokens: int = 64) -> List[str]:
    """One OpenAI call per prompt – keeps accounting simple."""
    replies = []
    for q in df_batch["user_prompt"]:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": q}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        replies.append(resp.choices[0].message.content.strip())
    return replies


def query_llm_and_log(df_prompts: pd.DataFrame,
                      prompts_per_call: int = BATCH_SIZE,
                      model: str = MODEL_NAME,
                      temperature: float = 0.0,
                      max_tokens: int = 64,
                      log_file: str = GPT_LOG_FILE) -> None:
    """
    Deduplicate to one prompt per (sid, rel_id), cap to prompts_per_call, call the model,
    and append exactly that many responses to gpt_log.parquet (minimal schema).
    """
    import os
    import pandas as pd
    from datetime import datetime

    # CHANGED: dedupe and cap BEFORE calling the model
    prio_cols = [c for c in ["cluster_rank", "qid", "rel_id", "sid"] if c in df_prompts.columns]  # CHANGED
    df_batch = df_prompts.sort_values(prio_cols) if prio_cols else df_prompts.copy()              # CHANGED
    key_cols = [c for c in ["sid","rel_id"] if c in df_batch.columns]                             # CHANGED
    if key_cols:
        df_batch = df_batch.drop_duplicates(key_cols, keep="first")                               # CHANGED
    df_batch = df_batch.head(prompts_per_call)                                                    # CHANGED

    # one OpenAI call per prompt (your existing helper)
    replies = gpt_batch_call(df_batch, model=model, temperature=temperature, max_tokens=max_tokens)

    # build minimal log rows
    ts = pd.Timestamp.utcnow().isoformat()
    df_new = pd.DataFrame({
        "timestamp": [ts] * len(df_batch),
        "qid":       df_batch["qid"].tolist() if "qid" in df_batch.columns else [pd.NA]*len(df_batch),
        "sid":       df_batch["sid"].tolist(),
        "rel_id":    df_batch["rel_id"].tolist(),
        "response":  replies,
    })

    # CHANGED: normalize text if helper exists (prevents \uXXXX in saved parquet)
    if ' _norm_fa ' in globals():  # won't trigger; keep safe check below
        pass
    try:
        df_new["response"] = df_new["response"].apply(_norm_fa)  # CHANGED
    except Exception:
        pass  # if _norm_fa is not defined, skip silently

    # CHANGED: de-dupe again and hard-cap to ensure EXACT count persisted
    key_cols2 = [c for c in ["timestamp","sid","rel_id","qid"] if c in df_new.columns]           # CHANGED
    if key_cols2:
        df_new = df_new.drop_duplicates(key_cols2, keep="first")                                  # CHANGED
    df_new = df_new.head(prompts_per_call)                                                        # CHANGED

    # CHANGED: safe append with schema alignment (silences FutureWarning)
    if os.path.exists(log_file):
        old = pd.read_parquet(log_file, engine='fastparquet')
    else:
        old = pd.DataFrame(columns=["timestamp","qid","sid","rel_id","response"])                 # CHANGED

    cols = list(dict.fromkeys(list(old.columns) + list(df_new.columns)))                          # CHANGED
    old = old.reindex(columns=cols)
    df_new = df_new.reindex(columns=cols)

    out = pd.concat([old, df_new], ignore_index=True, sort=False)
    out.to_parquet(log_file, engine='fastparquet', compression='snappy', index=False)
    print(f"logged {len(df_new)} responses → {os.path.basename(log_file)}")  # CHANGED

#%%
def run_random_sampling_loop(pct: float,
                             pkg_file: str = PKG_FILE,
                             prompts_per_call: int | None = None,   # CHANGE: allow None = send all
                             seed: int | None = None
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stage 13 one-shot:
      (1) sample ~pct% edges uniformly at random
      (2) fetch prompts, query LLM, append to gpt_log
      (3) evaluate just-logged rows
      (4) update PKG alpha/beta/theta
    Returns (row_metrics, cluster_metrics).
    """
    import pandas as pd

    # (1) random sampling
    edges = random_sample_pct(pct=pct, pkg_file=pkg_file, seed=seed)
    if edges.empty:
        print(f"[Stage13] No edges sampled at pct={pct}%.")
        return (pd.DataFrame(), pd.DataFrame())

    # (2) prompt + log
    edges_prom = fetch_prompts_for_edges(edges)  # UNCHANGED

    # CHANGE: if no cap is provided, send *all* sampled prompts in one go
    if prompts_per_call is None:
        prompts_per_call = len(edges_prom)

    print(f"[Stage13] Sampled {len(edges)} edges @ {pct}% → sending {len(edges_prom)} prompts "
          f"(prompts_per_call={prompts_per_call}).")

    query_llm_and_log(edges_prom, prompts_per_call=prompts_per_call)  # UNCHANGED API

    # (3) evaluate last batch
    row_metrics, cluster_metrics = evaluate_last_batch()

    # (4) update PKG
    update_pkg_with_eval(row_metrics, pkg_file=pkg_file)

    return row_metrics, cluster_metrics

#%%
run_random_sampling_loop(pct=20)
#%%
import pandas as pd
df = pd.read_parquet("gpt_log.parquet", engine='fastparquet')
df.to_csv('gpt_log.csv', index=False, encoding='utf-8-sig')

#%% md
# ## 14- Statistical Analysis by Question and Cluster Type
#%%
# === Stage 14 · Helpers (NEW) ================================================
import os, re, json, unicodedata, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Config (auto-fallbacks if globals not set)
GPT_LOG_FILE = globals().get("GPT_LOG_FILE", "gpt_log.parquet")
QUIZ_FILE    = "quiz_all.parquet"
PKG_FILE     = globals().get("PKG_FILE", "pkg.parquet")
OUT_DIR      = "stage14_reports"
os.makedirs(OUT_DIR, exist_ok=True)

def _read_parquet_any(path, columns=None):
    """Try fastparquet then pyarrow."""
    try:
        return pd.read_parquet(path, engine="fastparquet", columns=columns)
    except Exception:
        return pd.read_parquet(path, engine="pyarrow", columns=columns)

# -------- text & list parsing ----------
def _norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ي","ی").replace("ك","ک").replace("\u200c"," ").replace("\u0640","")
    s = re.sub(r"[«»\"'“”]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _to_list_maybe_json(x):
    if isinstance(x, list): return x
    if pd.isna(x): return []
    if isinstance(x, str) and x.strip().startswith(("[","{")):
        try:
            v = json.loads(x)
            if isinstance(v, list): return v
        except Exception:
            pass
    return [x]

def _parse_options(row):
    opts = row.get("options")
    if isinstance(opts, list): return [_norm(o) for o in opts]
    if isinstance(opts, str) and opts.strip().startswith("["):
        try:
            arr = json.loads(opts)
            if isinstance(arr, list): return [_norm(o) for o in arr]
        except Exception:
            return []
    return []

def _parse_gold_texts(row):
    g = row.get("gold", row.get("answer", None))
    vals = _to_list_maybe_json(g)
    vals = [_norm(v) for v in vals if v is not None]
    opts = _parse_options(row)
    out = []
    for v in vals:
        if re.fullmatch(r"\d+", v) and opts:
            i = int(v) - 1
            if 0 <= i < len(opts): out.append(opts[i])
        else:
            out.append(v)
    # dedupe
    seen=set(); res=[]
    for v in out:
        if v and v not in seen:
            seen.add(v); res.append(v)
    return res

def _parse_pred_texts(row):
    # prefer explicit parsed columns if present
    for c in ("pred","parsed_answer","choices","model_answer","response"):
        if c in row and pd.notna(row[c]):
            raw = row[c]; break
    else:
        return []
    vals = []
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list): vals = [_norm(v) for v in arr]
        except Exception:
            pass
    if not vals:
        s = _norm(raw)
        opts = _parse_options(row)
        if opts:
            letters = re.findall(r"\b([A-H])\b", s) + re.findall(r"\b([a-h])\b", s)
            numbers = re.findall(r"\b([1-9]|10)\b", s)
            map_letter = {chr(65+i): opts[i] for i in range(min(26, len(opts)))}
            map_letter.update({chr(97+i): opts[i] for i in range(min(26, len(opts)))})
            map_number = {str(i+1): opts[i] for i in range(len(opts))}
            for L in letters:
                if L in map_letter: vals.append(map_letter[L])
            for n in numbers:
                if n in map_number: vals.append(map_number[n])
            for o in opts:
                if o and o in s: vals.append(o)
        if not vals:
            parts = [t for t in re.split(r"[,\|؛،/؛/\n]+", s) if t.strip()]
            vals = [_norm(p) for p in parts]
    # dedupe
    seen=set(); out=[]
    for v in vals:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

# -------- per-row metrics (if needed) ----------
def _ensure_row_metrics(df):
    need = any(c not in df.columns for c in ("tp","fp","fn","prec","recall","f1"))
    if not need: return df
    tps, fps, fns, pres, recs, f1s = [], [], [], [], [], []
    for _, row in df.iterrows():
        gold = set(_parse_gold_texts(row))
        pred = set(_parse_pred_texts(row))
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        tps.append(tp); fps.append(fp); fns.append(fn)
        pres.append(prec); recs.append(rec); f1s.append(f1)
    df = df.copy()
    df["tp"]=tps; df["fp"]=fps; df["fn"]=fns
    df["prec"]=pres; df["recall"]=recs; df["f1"]=f1s
    return df

def _micro(tp, fp, fn):
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1   = (2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 0.0
    return prec, rec, f1

def _kappa_option_level(g: pd.DataFrame):
    """Cohen's κ treating each (question, option) as a binary label."""
    TP = TN = FP = FN = 0
    any_row = False
    for _, row in g.iterrows():
        opts = _parse_options(row)
        if not opts: continue
        any_row = True
        gold = set(_parse_gold_texts(row))
        pred = set(_parse_pred_texts(row))
        for o in opts:
            g_yes = (o in gold); p_yes = (o in pred)
            if g_yes and p_yes: TP += 1
            elif (not g_yes) and (not p_yes): TN += 1
            elif (not g_yes) and p_yes: FP += 1
            elif g_yes and (not p_yes): FN += 1
    if not any_row: return np.nan
    N = TP + TN + FP + FN
    if N == 0: return np.nan
    Po = (TP + TN) / N
    p_yes_gold = (TP + FN) / N
    p_yes_pred = (TP + FP) / N
    Pe = p_yes_gold * p_yes_pred + (1 - p_yes_gold) * (1 - p_yes_pred)
    if math.isclose(1 - Pe, 0.0): return np.nan
    return (Po - Pe) / (1 - Pe)

def _cronbach_alpha_group(_g: pd.DataFrame):
    """
    Cronbach's alpha needs >= 2 respondents; with one model/run it's not identifiable.
    We return NaN and note this in the report.
    """
    return np.nan

def _summarize_group(g: pd.DataFrame):
    tp = int(g["tp"].sum()); fp = int(g["fp"].sum()); fn = int(g["fn"].sum())
    prec, rec, f1 = _micro(tp, fp, fn)
    kappa = _kappa_option_level(g)
    alpha = _cronbach_alpha_group(g)
    return pd.Series({
        "n_rows": len(g), "TP": tp, "FP": fp, "FN": fn,
        "Precision": prec, "Recall": rec, "F1": f1,
        "Cohen_Kappa": kappa, "Cronbach_Alpha": alpha
    })

#%%
# === Stage 14 · Run & Plot (NEW) =============================================
# 1) Load logs + quiz (+ pkg for cluster if needed)
df_log = _read_parquet_any(GPT_LOG_FILE)
quiz   = _read_parquet_any(QUIZ_FILE)

# Enrich with q_format & options (if available)
join_cols = ["qid"] if ("qid" in df_log.columns and "qid" in quiz.columns) else ["sid","rel_id"]
extra_cols = [c for c in ("q_format","options") if c in quiz.columns]
if extra_cols:
    df_log = df_log.merge(quiz[join_cols + extra_cols].drop_duplicates(join_cols),
                          on=join_cols, how="left")

# Ensure cluster present; pull from PKG if needed
if "cluster" not in df_log.columns and os.path.exists(PKG_FILE):
    pkg = _read_parquet_any(PKG_FILE)
    if {"sid","rel_id","cluster"}.issubset(pkg.columns):
        df_log = df_log.merge(pkg[["sid","rel_id","cluster"]].drop_duplicates(["sid","rel_id"]),
                              on=["sid","rel_id"], how="left")

# 2) Ensure row metrics exist
df_log = _ensure_row_metrics(df_log)
if "q_format" not in df_log.columns: df_log["q_format"] = "unknown"
if "cluster"  not in df_log.columns: df_log["cluster"]  = "unknown"

# 3) Group summaries
by_qformat = (df_log.groupby("q_format", dropna=False)
                    .apply(_summarize_group)
                    .reset_index()
                    .sort_values("F1", ascending=False))

by_cluster = (df_log.groupby("cluster", dropna=False)
                    .apply(_summarize_group)
                    .reset_index()
                    .sort_values("F1", ascending=False))

by_qformat_cluster = (df_log.groupby(["q_format","cluster"], dropna=False)
                           .apply(_summarize_group)
                           .reset_index()
                           .sort_values(["q_format","F1"], ascending=[True, False]))

# 4) Save reports
by_qformat_path = os.path.join(OUT_DIR, "metrics_by_qformat.csv")
by_cluster_path = os.path.join(OUT_DIR, "metrics_by_cluster.csv")
by_qc_path      = os.path.join(OUT_DIR, "metrics_by_qformat_cluster.csv")
by_qformat.to_csv(by_qformat_path, index=False, encoding="utf-8")
by_cluster.to_csv(by_cluster_path, index=False, encoding="utf-8")
by_qformat_cluster.to_csv(by_qc_path, index=False, encoding="utf-8")

print("Saved:")
print(" -", by_qformat_path)
print(" -", by_cluster_path)
print(" -", by_qc_path)

print("\nPreview — metrics_by_qformat (top 10):")
print(by_qformat.head(10).to_string(index=False))
print("\nPreview — metrics_by_cluster  (top 10):")
print(by_cluster.head(10).to_string(index=False))

# 5) Plots (matplotlib only, one chart per figure, default colors)
# F1 by q_format
plt.figure()
plt.bar(by_qformat["q_format"].astype(str).values, by_qformat["F1"].values)
plt.title("F1 by Question Format")
plt.ylabel("F1 (micro)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f1_by_qformat.png"), dpi=150)
plt.show()

# Cohen's Kappa by q_format
plt.figure()
plt.bar(by_qformat["q_format"].astype(str).values, by_qformat["Cohen_Kappa"].values)
plt.title("Cohen's Kappa by Question Format")
plt.ylabel("Kappa")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kappa_by_qformat.png"), dpi=150)
plt.show()

# Top 25 clusters by F1
top_n = 25
top_clusters = by_cluster.head(top_n)
plt.figure()
plt.bar(top_clusters["cluster"].astype(str).values, top_clusters["F1"].values)
plt.title(f"Top {top_n} Clusters by F1")
plt.ylabel("F1 (micro)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"f1_by_cluster_top{top_n}.png"), dpi=150)
plt.show()

# Cohen's Kappa for same top clusters
plt.figure()
kappas = by_cluster.set_index("cluster").loc[top_clusters["cluster"], "Cohen_Kappa"].values
plt.bar(top_clusters["cluster"].astype(str).values, kappas)
plt.title(f"Cohen's Kappa for Top {top_n} Clusters (by F1)")
plt.ylabel("Kappa")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"kappa_by_cluster_top{top_n}.png"), dpi=150)
plt.show()

# Precision–Recall scatter (Top 200 clusters by F1)
top_scatter = by_cluster.head(200).copy()
plt.figure()
plt.scatter(top_scatter["Precision"].values, top_scatter["Recall"].values)
for _, r in top_scatter.iterrows():
    plt.annotate(str(r["cluster"]), (r["Precision"], r["Recall"]), fontsize=6, alpha=0.7)
plt.title("Precision vs Recall by Cluster (Top 200 by F1)")
plt.xlabel("Precision"); plt.ylabel("Recall")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "precision_recall_scatter_clusters.png"), dpi=150)
plt.show()

print("\n[Note] Cronbach’s alpha usually requires ≥2 respondents (e.g., multiple models or repeated runs). "
      "With a single model pass, α isn’t identifiable, so we report NaN. "
      "If you have multiple runs, I can enable α by stacking respondents.)")

#%%
df = pd.read_parquet('pkg_v0.parquet', engine='fastparquet')
df.to_csv('pkg_v0.csv', index=False, encoding='utf-8-sig')
#%%
# KGLens-style PKG Lens Map — whole graph + results in one plot
# Inputs:
#   /mnt/data/pkg_v0.csv  (sid, rel_id, cluster, cluster_rank, alpha, beta, theta)
#   /mnt/data/gpt_log.csv (sid, rel_id, tp, fp, fn, f1, ...)
# Outputs:
#   /mnt/data/kglens_pkg/kglens_pkg_map.png
#   /mnt/data/kglens_pkg/edge_metrics.csv

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

PKG_CSV = "pkg_v0.csv"
LOG_CSV = "gpt_log.csv"
OUT_DIR = "kglens_pkg"
os.makedirs(OUT_DIR, exist_ok=True)

def _read_csv(path):
    try:    return pd.read_csv(path)
    except: return pd.read_csv(path, engine="python", sep=None)

pkg = _read_csv(PKG_CSV)
log = _read_csv(LOG_CSV)

# Key alignment
pkg["sid"] = pkg["sid"].astype(str); log["sid"] = log["sid"].astype(str)
pkg["rel_id"] = pd.to_numeric(pkg["rel_id"], errors="coerce").astype("Int64")
log["rel_id"] = pd.to_numeric(log["rel_id"], errors="coerce").astype("Int64")
for c in ("tp","fp","fn","f1"):
    if c in log.columns: log[c] = pd.to_numeric(log[c], errors="coerce")

# Edge-level sums from the log
edge = (log.groupby(["sid","rel_id"], dropna=False)
            .agg(tp=("tp","sum"), fp=("fp","sum"), fn=("fn","sum"))
            .reset_index())

def _micro(tp, fp, fn):
    tp, fp, fn = float(tp or 0), float(fp or 0), float(fn or 0)
    prec = (tp/(tp+fp)) if (tp+fp)>0 else 0.0
    rec  = (tp/(tp+fn)) if (tp+fn)>0 else 0.0
    f1   = (2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.0
    return prec, rec, f1

edge["Precision"], edge["Recall"], edge["F1"] = zip(*edge.apply(lambda r: _micro(r["tp"], r["fp"], r["fn"]), axis=1))

# Join PKG attributes
keep = [c for c in ("sid","rel_id","cluster","cluster_rank","alpha","beta","theta") if c in pkg.columns]
edge_pkg = edge.merge(pkg[keep].drop_duplicates(["sid","rel_id"]), on=["sid","rel_id"], how="left")
edge_pkg["cluster"] = edge_pkg["cluster"].astype("Int64") if "cluster" in edge_pkg.columns else pd.Series([np.nan]*len(edge_pkg), dtype="Int64")

# Order clusters by weakness (micro-F1)
clu = (edge_pkg.groupby("cluster", dropna=False)
              .agg(TP=("tp","sum"), FP=("fp","sum"), FN=("fn","sum"), n_edges=("sid","size"))
              .reset_index())
clu["F1"] = clu.apply(lambda r: _micro(r["TP"], r["FP"], r["FN"])[2], axis=1)
clu_sorted = clu.sort_values(["F1","n_edges"], ascending=[True, False]).reset_index(drop=True)
cluster_to_x = {cl: i for i, cl in enumerate(clu_sorted["cluster"].tolist())}
edge_pkg["x_cluster"] = edge_pkg["cluster"].map(cluster_to_x)

# Order subjects inside each cluster by their worst edge F1
subj_rank = (edge_pkg.groupby(["cluster","sid"], dropna=False)
                    .agg(minF1=("F1","min"), n_edges=("F1","size"))
                    .reset_index()
                    .sort_values(["cluster","minF1","n_edges"], ascending=[True, True, False]))
subj_rank["y_rank"] = subj_rank.groupby("cluster").cumcount()
subj_to_y = {(r["cluster"], r["sid"]): int(r["y_rank"]) for _, r in subj_rank.iterrows()}
edge_pkg["y_subj"] = edge_pkg.apply(lambda r: subj_to_y.get((r["cluster"], r["sid"]), np.nan), axis=1)

# Coordinates + encodings
rng = np.random.default_rng(42)
edge_pkg["x"] = edge_pkg["x_cluster"] + (rng.random(len(edge_pkg)) - 0.5) * 0.6
edge_pkg["y"] = edge_pkg["y_subj"]    + (rng.random(len(edge_pkg)) - 0.5) * 0.6
edge_pkg["weakness"] = 1.0 - edge_pkg["F1"].fillna(0.0)

if {"alpha","beta"}.issubset(edge_pkg.columns):
    strength = (edge_pkg["alpha"].fillna(1).astype(float) + edge_pkg["beta"].fillna(1).astype(float)).clip(lower=1.0)
    norm = (strength - strength.min()) / max(1e-9, strength.max() - strength.min())
    edge_pkg["size"] = 8 + (1.0 - norm) * 72    # larger = less evidence
else:
    edge_pkg["size"] = 20.0

# Save per-edge table
edge_pkg.to_csv(os.path.join(OUT_DIR, "edge_metrics.csv"), index=False, encoding="utf-8")

# Single KGLens-style plot
plt.figure(figsize=(14, max(6, min(40, int(subj_rank["y_rank"].max() + 5)))))
sc = plt.scatter(edge_pkg["x"].values,
                 edge_pkg["y"].values,
                 s=edge_pkg["size"].values,
                 c=edge_pkg["weakness"].values,
                 alpha=0.75)
xticks_pos = list(range(len(clu_sorted)))
xticks_lbl = [str(int(c)) if pd.notna(c) else "unknown" for c in clu_sorted["cluster"].tolist()]
plt.xticks(xticks_pos, xticks_lbl, rotation=90)

max_y = int(subj_rank["y_rank"].max()) if len(subj_rank) else 0
yticks = list(range(0, max(1, max_y+1), max(1, (max_y+1)//10 or 1)))
plt.yticks(yticks)

plt.title("KGLens-style PKG Lens Map — whole graph in one view\n"
          "X: clusters (weak→strong), Y: subjects (weak→strong within cluster), "
          "dot=color=weakness (1−F1), dot size=less evidence (α+β)")
plt.xlabel("Cluster (sorted by micro-F1, weakest → strongest)")
plt.ylabel("Subject rank within cluster (lower = weaker)")

cb = plt.colorbar(sc); cb.set_label("Weakness (1 − F1)")
plt.tight_layout()
out_png = os.path.join(OUT_DIR, "kglens_pkg_map.png")
plt.savefig(out_png, dpi=150)
plt.show()

print("Saved plot:", out_png)
print("Edge metrics:", os.path.join(OUT_DIR, "edge_metrics.csv"))

#%%

#%%
