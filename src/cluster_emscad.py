"""Steps 2-8 of the topic-modelling study: features, K-Means, HDBSCAN, LDA, comparison, sensitivity.

AI assistance: this script was written with help from Claude (an AI assistant made by Anthropic).
See the AI-use statement in the dissertation.

It implements my analysis plan (a private working document, not in this repository). Every setting
below comes from that plan, which is written before any clustering is run.

GUARD: the real analysis refuses to run until the plan is locked, i.e. the "Date locked" line and
the "Your expectations" line in the plan have been filled in. `smoke` is a software test on a small
random subsample; it writes to _smoke folders and prints no themes.

Run from the Dissertation folder (use the project's own Python):

    .venv\\Scripts\\python.exe src\\cluster_emscad.py smoke                 # test the code (any time)

    .venv\\Scripts\\python.exe src\\cluster_emscad.py features              # R1: TF-IDF + SVD (primary text, base settings)
    .venv\\Scripts\\python.exe src\\cluster_emscad.py embed                 # R2: sentence embeddings (needs sentence-transformers)
                                                                           #     then run kselect/kmeans/hdbscan with --variant embed
    .venv\\Scripts\\python.exe src\\cluster_emscad.py kselect               # K-Means k = 5..30, 5 seeds, silhouette and stability
    .venv\\Scripts\\python.exe src\\cluster_emscad.py kmeans                # final model at the chosen k + themes + comparison
    .venv\\Scripts\\python.exe src\\cluster_emscad.py hdbscan               # density-based clustering, unclustered share
    .venv\\Scripts\\python.exe src\\cluster_emscad.py lda                   # LDA comparison around the chosen k
    .venv\\Scripts\\python.exe src\\cluster_emscad.py sensitivity           # S1, S2 and preprocessing variants vs primary
    .venv\\Scripts\\python.exe src\\cluster_emscad.py balanced              # class-balance check: 5 balanced refits (plan section 8)
    .venv\\Scripts\\python.exe src\\cluster_emscad.py hf --hf-approved      # OPTIONAL, descriptive only (plan section 10b)

Outputs: data/interim/ (labels, features, reading samples; not for git) and results/ (aggregated
tables and figures; includes results/run_log.jsonl with settings and package versions).
"""
import argparse
import itertools
import json
import platform
import re
import shutil
import sys
import time
import zlib
from pathlib import Path

import joblib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import sklearn  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.cluster import HDBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "emscad_clean.parquet"
PLAN = ROOT / "notes" / "analysis_plan_topic_modelling_2026-11-19.md"

# ---- settings fixed by the plan -------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
K_MIN, K_MAX = 5, 30
SVD_DIMS = 100
STABILITY_MIN = 0.80          # mean pairwise ARI across seeds
SIL_SAMPLE = 5000             # fixed random sample of adverts for silhouette
HDB_GRID = [30, 50, 100]
HDB_MIN_SAMPLES = 10
N_READ = 15                   # adverts read per cluster when naming themes
N_SECOND_CODER = 100
BOOT = 1000
MIN_FRAUD_TO_INTERPRET = 20
# Generic advert boilerplate, chosen in advance of looking at any clusters (plan section 4).
DOMAIN_STOP = ["responsibilities", "requirements", "qualifications", "candidate", "candidates",
               "position", "role", "apply", "looking", "company", "job", "benefits", "must",
               "required", "ability", "skills", "experience", "strong"]
SPECS = ["primary", "S1", "S2"]
VARIANTS = ["base", "unigram", "domainstop", "stem", "embed"]   # "embed" = R2, sentence embeddings
DEFAULT_ENCODER = "all-MiniLM-L6-v2"
EMBED_CHUNK_WORDS = 150       # long fields are split into chunks this long so nothing is truncated
EMBED_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
N_BALANCED_DRAWS = 5
HF_FILE = ROOT / "Tech-Job-Scams-and-Predatory-Recruitment" / "tech_job_scams_dataset.parquet"
SENSITIVITY_RUNS = [("S1", "base"), ("S2", "base"), ("primary", "unigram"),
                    ("primary", "domainstop"), ("primary", "stem")]


class Cfg:  # filled by configure()
    pass


CFG = Cfg()


def configure(smoke: bool):
    base_i, base_r = ROOT / "data" / "interim", ROOT / "results"
    CFG.smoke = smoke
    if smoke:
        CFG.interim, CFG.results = base_i / "_smoke", base_r / "_smoke"
        CFG.seeds, CFG.ks, CFG.svd, CFG.hdb, CFG.hdb_min_samples = [0, 1], list(range(5, 8)), 20, [10], 3
        CFG.sample_n, CFG.lda_offsets, CFG.sil_sample, CFG.min_df, CFG.boot = 800, [0], 300, 2, 100
        CFG.balanced_draws = 2
    else:
        CFG.balanced_draws = N_BALANCED_DRAWS
        CFG.interim, CFG.results = base_i, base_r
        CFG.seeds, CFG.ks, CFG.svd, CFG.hdb, CFG.hdb_min_samples = SEEDS, list(range(K_MIN, K_MAX + 1)), SVD_DIMS, HDB_GRID, HDB_MIN_SAMPLES
        CFG.sample_n, CFG.lda_offsets, CFG.sil_sample, CFG.min_df, CFG.boot = None, [-2, -1, 0, 1, 2], SIL_SAMPLE, 5, BOOT
    CFG.interim.mkdir(parents=True, exist_ok=True)
    CFG.results.mkdir(parents=True, exist_ok=True)


def say(msg):
    """Print results only in real runs. The smoke test never prints outcome numbers."""
    if not CFG.smoke:
        print(msg)


# ---- guard ----------------------------------------------------------------------------------
def check_lock():
    if not PLAN.exists():
        # the analysis plan is a private working document that is not part of the public repository
        print("Note: the analysis plan file was not found, so the lock check was skipped.")
        return "plan file not found"
    text = PLAN.read_text(encoding="utf-8")
    problems = []
    if re.search(r"Date locked:\s*_{3,}", text):
        problems.append('the "Date locked" line is still blank')
    if re.search(r"Your expectations:\*\s*_{3,}", text):
        problems.append('your expectations (section 9) are still blank')
    if problems:
        sys.exit("The analysis plan is not locked: " + "; ".join(problems) + ".\n"
                 f"Fill these in, in {PLAN.name}, before running the real analysis. "
                 "(`smoke` can be run at any time.)")
    m = re.search(r"Date locked:\s*([^\(\n*]+)", text)
    return m.group(1).strip() if m else ""


def log_run(command, params, seconds):
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "command": command, "params": params,
           "seconds": round(seconds, 1), "smoke": CFG.smoke,
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    with open(CFG.results / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---- data and features ----------------------------------------------------------------------
def load_data():
    df = pd.read_parquet(DATA)
    if CFG.sample_n:
        df = df.sample(CFG.sample_n, random_state=0)
    return df.reset_index(drop=True)


def get_text(df, spec):
    if spec == "primary":
        return df["text_all"]
    if spec == "S1":   # no company profile
        return (df["title"] + " " + df["description"] + " " + df["requirements"] + " " + df["benefits"]
                ).str.replace(r"\s+", " ", regex=True).str.strip()
    if spec == "S2":   # description only
        return df["description"]
    raise SystemExit(f"unknown spec {spec}")


def stop_words(variant):
    sw = set(ENGLISH_STOP_WORDS)
    if variant == "domainstop":
        sw |= set(DOMAIN_STOP)
    return sorted(sw)


def ngram_range(variant):
    return (1, 1) if variant == "unigram" else (1, 2)


def make_tfidf(variant):
    kw = dict(min_df=CFG.min_df, max_df=0.5, lowercase=True, norm="l2")
    if variant == "stem":
        try:
            from nltk.stem import PorterStemmer
        except ImportError:
            raise SystemExit("variant 'stem' needs nltk:  .venv\\Scripts\\python.exe -m pip install nltk")
        ps = PorterStemmer()
        base = TfidfVectorizer(stop_words=stop_words("base"), ngram_range=(1, 2)).build_analyzer()
        return TfidfVectorizer(analyzer=lambda d: [ps.stem(w) for w in base(d)], **kw)
    return TfidfVectorizer(stop_words=stop_words(variant), ngram_range=ngram_range(variant), **kw)


def tag(spec, variant):
    return f"{spec}_{variant}"


# ---- sentence embeddings (R2) ---------------------------------------------------------------
class StubEncoder:
    """Fake encoder used ONLY by the smoke test: hashed bag of words -> 32 numbers. Not a real model."""
    dim = 32

    def encode(self, texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        return normalize(out) if normalize_embeddings else out


def get_encoder(name):
    if name == "stub":
        if not CFG.smoke:
            raise SystemExit("the stub encoder is for the smoke test only")
        return StubEncoder()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("embeddings need sentence-transformers:\n"
                         "  .venv\\Scripts\\python.exe -m pip install sentence-transformers")
    local = ROOT / "models" / name      # optional local copy of the model, if downloading by name fails
    return SentenceTransformer(str(local) if local.exists() else name)


def chunk_words(text, size=EMBED_CHUNK_WORDS):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def embed_adverts(df, enc, batch=64):
    """One vector per advert: split fields longer than EMBED_CHUNK_WORDS into chunks, average the chunk
    vectors within each field, average the non-empty field vectors, then L2-normalise."""
    texts, owner = [], []
    for ai, row in enumerate(df[EMBED_FIELDS].itertuples(index=False)):
        for fi, val in enumerate(row):
            for ch in chunk_words(val):
                texts.append(ch)
                owner.append((ai, fi))
    E = np.asarray(enc.encode(texts, batch_size=batch, show_progress_bar=not CFG.smoke, normalize_embeddings=True),
                   dtype=np.float32)
    n, nf, dim = len(df), len(EMBED_FIELDS), E.shape[1]
    sums = np.zeros((n, nf, dim), dtype=np.float32)
    counts = np.zeros((n, nf), dtype=np.float32)
    for (ai, fi), e in zip(owner, E):
        sums[ai, fi] += e
        counts[ai, fi] += 1
    has = counts > 0
    field_vec = sums / np.maximum(counts, 1)[..., None]
    adv = (field_vec * has[..., None]).sum(axis=1) / np.maximum(has.sum(axis=1), 1)[:, None]
    return normalize(adv).astype(np.float32)


def build_features(df, spec="primary", variant="base", encoder=DEFAULT_ENCODER, refresh=False):
    t0 = time.time()
    t = tag(spec, variant)
    is_embed = variant == "embed"
    if is_embed and spec != "primary":
        raise SystemExit("the embedding representation (R2) is defined for the primary text only")
    # TF-IDF is always built: it is the representation (R1) or, for R2, the vocabulary used to name themes.
    vec = make_tfidf("base" if is_embed else variant)
    X = vec.fit_transform(get_text(df, spec)).tocsr()
    svd = None
    if is_embed:
        emb_path = CFG.interim / f"svd_{t}.npy"
        if emb_path.exists() and not refresh and np.load(emb_path).shape[0] == len(df):
            Z = np.load(emb_path)      # embedding is slow, so reuse it (use --refresh to redo)
            print("  reusing saved embeddings (use --refresh to recompute)")
        else:
            Z = embed_adverts(df, get_encoder(encoder))
    else:
        svd = TruncatedSVD(n_components=min(CFG.svd, X.shape[1] - 1), random_state=0)
        Z = normalize(svd.fit_transform(X)).astype(np.float32)
        try:
            joblib.dump({"vec": vec, "svd": svd}, CFG.interim / f"model_{t}.joblib")   # for the optional hf step
        except Exception:
            pass
    sp.save_npz(CFG.interim / f"tfidf_{t}.npz", X)
    np.save(CFG.interim / f"svd_{t}.npy", Z)
    json.dump(list(vec.get_feature_names_out()), open(CFG.interim / f"vocab_{t}.json", "w", encoding="utf-8"))
    df[["job_id", "fraudulent"]].to_parquet(CFG.interim / "ids.parquet", index=False)
    info = {"spec": spec, "variant": variant, "n_adverts": int(X.shape[0]), "n_terms": int(X.shape[1]),
            "sparsity_pct": round(100 * (1 - X.nnz / (X.shape[0] * X.shape[1])), 2),
            "vector_dims": int(Z.shape[1]),
            "svd_variance_explained": None if svd is None else round(float(svd.explained_variance_ratio_.sum()), 3),
            "encoder": encoder if is_embed else None,
            "adverts_with_no_terms": int((X.getnnz(axis=1) == 0).sum())}
    (CFG.results / f"features_info_{t}.json").write_text(json.dumps(info, indent=2))
    log_run("features", info, time.time() - t0)
    print(f"features {t}: {info['n_adverts']} adverts, {info['n_terms']} terms, "
          f"{info['sparsity_pct']}% sparse, SVD variance {info['svd_variance_explained']}, "
          f"{info['adverts_with_no_terms']} with no terms")


def load_features(spec, variant):
    t = tag(spec, variant)
    Z = np.load(CFG.interim / f"svd_{t}.npy")
    X = sp.load_npz(CFG.interim / f"tfidf_{t}.npz").tocsr()
    vocab = json.load(open(CFG.interim / f"vocab_{t}.json", encoding="utf-8"))
    ids = pd.read_parquet(CFG.interim / "ids.parquet")
    return Z, X, np.array(vocab), ids


# ---- choosing k -----------------------------------------------------------------------------
def kselect(spec="primary", variant="base"):
    t0 = time.time()
    Z, _, _, ids = load_features(spec, variant)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(Z), size=min(CFG.sil_sample, len(Z)), replace=False)
    rows = []
    for k in CFG.ks:
        labs, wss = [], []
        for s in CFG.seeds:
            km = KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z)
            labs.append(km.labels_)
            wss.append(km.inertia_)
        aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labs, 2)]
        nmis = [normalized_mutual_info_score(a, b) for a, b in itertools.combinations(labs, 2)]
        sils = [silhouette_score(Z[idx], l[idx]) for l in labs]
        sizes = np.bincount(labs[0], minlength=k)
        rows.append({"k": k, "wss_mean": float(np.mean(wss)), "silhouette_mean": float(np.mean(sils)),
                     "stability_ari_mean": float(np.mean(aris)), "stability_ari_min": float(np.min(aris)),
                     "stability_nmi_mean": float(np.mean(nmis)),
                     "smallest_cluster_seed0": int(sizes.min()), "largest_cluster_seed0": int(sizes.max())})
        say(f"  k={k:2d}  silhouette {rows[-1]['silhouette_mean']:.3f}  stability ARI {rows[-1]['stability_ari_mean']:.3f}")
    res = pd.DataFrame(rows)
    stable = res[res["stability_ari_mean"] >= STABILITY_MIN]
    if len(stable):
        k_best = int(stable.loc[stable["silhouette_mean"].idxmax(), "k"])
        rule = f"highest silhouette among k with mean stability ARI >= {STABILITY_MIN}"
    else:
        k_best = int(res.loc[res["stability_ari_mean"].idxmax(), "k"])
        rule = f"no k reached stability {STABILITY_MIN}; took the k with the highest stability (plan fallback)"
    t = tag(spec, variant)
    res.to_csv(CFG.results / f"kselect_{t}.csv", index=False)
    (CFG.results / f"chosen_k_{t}.json").write_text(json.dumps({"k": k_best, "rule": rule}, indent=2))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, col, title in zip(ax, ["wss_mean", "silhouette_mean", "stability_ari_mean"],
                             ["Within-cluster sum of squares (elbow)", "Average silhouette width", "Stability across seeds (ARI)"]):
        a.plot(res["k"], res[col], marker="o")
        a.axvline(k_best, color="red", linestyle="--", linewidth=1)
        a.set_title(title, fontsize=10)
        a.set_xlabel("k")
    ax[2].axhline(STABILITY_MIN, color="grey", linestyle=":")
    fig.tight_layout()
    fig.savefig(CFG.results / f"kselect_{t}.png", dpi=150)
    plt.close(fig)
    log_run("kselect", {"spec": spec, "variant": variant, "k": k_best, "rule": rule}, time.time() - t0)
    say(f"chosen k = {k_best}  ({rule})")
    return k_best


# ---- comparison tables ----------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def theme_table(labels, y, name):
    """Per-theme counts, fraud share with bootstrap CI, exact test vs overall rate (Holm), chi-square."""
    labels, y = np.asarray(labels), np.asarray(y)
    p0 = y.mean()
    rng = np.random.RandomState(0)
    rows = []
    for c in sorted(set(labels)):
        yc = y[labels == c]
        n, nf = len(yc), int(yc.sum())
        boot = rng.choice(yc, size=(CFG.boot, n)).mean(axis=1)
        rows.append({"theme": c, "n_adverts": n, "n_fraudulent": nf, "fraud_share": nf / n,
                     "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                     "p_vs_overall": float(stats.binomtest(nf, n, p0).pvalue)})
    tab = pd.DataFrame(rows)
    tab["p_holm"] = holm(tab["p_vs_overall"])
    tab["overall_fraud_share"] = p0
    tab["direction"] = np.where(tab["p_holm"] < 0.05, np.where(tab["fraud_share"] > p0, "fraud-heavy", "legitimate-heavy"), "not different")
    tab["interpret"] = tab["n_fraudulent"] >= MIN_FRAUD_TO_INTERPRET
    ct = pd.crosstab(labels, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
    summary = {"name": name, "n": int(len(y)), "themes": int(ct.shape[0]), "chi2": float(chi2), "dof": int(dof),
               "p": float(p), "cramers_v": v, "overall_fraud_share": float(p0)}
    return tab, summary


# ---- final K-Means, themes, comparison ------------------------------------------------------
def kmeans_final(spec="primary", variant="base", k=None):
    t0 = time.time()
    t = tag(spec, variant)
    if k is None:
        k = json.load(open(CFG.results / f"chosen_k_{t}.json"))["k"]
    Z, X, vocab, ids = load_features(spec, variant)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    labels = km.labels_
    np.save(CFG.interim / f"centroids_kmeans_{t}.npy", km.cluster_centers_)
    out = ids.copy()
    out["cluster"] = labels
    out.to_parquet(CFG.interim / f"labels_kmeans_{t}.parquet", index=False)

    # top terms per cluster: highest mean TF-IDF, and most over-represented against the other clusters
    n = X.shape[0]
    tot = np.asarray(X.sum(axis=0)).ravel()
    rows = []
    for c in range(k):
        m = labels == c
        sc = np.asarray(X[m].sum(axis=0)).ravel()
        mean_c, mean_rest = sc / m.sum(), (tot - sc) / max(n - m.sum(), 1)
        top = np.argsort(-mean_c)[:15]
        over = np.argsort(-(mean_c - mean_rest))[:15]
        for r in range(15):
            rows.append({"cluster": c, "n_adverts": int(m.sum()), "rank": r + 1,
                         "top_mean_tfidf_term": vocab[top[r]], "over_represented_term": vocab[over[r]]})
    pd.DataFrame(rows).to_csv(CFG.results / f"themes_terms_{t}.csv", index=False)

    # blind reading sample (no fraud label shown) and the second-coder sheet
    full = pd.read_parquet(DATA).set_index("job_id")
    rs = []
    for c in range(k):
        jid = out.loc[out.cluster == c, "job_id"]
        pick = jid.sample(min(N_READ, len(jid)), random_state=0)
        rs += [{"cluster": c, "job_id": j, "text": full.loc[j, "text_all"]} for j in pick]
    pd.DataFrame(rs).to_csv(CFG.interim / f"reading_sample_{t}.csv", index=False)
    pool = out["job_id"].sample(min(N_SECOND_CODER, len(out)), random_state=1)
    pd.DataFrame({"job_id": pool, "text": [full.loc[j, "text_all"] for j in pool], "theme_label": ""}
                 ).to_csv(CFG.interim / f"second_coder_sheet_{t}.csv", index=False)
    out[out.job_id.isin(pool)][["job_id", "cluster"]].to_csv(CFG.interim / f"second_coder_key_{t}.csv", index=False)
    pd.DataFrame({"cluster": range(k), "label": "", "definition": "", "includes": "", "excludes": ""}
                 ).to_csv(CFG.results / f"codebook_template_{t}.csv", index=False)

    tab, summ = theme_table(labels, ids["fraudulent"].values, f"kmeans_{t}_k{k}")
    tab.to_csv(CFG.results / f"theme_comparison_kmeans_{t}.csv", index=False)
    (CFG.results / f"theme_comparison_kmeans_{t}.json").write_text(json.dumps(summ, indent=2))
    log_run("kmeans", {"spec": spec, "variant": variant, "k": k, **{x: summ[x] for x in ("chi2", "p", "cramers_v")}}, time.time() - t0)
    say(f"kmeans {t} k={k}: chi2={summ['chi2']:.1f}, p={summ['p']:.3g}, Cramer's V={summ['cramers_v']:.3f}; "
        f"{int((tab['direction'] == 'fraud-heavy').sum())} fraud-heavy, {int((tab['direction'] == 'legitimate-heavy').sum())} legitimate-heavy themes")
    return labels


# ---- HDBSCAN --------------------------------------------------------------------------------
def hdbscan_run(spec="primary", variant="base"):
    t0 = time.time()
    t = tag(spec, variant)
    Z, _, _, ids = load_features(spec, variant)
    y = ids["fraudulent"].values
    km_path = CFG.interim / f"labels_kmeans_{t}.parquet"
    km_lab = pd.read_parquet(km_path)["cluster"].values if km_path.exists() else None
    rows = []
    for mcs in CFG.hdb:
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=CFG.hdb_min_samples).fit(Z).labels_
        ok = lab >= 0
        ncl = len(set(lab[ok]))
        row = {"min_cluster_size": mcs, "n_clusters": ncl, "unclustered_share": float((~ok).mean())}
        if km_lab is not None and ok.sum() > 1:
            row["ari_vs_kmeans_clustered_only"] = float(adjusted_rand_score(km_lab[ok], lab[ok]))
        if ncl >= 2:
            tab, summ = theme_table(lab[ok], y[ok], f"hdbscan_{t}_mcs{mcs}")
            tab.to_csv(CFG.results / f"theme_comparison_hdbscan_{t}_mcs{mcs}.csv", index=False)
            row.update({"chi2": summ["chi2"], "p": summ["p"], "cramers_v": summ["cramers_v"]})
        pd.DataFrame({"job_id": ids["job_id"], "cluster": lab}).to_parquet(CFG.interim / f"labels_hdbscan_{t}_mcs{mcs}.parquet", index=False)
        rows.append(row)
        say(f"  min_cluster_size={mcs}: {ncl} clusters, {row['unclustered_share']:.0%} unclustered")
    pd.DataFrame(rows).to_csv(CFG.results / f"hdbscan_summary_{t}.csv", index=False)
    log_run("hdbscan", {"spec": spec, "variant": variant, "grid": CFG.hdb}, time.time() - t0)


# ---- LDA ------------------------------------------------------------------------------------
def fit_lda(C, K):
    """Fit LDA on all cores; if a worker process crashes (this can happen on Windows), retry on one core.
    The random seed is fixed, so the result does not depend on the number of cores."""
    for jobs in (-1, 1):
        try:
            lda = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=15,
                                            random_state=0, n_jobs=jobs)
            return lda, lda.fit_transform(C)
        except Exception as e:
            if jobs == 1:
                raise
            print(f"  note: LDA on all cores failed ({type(e).__name__}); retrying on one core")


def lda_run(spec="primary", variant="base"):
    if variant == "embed":
        raise SystemExit("LDA works on word counts; run it with --variant base")
    t0 = time.time()
    t = tag(spec, variant)
    _, _, vocab, ids = load_features(spec, variant)
    df = load_data()
    cv = CountVectorizer(vocabulary=list(vocab), stop_words=stop_words(variant), ngram_range=ngram_range(variant), lowercase=True)
    C = cv.transform(get_text(df, spec))
    k0 = json.load(open(CFG.results / f"chosen_k_{t}.json"))["k"]
    km_path = CFG.interim / f"labels_kmeans_{t}.parquet"
    km_lab = pd.read_parquet(km_path)["cluster"].values if km_path.exists() else None
    for off in CFG.lda_offsets:
        K = k0 + off
        if K < 2:
            continue
        lda, theta = fit_lda(C, K)
        dom = theta.argmax(axis=1)
        rows = []
        for tpc in range(K):
            top = np.argsort(-lda.components_[tpc])[:15]
            rows += [{"topic": tpc, "rank": r + 1, "term": vocab[top[r]]} for r in range(15)]
        pd.DataFrame(rows).to_csv(CFG.results / f"lda_terms_{t}_K{K}.csv", index=False)
        tab, summ = theme_table(dom, ids["fraudulent"].values, f"lda_{t}_K{K}")
        tab.to_csv(CFG.results / f"theme_comparison_lda_{t}_K{K}.csv", index=False)
        extra = f", ARI vs K-Means {adjusted_rand_score(km_lab, dom):.3f}" if (km_lab is not None and K == k0) else ""
        say(f"  LDA K={K}: chi2={summ['chi2']:.1f}, p={summ['p']:.3g}, Cramer's V={summ['cramers_v']:.3f}{extra}")
        log_run("lda", {"spec": spec, "variant": variant, "K": K, "max_iter": 15}, time.time() - t0)


# ---- sensitivity and external check ---------------------------------------------------------
def sensitivity(df=None):
    t0 = time.time()
    k = json.load(open(CFG.results / "chosen_k_primary_base.json"))["k"]
    prim = pd.read_parquet(CFG.interim / "labels_kmeans_primary_base.parquet")
    df = load_data() if df is None else df
    rows = []
    for spec, variant in SENSITIVITY_RUNS:
        t = tag(spec, variant)
        if variant == "stem":
            try:
                import nltk  # noqa: F401
            except ImportError:
                print(f"  skipped {t}: nltk not installed")
                continue
        if not (CFG.interim / f"svd_{t}.npy").exists():
            build_features(df, spec, variant)
        kmeans_final(spec, variant, k=k)
        lab = pd.read_parquet(CFG.interim / f"labels_kmeans_{t}.parquet")
        m = prim.merge(lab, on=["job_id", "fraudulent"], suffixes=("_primary", "_alt"))
        summ = json.load(open(CFG.results / f"theme_comparison_kmeans_{t}.json"))
        rows.append({"run": t, "k": k, "ari_vs_primary": adjusted_rand_score(m["cluster_primary"], m["cluster_alt"]),
                     "nmi_vs_primary": normalized_mutual_info_score(m["cluster_primary"], m["cluster_alt"]),
                     "chi2": summ["chi2"], "cramers_v": summ["cramers_v"]})
    pd.DataFrame(rows).to_csv(CFG.results / "sensitivity_summary.csv", index=False)

    # external check: agreement with EMSCAD's own `function` and `industry` columns (primary run)
    meta = df[["job_id", "function", "industry"]].merge(prim[["job_id", "cluster"]], on="job_id")
    ext = {}
    for col in ("function", "industry"):
        sub = meta.dropna(subset=[col])
        ext[col] = {"n": int(len(sub)), "nmi": float(normalized_mutual_info_score(sub[col], sub["cluster"]))}
        if col == "function":
            pd.crosstab(sub[col], sub["cluster"]).to_csv(CFG.results / "crosstab_function_vs_cluster.csv")
    (CFG.results / "external_check.json").write_text(json.dumps(ext, indent=2))
    log_run("sensitivity", {"k": k, "runs": [r["run"] for r in rows], "external": ext}, time.time() - t0)
    say("sensitivity:\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    say("external check (NMI with EMSCAD columns): " + str({c: round(v["nmi"], 3) for c, v in ext.items()}))


# ---- class-balance check (plan section 8) ---------------------------------------------------
def balanced_check(spec="primary", variant="base"):
    """Refit K-Means at the chosen k on balanced samples (all fraudulent adverts + an equal number of randomly
    chosen legitimate ones) and compare with the primary run."""
    t0 = time.time()
    t = tag(spec, variant)
    Z, _, _, ids = load_features(spec, variant)
    k = json.load(open(CFG.results / f"chosen_k_{t}.json"))["k"]
    prim = pd.read_parquet(CFG.interim / f"labels_kmeans_{t}.parquet")["cluster"].values
    prim_dir = pd.read_csv(CFG.results / f"theme_comparison_kmeans_{t}.csv").set_index("theme")["direction"].to_dict()
    y = ids["fraudulent"].values
    fraud_idx, legit_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    runs = []
    rows = {c: {"theme": c, "primary_direction": prim_dir.get(c)} for c in range(k)}
    for s in range(CFG.balanced_draws):
        rng = np.random.RandomState(s)
        idx = np.concatenate([fraud_idx, rng.choice(legit_idx, size=len(fraud_idx), replace=False)])
        bl = KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z[idx]).labels_
        runs.append({"seed": s, "n_adverts": int(len(idx)),
                     "ari_vs_primary": float(adjusted_rand_score(prim[idx], bl)),
                     "nmi_vs_primary": float(normalized_mutual_info_score(prim[idx], bl))})
        tab, _ = theme_table(bl, y[idx], f"balanced_seed{s}")
        bdir = tab.set_index("theme")["direction"].to_dict()
        ct = pd.crosstab(prim[idx], bl)           # primary themes (rows) against balanced clusters (columns)
        for c in range(k):
            if c in ct.index:
                best = ct.loc[c].idxmax()          # balanced cluster that overlaps most with primary theme c
                rows[c][f"balanced_seed{s}"] = bdir.get(best)
            else:
                rows[c][f"balanced_seed{s}"] = None
    themes = pd.DataFrame(list(rows.values()))
    cols = [c for c in themes.columns if c.startswith("balanced_seed")]
    themes["seeds_same_direction"] = themes.apply(lambda r: int(sum(r[c] == r["primary_direction"] for c in cols)), axis=1)
    pd.DataFrame(runs).to_csv(CFG.results / f"balanced_check_runs_{t}.csv", index=False)
    themes.to_csv(CFG.results / f"balanced_check_themes_{t}.csv", index=False)
    log_run("balanced", {"spec": spec, "variant": variant, "k": k, "draws": CFG.balanced_draws}, time.time() - t0)
    say("balanced check:\n" + pd.DataFrame(runs).round(3).to_string(index=False))
    say(f"themes whose direction (fraud-heavy / legitimate-heavy / not different) is the same in all "
        f"{CFG.balanced_draws} draws: {int((themes['seeds_same_direction'] == CFG.balanced_draws).sum())} of {k}")


# ---- optional supplementary step: Hugging Face scam messages (plan section 10b) -------------
def hf_assign(spec="primary", variant="base"):
    """Assign each Hugging Face message to the nearest EMSCAD theme. Descriptive only. Needs approval (see plan 10b)."""
    t0 = time.time()
    if variant == "embed":
        raise SystemExit("the Hugging Face step uses the TF-IDF representation (R1): run it with --variant base")
    t = tag(spec, variant)
    model = joblib.load(CFG.interim / f"model_{t}.joblib")
    vec, svd = model["vec"], model["svd"]
    C = np.load(CFG.interim / f"centroids_kmeans_{t}.npy")
    Zp, _, _, _ = load_features(spec, variant)
    lab = pd.read_parquet(CFG.interim / f"labels_kmeans_{t}.parquet")["cluster"].values
    d_own = np.linalg.norm(Zp - C[lab], axis=1)
    thr = np.array([np.percentile(d_own[lab == c], 95) if (lab == c).any() else np.inf for c in range(len(C))])
    hf = pd.read_parquet(HF_FILE)
    H = vec.transform(hf["text_sample"])
    no_terms = H.getnnz(axis=1) == 0
    Zh = normalize(svd.transform(H)).astype(np.float32)
    D = np.linalg.norm(Zh[:, None, :] - C[None, :, :], axis=2)
    near, dist = D.argmin(axis=1), D.min(axis=1)
    theme = np.where(no_terms, -1, near)                       # -1: no word in common with the EMSCAD vocabulary
    outside = (~no_terms) & (dist > thr[near])                 # farther than 95% of that theme's own adverts
    out = pd.DataFrame({"scam_type": hf["scam_type"].values, "theme": theme, "outside": outside, "text": hf["text_sample"].values})
    dist_df = out.drop_duplicates("text")
    pd.crosstab(out["theme"], out["scam_type"]).to_csv(CFG.results / f"hf_theme_by_scam_type_all_rows_{t}.csv")
    pd.crosstab(dist_df["theme"], dist_df["scam_type"]).to_csv(CFG.results / f"hf_theme_by_scam_type_distinct_{t}.csv")
    summary = {"rows": int(len(out)), "distinct_texts": int(len(dist_df)),
               "no_shared_vocabulary_share_all": float((out["theme"] == -1).mean()),
               "outside_every_theme_share_all": float(out["outside"].mean()),
               "outside_every_theme_share_distinct": float(dist_df["outside"].mean()),
               "note": "descriptive only; 1,000 rows come from about 217 distinct texts built from 3 templates"}
    (CFG.results / f"hf_assignment_summary_{t}.json").write_text(json.dumps(summary, indent=2))
    log_run("hf", {"spec": spec, "variant": variant, **summary}, time.time() - t0)
    say("hf assignment: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k != "note"}))


# ---- smoke test -----------------------------------------------------------------------------
def smoke():
    """Run the whole pipeline on a small random subsample to test the code. Prints no themes."""
    configure(True)
    shutil.rmtree(CFG.interim, ignore_errors=True)
    shutil.rmtree(CFG.results, ignore_errors=True)
    configure(True)
    df = load_data()
    print(f"SMOKE TEST on {len(df)} random adverts. No themes or outcome statistics are printed.")
    build_features(df, "primary", "base")
    print("  ok: features")
    kselect("primary", "base")
    print("  ok: kselect")
    kmeans_final("primary", "base")
    print("  ok: kmeans")
    hdbscan_run("primary", "base")
    print("  ok: hdbscan")
    lda_run("primary", "base")
    print("  ok: lda")
    global SENSITIVITY_RUNS
    SENSITIVITY_RUNS = [("S1", "base"), ("S2", "base"), ("primary", "unigram"), ("primary", "domainstop")]
    sensitivity(df)
    print("  ok: sensitivity")
    balanced_check("primary", "base")
    print("  ok: balanced check")
    hf_assign("primary", "base")
    print("  ok: hugging face assignment")
    # R2 (sentence embeddings), tested with a stand-in encoder because the real model is not part of the test
    build_features(df, "primary", "embed", encoder="stub")
    print("  ok: embed")
    kselect("primary", "embed")
    kmeans_final("primary", "embed")
    hdbscan_run("primary", "embed")
    print("  ok: R2 kselect, kmeans, hdbscan")
    expected = ["kselect_primary_base.csv", "chosen_k_primary_base.json", "themes_terms_primary_base.csv",
                "theme_comparison_kmeans_primary_base.csv", "hdbscan_summary_primary_base.csv",
                "sensitivity_summary.csv", "external_check.json", "run_log.jsonl",
                "balanced_check_runs_primary_base.csv", "balanced_check_themes_primary_base.csv",
                "hf_assignment_summary_primary_base.json", "kselect_primary_embed.csv",
                "theme_comparison_kmeans_primary_embed.csv", "hdbscan_summary_primary_embed.csv"]
    missing = [f for f in expected if not (CFG.results / f).exists()]
    print("SMOKE TEST", "FAILED, missing: " + ", ".join(missing) if missing else "OK: all steps ran and wrote their outputs")
    return not missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["smoke", "features", "embed", "kselect", "kmeans", "hdbscan", "lda",
                                        "sensitivity", "balanced", "hf"])
    ap.add_argument("--spec", default="primary", choices=SPECS)
    ap.add_argument("--variant", default="base", choices=VARIANTS)
    ap.add_argument("--k", type=int, default=None, help="kmeans only: override the chosen k (log the reason in the plan)")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, help="embed only: sentence-transformers model name")
    ap.add_argument("--refresh", action="store_true", help="embed only: recompute the saved embeddings")
    ap.add_argument("--hf-approved", action="store_true",
                    help="hf only: confirms my supervisor wants this and the ethics check allows it (plan section 10b)")
    a = ap.parse_args()

    if a.command == "smoke":
        sys.exit(0 if smoke() else 1)

    lock = check_lock()
    configure(False)
    print(f"Plan locked on: {lock}")
    if a.command == "features":
        build_features(load_data(), a.spec, a.variant, a.encoder, a.refresh)
    elif a.command == "embed":
        build_features(load_data(), "primary", "embed", a.encoder, a.refresh)
    elif a.command == "balanced":
        balanced_check(a.spec, a.variant)
    elif a.command == "hf":
        if not a.hf_approved:
            sys.exit("The Hugging Face step is optional and needs two confirmations first (plan section 10b): "
                     "my supervisor wants it, and the ethics check allows it. If both are true, add --hf-approved.")
        hf_assign(a.spec, a.variant)
    elif a.command == "kselect":
        kselect(a.spec, a.variant)
    elif a.command == "kmeans":
        kmeans_final(a.spec, a.variant, a.k)
    elif a.command == "hdbscan":
        hdbscan_run(a.spec, a.variant)
    elif a.command == "lda":
        lda_run(a.spec, a.variant)
    elif a.command == "sensitivity":
        sensitivity()


if __name__ == "__main__":
    main()
