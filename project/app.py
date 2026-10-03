"""
Streamlit app: AE-based damage characterisation of GFRP composites,
with a general "Train on Your Own Data" module for any AE dataset.
Run:  streamlit run app.py      (run ae_clustering_analysis.py once first)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import io
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, f1_score

BASE = Path(__file__).parent
FIG, RES = BASE / "figures", BASE / "results"
MODEL_PATH = BASE / "models" / "ae_cluster_model.joblib"
COLORS = {"LE-Tensile": "#2a78d6", "HE-Tensile": "#eb6834", "LE-Shear": "#1baf7a", "HE-Shear": "#eda100"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASE_COLS = ["Amplitude", "Energy", "Duration", "Counts", "Risetime"]
FEATURES = BASE_COLS + ["RA", "AF"]
LOG_FEATURES = ["Energy", "Duration", "Counts", "Risetime", "RA", "AF"]
ALIASES = {"Amplitude": ["amplitude", "amp", "ampl", "a(db)", "amp(db)"],
           "Energy": ["energy", "ener", "e", "absenergy", "abs energy"],
           "Duration": ["duration", "dur", "d"],
           "Counts": ["counts", "count", "cnts", "coun"],
           "Risetime": ["risetime", "rise time", "rise", "r"]}

st.set_page_config(page_title="GFRP AE Damage Characterisation", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def add_ra_af(df, amp_in_db=True):
    df = df.copy()
    amp_v = 1e-6 * 10 ** (df["Amplitude"] / 20) if amp_in_db else df["Amplitude"]
    df["RA"] = (df["Risetime"] / 1000) / amp_v
    df["AF"] = df["Counts"] / df["Duration"] * 1000
    return df


def apply_map(df, column_map):
    """Copy the user's columns onto the standard AE parameter names."""
    if not column_map:
        return df
    values = {target: df[source].values for source, target in column_map.items()}
    out = df.drop(columns=[c for c in BASE_COLS if c in df.columns])
    for target, v in values.items():
        out[target] = v
    return out


def predict(df, bundle):
    df = apply_map(df, bundle.get("column_map", {}))
    df = add_ra_af(df, bundle.get("amp_in_db", True))
    X = df[bundle["features"]].apply(pd.to_numeric, errors="coerce").fillna(df[bundle["features"]].median())
    proba = bundle["model"].predict_proba(X)
    df["Predicted mechanism"] = bundle["model"].classes_[proba.argmax(1)]
    df["Confidence"] = proba.max(1).round(3)
    df["Status"] = np.where(df["Confidence"] >= bundle["conf_threshold"], "Confident", "Uncertain")
    return df


def show(fig, caption):
    p = FIG / fig
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


# ----------------------------------------------------------------------
# Train on your own data: general pipeline for any AE dataset
# ----------------------------------------------------------------------
def guess_column(columns, target):
    """Find the column that most likely holds a given AE parameter."""
    low = {c: str(c).strip().lower().replace("_", " ") for c in columns}
    for alias in ALIASES[target]:
        for c, name in low.items():
            if name == alias:
                return c
    for c, name in low.items():
        if name.replace(" ", "").startswith(target.lower()[:4]):
            return c
    return None


def prepare(df, amp_in_db=True):
    """Keep valid hits, add RA and AF, return the cleaned table and the scaled features."""
    d = df.copy()
    d[BASE_COLS] = d[BASE_COLS].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=BASE_COLS)
    d = d[(d["Duration"] > 0) & (d["Counts"] > 0) & (d["Risetime"] >= 0) & (d["Energy"] >= 0)]
    if not amp_in_db:
        d = d[d["Amplitude"] > 0]
    d = add_ra_af(d, amp_in_db).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES).reset_index(drop=True)
    Z = pd.DataFrame({c: np.log1p(d[c]) if c in LOG_FEATURES else d[c] for c in FEATURES})
    return d, Z


def score_k(X, ks=range(2, 9), seed=42):
    rows = []
    sub = X if len(X) <= 20000 else X[np.random.default_rng(seed).choice(len(X), 20000, replace=False)]
    for k in ks:
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(sub)
        rows.append({"k": k, "Silhouette": silhouette_score(sub, lab, sample_size=min(5000, len(sub)), random_state=0),
                     "Davies-Bouldin": davies_bouldin_score(sub, lab)})
    return pd.DataFrame(rows)


def pick_k(scores):
    """First local minimum of the Davies-Bouldin index for k >= 3 (k = 2 only splits weak and strong hits)."""
    db = scores.set_index("k")["Davies-Bouldin"]
    ks = [k for k in db.index if k >= 3]
    for i, k in enumerate(ks):
        nxt = ks[i + 1] if i + 1 < len(ks) else None
        if (i == 0 or db[k] <= db[ks[i - 1]]) and (nxt is None or db[k] <= db[nxt]):
            return int(k)
    return int(db.loc[ks].idxmin())


def name_clusters_general(centres):
    """Name clusters from centroid physics: RA vs AF gives tensile or shear, energy gives the level."""
    shear_score = centres["RA"] - centres["AF"]
    is_shear = shear_score > shear_score.median()
    if is_shear.nunique() == 1:                        # all on one side: split at the top half by rank
        is_shear = shear_score.rank() > len(centres) / 2
    names = {}
    for flag, kind in ((False, "Tensile"), (True, "Shear")):
        idx = centres.index[is_shear == flag]
        order = centres.loc[idx, "Energy"].sort_values().index
        tags = {1: [""], 2: ["LE-", "HE-"], 3: ["LE-", "ME-", "HE-"]}.get(len(order), [f"E{i + 1}-" for i in range(len(order))])
        for c, tag in zip(order, tags):
            names[c] = f"{tag}{kind}"
    return names


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes, file_name, column_map, amp_in_db, k_choice, seed=42):
    raw = pd.read_csv(io.BytesIO(file_bytes)) if file_name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(file_bytes))
    d, Z = prepare(apply_map(raw, column_map), amp_in_db)
    scaler = StandardScaler().fit(Z)
    X = scaler.transform(Z)
    scores = score_k(X, seed=seed)
    k = pick_k(scores) if k_choice == 0 else k_choice
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    centres = pd.DataFrame(km.cluster_centers_, columns=FEATURES)      # standardised log space
    names = name_clusters_general(centres)
    d["Mechanism"] = pd.Series(km.labels_).map(names).values
    level = lambda n: {"LE": 0, "ME": 1, "HE": 2}.get(n.split("-")[0], int(n.split("-")[0][1:]) if n[0] == "E" else 0)
    order = sorted(names.values(), key=lambda n: ("Shear" in n, level(n)))
    quality = {"k": k, "silhouette": silhouette_score(X, km.labels_, sample_size=min(5000, len(X)), random_state=0),
               "davies_bouldin": davies_bouldin_score(X, km.labels_), "dropped": len(raw) - len(d)}
    # classifier for instant prediction on further files
    tr, te = train_test_split(np.arange(len(d)), test_size=0.10, stratify=d["Mechanism"], random_state=seed)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced")
    rf.fit(d[FEATURES].iloc[tr], d["Mechanism"].iloc[tr])
    pred = rf.predict(d[FEATURES].iloc[te])
    quality["accuracy"] = accuracy_score(d["Mechanism"].iloc[te], pred)
    quality["macro_f1"] = f1_score(d["Mechanism"].iloc[te], pred, average="macro")
    quality["train"], quality["test"] = len(tr), len(te)
    new_bundle = {"model": rf, "features": FEATURES, "classes": order, "conf_threshold": 0.70,
                  "descriptions": {n: n for n in order}, "column_map": column_map, "amp_in_db": amp_in_db,
                  "source": file_name}
    return d, scores, quality, order, new_bundle


def ra_af_figure(d, order):
    s = d.sample(min(6000, len(d)), random_state=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, name in enumerate(order):
        p = s[s["Mechanism"] == name]
        ax.scatter(p["RA"], p["AF"], s=9, alpha=0.55, color=PALETTE[i % len(PALETTE)], edgecolors="none", label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("RA value (ms/V, log scale)"); ax.set_ylabel("Average frequency AF (kHz, log scale)")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#e6e5e0", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(markerscale=2.5, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


bundle = load_model()
st.sidebar.title("GFRP AE Damage")
page = st.sidebar.radio("Module", ["Home", "Damage Clustering", "Model Validation",
                                    "Source Study Comparison", "Predict New AE Data",
                                    "Train on Your Own Data"])
if bundle is None:
    st.error("Model not found. Run `python ae_clustering_analysis.py` first.")
    st.stop()

if page == "Home":
    st.title("Damage Characterisation of GFRP Composites under Impact and Compression")
    st.caption("Acoustic Emission and Data-Driven Analysis")
    c = st.columns(4)
    c[0].metric("AE hits", "30,574")
    c[1].metric("Specimens", "8")
    c[2].metric("Damage mechanisms found", "4")
    c[3].metric("RF accuracy (90/10)", "98.1%")
    st.markdown("""
**Workflow:** AE dataset → RA/AF features → unsupervised K-means clustering (no labels)
→ physical validation (RA–AF, loading history, impact effect) → Random Forest for instant classification
of new AE files, benchmarked against kNN, with prediction confidence.
""")
    prof = pd.read_csv(RES / "03_cluster_profiles.csv")
    st.subheader("Identified damage mechanisms")
    st.dataframe(prof[["cluster", "hits", "share_%", "Amplitude", "Energy", "RA", "AF", "interpretation"]],
                 hide_index=True, width="stretch")

elif page == "Damage Clustering":
    st.title("Unsupervised Damage Clustering")
    c1, c2 = st.columns(2)
    with c1: show("01_cluster_number_selection.png", "Number of clusters selected by Davies–Bouldin index")
    with c2: show("02_pca_clusters.png", "Four stable clusters (PCA view)")
    c1, c2 = st.columns(2)
    with c1: show("03_ra_af_classification.png", "RA–AF: tensile vs shear mechanisms")
    with c2: show("04_damage_progression.png", "Shear damage grows late in loading")
    c1, c2 = st.columns(2)
    with c1: show("05_cai_vs_pure_compression.png", "Impacted specimens show more delamination-type damage")
    with c2: show("07_gap_band_hits_assigned.png", "Hits outside manual frequency bands are now classified")

elif page == "Model Validation":
    st.title("Random Forest Model Validation")
    st.dataframe(pd.read_csv(RES / "10_rf_vs_knn_summary.csv"), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1: show("10_rf_vs_knn_benchmark.png", "Random Forest vs kNN")
    with c2: show("09_rf_confusion_matrix.png", "Confusion matrix, 10% test set")
    c1, c2 = st.columns(2)
    with c1: show("08_rf_feature_importance.png", "Feature importance")
    with c2: show("11_prediction_confidence.png", "Prediction confidence")

elif page == "Source Study Comparison":
    st.title("Comparison with Source Study [1]")
    show("12_source_study_comparison.png", "Delamination-type energy share by cenosphere content")
    st.markdown("""
- **Agrees:** impacted specimens show more than twice the delamination-type activity (14.2% vs 6.0% of hits).
- **Agrees:** non-impacted, 3 wt% vs 0 wt%, delamination-type energy share falls by 54% (paper: 41.17% reduction).
- **Not reproduced:** impacted 3 wt% specimen, attributed to one specimen per condition and 7× more recorded hits.
""")

elif page == "Train on Your Own Data":
    st.title("Train on Your Own AE Data")
    st.markdown("Upload an AE file from **any material or test**. The app groups the hits into damage mechanisms "
                "without labels, names the groups from their RA–AF and energy behaviour, and builds a classifier "
                "for that material.")
    up = st.file_uploader("AE file (CSV/XLSX), one row per hit", type=["csv", "xlsx", "xls"], key="own")
    if up is not None:
        data = up.getvalue()
        head = pd.read_csv(io.BytesIO(data), nrows=5) if up.name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(data), nrows=5)
        cols = list(head.columns)
        st.subheader("1. Match your columns")
        c = st.columns(5)
        chosen = {}
        for i, target in enumerate(BASE_COLS):
            g = guess_column(cols, target)
            chosen[target] = c[i].selectbox(target, cols, index=cols.index(g) if g in cols else 0, key=f"map_{target}")
        c1, c2, c3 = st.columns(3)
        amp_in_db = c1.radio("Amplitude unit", ["dB", "Volts"], horizontal=True) == "dB"
        k_opt = c2.selectbox("Number of clusters", ["Auto (Davies–Bouldin)"] + [str(k) for k in range(2, 9)])
        time_col = c3.selectbox("Time column (optional)", ["None"] + cols,
                                index=(cols.index("Time") + 1) if "Time" in cols else 0)
        if len(set(chosen.values())) < 5:
            st.error("Each AE parameter must use a different column.")
        elif st.button("Run clustering", type="primary"):
            st.session_state["own_run"] = (data, up.name, tuple(sorted((v, k) for k, v in chosen.items())),
                                           amp_in_db, 0 if k_opt.startswith("Auto") else int(k_opt), time_col)

        run = st.session_state.get("own_run")
        if run and run[1] == up.name:
            data, name, cmap, amp_in_db, k_choice, time_col = run
            with st.spinner("Clustering AE hits..."):
                try:
                    d, scores, q, order, new_bundle = run_pipeline(data, name, dict(cmap), amp_in_db, k_choice)
                except Exception as exc:
                    st.error(f"Could not process this file: {exc}")
                    st.stop()
            if len(d) < 200:
                st.error(f"Only {len(d)} valid hits found. At least 200 are needed for clustering.")
                st.stop()

            st.subheader("2. Clusters found")
            c = st.columns(4)
            c[0].metric("Valid hits", f"{len(d):,}")
            c[1].metric("Clusters", q["k"])
            c[2].metric("Silhouette", f"{q['silhouette']:.3f}")
            c[3].metric("Davies–Bouldin", f"{q['davies_bouldin']:.3f}")
            if q["dropped"]:
                st.caption(f"{q['dropped']:,} rows were skipped (missing or non-physical values).")
            prof = d.groupby("Mechanism")[FEATURES].median().loc[order].round(1)
            prof.insert(0, "Hits", d["Mechanism"].value_counts()[order])
            prof.insert(1, "Share %", (prof["Hits"] / len(d) * 100).round(1))
            st.dataframe(prof, width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(ra_af_figure(d, order))
                st.caption("RA–AF plot: tensile-type hits lie at low RA and high AF, shear-type at high RA and low AF.")
            with c2:
                st.line_chart(scores.set_index("k")[["Davies-Bouldin"]], color="#2a78d6")
                st.caption(f"Davies–Bouldin index for k = 2 to 8 (lower is better). k = {q['k']} was used.")

            if time_col != "None" and time_col in d.columns:
                t = pd.to_numeric(d[time_col], errors="coerce")
                if t.notna().sum() > 0 and t.max() > t.min():
                    stage = pd.cut((t - t.min()) / (t.max() - t.min()), [-0.01, 1 / 3, 2 / 3, 1.01], labels=["Early", "Mid", "Late"])
                    share = (pd.crosstab(stage, d["Mechanism"], normalize="index")[order] * 100).round(1)
                    st.subheader("Damage progression")
                    fig, ax = plt.subplots(figsize=(9, 2.2))
                    left = np.zeros(len(share))
                    for i, name in enumerate(order):
                        ax.barh(share.index.astype(str), share[name], left=left, color=PALETTE[i % len(PALETTE)],
                                edgecolor="white", linewidth=2, height=0.62, label=name)
                        for y, (l, v) in enumerate(zip(left, share[name])):
                            if v >= 7:
                                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8)
                        left += share[name].values
                    ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("Share of AE hits (%)")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.legend(ncol=len(order), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
                    fig.tight_layout(); st.pyplot(fig)
                    st.caption("Share of each mechanism in the early, mid and late thirds of the recording "
                               "(meaningful when the file holds a single test).")

            st.subheader("3. Classifier for this material")
            c = st.columns(3)
            c[0].metric("Accuracy (90/10 split)", f"{q['accuracy']:.1%}")
            c[1].metric("Macro-F1", f"{q['macro_f1']:.1%}")
            c[2].metric("Train / test hits", f"{q['train']:,} / {q['test']:,}")
            st.caption("Accuracy shows how well the classifier reproduces the clusters on unseen hits. "
                       "Mechanism names are interpretations from RA–AF and energy; confirm them with microscopy for a new material.")
            c1, c2, c3 = st.columns(3)
            if c1.button("Use this model on the Predict page"):
                st.session_state["custom_bundle"] = new_bundle
                st.success("Done. Open 'Predict New AE Data' and choose 'My trained model'.")
            buf = io.BytesIO(); joblib.dump(new_bundle, buf)
            c2.download_button("Download model (.joblib)", buf.getvalue(), "my_ae_model.joblib")
            c3.download_button("Download labelled hits (CSV)", d.to_csv(index=False).encode(), "ae_hits_with_mechanisms.csv", "text/csv")

else:
    st.title("Predict Damage Mechanisms for New AE Data")
    custom = st.session_state.get("custom_bundle")
    if custom is not None:
        which = st.radio("Model", ["Built-in GFRP model", f"My trained model ({custom['source']})"], horizontal=True)
        if which.startswith("My"):
            bundle = custom
    up = st.file_uploader("Upload AE file (CSV/XLSX) with Amplitude, Energy, Duration, Counts, Risetime",
                          type=["csv", "xlsx", "xls"])
    if up is not None:
        df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        need = ["Amplitude", "Energy", "Duration", "Counts", "Risetime"]
        cmap = bundle.get("column_map", {})
        miss = [c for c in (cmap or need) if c not in df.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            out = predict(df, bundle)
            c = st.columns(3)
            c[0].metric("Hits classified", f"{len(out):,}")
            c[1].metric("Confident predictions", f"{(out.Status == 'Confident').mean():.1%}")
            c[2].metric("Dominant mechanism", out["Predicted mechanism"].mode()[0])
            dist = out["Predicted mechanism"].value_counts().reindex(bundle["classes"]).fillna(0).rename("Hits")
            st.bar_chart(dist, color="#2a78d6")
            st.dataframe(out[need + ["RA", "AF", "Predicted mechanism", "Confidence", "Status"]].head(500),
                         width="stretch")
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                               "ae_predictions.csv", "text/csv")

    st.subheader("Single AE hit")
    c = st.columns(5)
    vals = {k: c[i].number_input(k, value=v, min_value=0.0) for i, (k, v) in enumerate(
        {"Amplitude": 60.0, "Energy": 15.0, "Duration": 600.0, "Counts": 40.0, "Risetime": 100.0}.items())}
    if st.button("Classify hit"):
        r = predict(pd.DataFrame([vals]), bundle).iloc[0]
        st.success(f"**{r['Predicted mechanism']}**: {bundle['descriptions'][r['Predicted mechanism']]} "
                   f"(confidence {r['Confidence']:.0%}, {r['Status'].lower()})")
