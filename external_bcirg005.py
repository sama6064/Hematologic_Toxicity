"""External evaluation in BCIRG-005 (control arm, AC -> docetaxel).

Reproduces the external numbers reported in the paper:
  1. The early-cytopenia rule (ANC or platelet grade >=1 by day 28) for a later (after day 28)
     grade >=3 neutropenia, leukopenia or thrombocytopenia, in CALGB 40603 and in BCIRG-005,
     with the same landmark definition in both trials.
  2. Cross-trial transport: models trained on the shared features (early cytopenia grades,
     with or without age) in one trial and tested in the other.
  3. Whether a threshold chosen in one trial carries over to the other.

Data (not redistributed; Project Data Sphere):
  data/treated2.csv, data/aeclean.csv               CALGB 40603
  data/bcirg005/{hematol,hemhdr,ae,demog,rando}.sas7bdat   BCIRG-005
Run:  python external_bcirg005.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = "data"
LANDMARK = 28
TYPES = ("Neutropenia", "Leukopenia", "Thrombocytopenia")

# ---------- CALGB 40603: hematologic AEs with onset day ----------
pt = pd.read_csv(f"{DATA}/treated2.csv")
ae = pd.read_csv(f"{DATA}/aeclean.csv")
ae = ae[ae.GRADE_ID.between(1, 5) & ae.aestart.notna()].copy()
name = {"Neutrophils/granulocytes (ANC/AGC)": "Neutropenia", "Leukocytes (total WBC)": "Leukopenia", "Platelets": "Thrombocytopenia"}
ae["type"] = ae.EVENTNAME.map(name)
calgb_ev = ae[ae.type.notna()].rename(columns={"MASK_ID": "id", "aestart": "day", "GRADE_ID": "g"})[["id", "type", "day", "g"]]

# ---------- BCIRG-005: grades from laboratory values plus reported AEs ----------
B = f"{DATA}/bcirg005/"
read = lambda n: pd.read_sas(B + n + ".sas7bdat", encoding="latin1")
labs, hdr, ae_b, demog, rando = read("hematol"), read("hemhdr"), read("ae"), read("demog"), read("rando")
labs = labs.merge(hdr[["RSUBJID", "EVENT_ID", "SEQNO", "ASSD_DY"]], on=["RSUBJID", "EVENT_ID", "SEQNO"], how="left")
labs = labs[labs.CLAB_VAL.notna() & labs.ASSD_DY.notna() & labs.LTEST.isin(["NEUTROPHILS", "WBC", "PLATELETS"])].copy()
labs["value"] = np.where(labs.LTEST.isin(["NEUTROPHILS", "WBC"]) & (labs.CLAB_VAL > 100), labs.CLAB_VAL / 1000, labs.CLAB_VAL)
CUTS = {"NEUTROPHILS": [2.0, 1.5, 1.0, 0.5], "WBC": [4.0, 3.0, 2.0, 1.0], "PLATELETS": [150, 75, 50, 25]}   # CTCAE lower limits, grades 1-4
labs["g"] = [sum(x < c for c in CUTS[t]) for t, x in zip(labs.LTEST, labs.value)]
labs["type"] = labs.LTEST.map({"NEUTROPHILS": "Neutropenia", "WBC": "Leukopenia", "PLATELETS": "Thrombocytopenia"})
bev = labs.rename(columns={"ASSD_DY": "day"})[["RSUBJID", "type", "day", "g"]]
ae_b["term"] = ae_b.NCICRI.astype(str).str.upper(); ae_b["type"] = None
for prefix, t in [("NEUTROPHILS", "Neutropenia"), ("LEUKOCYTES", "Leukopenia"), ("PLATELETS", "Thrombocytopenia")]:
    ae_b.loc[ae_b.term.str.startswith(prefix), "type"] = t
rep = ae_b[ae_b.type.notna() & ae_b.GRAD.notna()].rename(columns={"ST_DY": "day", "GRAD": "g"})[["RSUBJID", "type", "day", "g"]]
bcirg_ev = pd.concat([bev, rep], ignore_index=True).rename(columns={"RSUBJID": "id"})
bcirg_ids = rando.RSUBJID.values
age_b = demog.drop_duplicates("RSUBJID").set_index("RSUBJID").AGENO

# ---------- landmark cohort for the three-cytopenia composite ----------
def cohort(ev, ids, age, carbo=None):
    ev = ev[ev.day >= -7]
    e3 = ev[ev.g >= 3]; first = e3.groupby("id").day.min()
    ids = np.array(ids); ids = ids[~(pd.Series(ids).map(first).values <= LANDMARK)]
    d = pd.DataFrame({"id": ids})
    for ty, k in zip(TYPES, "NLP"):
        d[k] = ev[(ev.day <= LANDMARK) & (ev.type == ty)].groupby("id").g.max().reindex(ids).fillna(0).values
    d["age"] = pd.Series(ids).map(age).values.astype(float)
    d["y"] = d.id.isin(e3[e3.day > LANDMARK].id).astype(int)   # first event after day 28, to end of follow-up
    return d
C = cohort(calgb_ev, pt.MASK_ID.values, pt.set_index("MASK_ID").AGE_C)
Bc = cohort(bcirg_ev, bcirg_ids, age_b)

# ---------- 1. rule: ANC or platelet grade >=1 by day 28 ----------
for nm, d in [("CALGB 40603", C), ("BCIRG-005", Bc)]:
    r = ((d.N >= 1) | (d.P >= 1)).values; y = d.y.values
    print(f"[1] {nm:12s} n={len(d)} events={y.sum()}  rule sensitivity={(r & (y == 1)).sum()/y.sum():.3f}  specificity={((~r) & (y == 0)).sum()/(y == 0).sum():.3f}")
    print(f"    later risk: rule positive {y[r].sum()}/{r.sum()} = {100*y[r].mean():.1f}%   rule negative {y[~r].sum()}/{(~r).sum()} = {100*y[~r].mean():.1f}%")

# ---------- 2. cross-trial transport ----------
def model(n):
    imp = SimpleImputer(strategy="median")
    if n == "LR": return make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=2000))
    if n == "RF": return make_pipeline(imp, RandomForestClassifier(300, min_samples_leaf=20, random_state=0, n_jobs=-1))
    return make_pipeline(imp, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30, random_state=0))
# note: CALGB records age as a 3-level group and BCIRG as years, so "age" is only partly comparable
rows = []
for feats in (["N", "L", "P"], ["N", "L", "P", "age"]):
    for m in ("LR", "RF", "GB"):
        for tr, te, lab in ((C, Bc, "CALGB -> BCIRG"), (Bc, C, "BCIRG -> CALGB")):
            p = model(m).fit(tr[feats], tr.y).predict_proba(te[feats])[:, 1]
            rows.append(dict(direction=lab, features="+".join(feats), model=m, AUC=round(roc_auc_score(te.y, p), 3)))
R = pd.DataFrame(rows)
print("[2] cross-trial AUC\n", R.to_string(index=False))
print(R.groupby("direction").AUC.agg(["min", "max"]))

# ---------- 3. does a threshold chosen in one trial carry over? ----------
from sklearn.model_selection import StratifiedKFold, cross_val_predict
def thr_sens80(y, p):
    best = None
    for t in np.unique(p):
        if ((p >= t) & (y == 1)).sum() / y.sum() >= 0.80: best = t
    return best
feats = ["N", "L", "P"]
for tr, te, lab in ((C, Bc, "CALGB -> BCIRG"), (Bc, C, "BCIRG -> CALGB")):
    for m in ("LR", "RF"):
        p_tr = cross_val_predict(model(m), tr[feats], tr.y, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_proba")[:, 1]
        thr = thr_sens80(tr.y.values, p_tr)
        p = model(m).fit(tr[feats], tr.y).predict_proba(te[feats])[:, 1]
        pr = p >= thr; y = te.y.values
        print(f"[3] {lab} {m}: threshold set for sensitivity 0.80 in training -> test sensitivity {(pr & (y==1)).sum()/y.sum():.2f}, specificity {((~pr) & (y==0)).sum()/(y==0).sum():.2f}")
