# Early prediction of severe toxicity after day 28 (CALGB 40603)

Code for *Early Prediction of Severe Hematologic Versus Any-Type Toxicity After Day 28 of Neoadjuvant Chemotherapy: A Landmark Analysis of CALGB 40603 Trial Data*.

| File | Content |
|---|---|
| `RAQEEB_Toxicity_Landmark.ipynb` | Landmark cohorts, features, nested cross-validation and every experiment in the paper, run for the hematologic composite (`heme`, primary) and any grade ≥3 AE (`any`). Section 20 answers RQ6: specific outcomes vs. any grade ≥3 AE, each with baseline variables alone and with early AEs. Sections 22–29 were added during revision: outcome-window sensitivity analyses, paired tests for RQ2, sample size (Riley et al.), calibration and decision curves of the primary model, RQ6 intervals over all three seeds, RQ3 tests with risk tiers within arms, and (Section 29) participant flow, early-AE feature cut-offs at days 21 and 14, seed-averaged metric intervals, the final model refitted on all patients, and a three-variable logistic score. Saved with outputs. |
| `external_bcirg005.py` | External consistency check in the BCIRG-005 control arm. |
| `results/` | CSV outputs of the notebook (`res_heme_*`, `res_any_*`, `summary_*`, `rq6_specific_vs_any.csv`) the output of the external script, and `revision/` (outputs of Sections 22–29). |

## Data

Not included. Available from [Project Data Sphere](https://www.projectdatasphere.org):

```
data/treated2.csv, data/aeclean.csv                         CALGB 40603 (Alliance)
data/bcirg005/{hematol,hemhdr,ae,demog,rando}.sas7bdat      BCIRG-005 (Sanofi)
```

## Run

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute RAQEEB_Toxicity_Landmark.ipynb
python external_bcirg005.py
```

The notebook takes about two and a half hours on 2 CPU cores. Seeds are fixed for data splitting and randomized learners. The liblinear and saga solvers draw from NumPy's global generator, which is seeded only from Section 22 onward, so a re-run of Sections 1–21 can differ in the third decimal for the L1 and elastic-net models.

Sections 1–21 were executed first; Sections 22–29 were executed later in a separate session after re-running the setup cells (Sections 0–4 and the configuration definitions), which is why their execution counts continue from the end of Section 21 (Sections 28 and 29 in later sessions).
