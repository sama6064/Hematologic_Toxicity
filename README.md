# Early prediction of severe hematologic toxicity after day 28 (CALGB 40603)

Code for *Early Prediction of Severe Hematologic Toxicity After Day 28 of Neoadjuvant Chemotherapy: A Landmark Analysis of CALGB 40603 Trial Data*.

| File | Content |
|---|---|
| `RAQEEB_Hematologic_Toxicity.ipynb` | Landmark cohort, features, nested cross-validation and all experiments reported in the paper. Saved with outputs. |
| `external_bcirg005.py` | External evaluation in the BCIRG-005 control arm. |
| `results/` | CSV outputs of the notebook and the output of the external script. |

## Data

Not included. Available from [Project Data Sphere](https://www.projectdatasphere.org):

```
data/treated2.csv, data/aeclean.csv                         CALGB 40603 (Alliance)
data/bcirg005/{hematol,hemhdr,ae,demog,rando}.sas7bdat      BCIRG-005 (Sanofi)
```

## Run

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute RAQEEB_Hematologic_Toxicity.ipynb
python external_bcirg005.py
```

The notebook takes about 40 minutes on 2 CPU cores. Seeds are fixed.
