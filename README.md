# retinal-age-factors

Code for the RETFound DINOv2 reanalysis of *Factors Associated with Machine
Learning–Based Predictions of Retinal Aging Using Teleretinal Screening Images
from Patients with Diabetes* (Nguyen et al., Ophthalmology Science 2026,
[PMC12992952](https://pmc.ncbi.nlm.nih.gov/articles/PMC12992952/)).

**Code only.** No participant data are included. The CUTI images and chart and
EHR extracts contain protected health information and stay outside git.

> Status: the code was pushed before end-to-end testing. The dataset build and
> the fold splits were checked against the reference run (see Verification).
> Neither a full training run nor the AI-READI stages have been run with this
> code. The factor-association analysis has not been ported yet.

## Pipeline

```bash
pip install -e .        # or: uv pip install -e .
bash run_all.sh         # download -> make-splits -> run-cv -> build-dataset
```

| Stage | Command | Output |
|---|---|---|
| Public data | `python -m retinal_age_factors download` | `data/public/` (ACS 2024 5-yr, ReADI ADI, ZCTA-tract file, SVI 2022, PLACES 2024, Opportunity Insights) |
| Folds | `make-splits` | `outputs/retinal_age/manifests/` |
| 10-fold training | `run-cv --gpus 0 1 2` | `outputs/retinal_age/{checkpoints,predictions,metrics}/` |
| Dataset | `build-dataset` | `outputs/full_rag_data.csv`, `outputs/full_rag_data_patients.csv` |
| AI-READI | `prepare-aireadi`, `run-automorph`, `train-final`, `external` | `outputs/retinal_age/aireadi/`, `metrics/aireadi_external.json` |

### Model

RETFound DINOv2-MEH (`YukunZhou/RETFound_dinov2_meh`, ViT-L/14), fully fine-tuned
with the official RETFound DINOv2 recipe: batch 24, base LR 5e-3, 50 epochs,
10 warmup epochs, layer decay 0.65, drop path 0.2, RandAugment, and random
erasing. The head is a single output trained with L1 loss on age in years.
Left eyes are mirrored before input. Folds come from `StratifiedGroupKFold`
grouped by MRN and stratified by age decile (seed 2026); fold *k* is test,
fold *k*+1 is validation, and the rest is training (80/10/10). Validation
patient-level MAE selects the checkpoint.

### Retinal age gap

`Retinal_Age_Gap` = out-of-fold predicted age − chronological age, per image.
`Patient_Retinal_Age_Gap` = mean predicted age − mean chronological age across
a patient's images. The analysis unit in the paper is the patient.

## Required inputs (`data/`, or set `RAF_DATA_DIR`)

| File | Content |
|---|---|
| `Good_quality_only.csv` | Chart review of each quality-filtered screening visit (1,861 visits) |
| `human_data.csv` | AutoMorph-passed images: `path, MRN, date, age, sex` (4,021 images, 1,629 patients) |
| `Good_quality/` | The fundus photographs |
| `RDV0002447_v6.csv` | EHR vitals, labs, and medications |
| `Modified_RUCA_Zip_Codes_2010.csv` | ZIP-level RUCA codes |
| `aireadi/` | AI-READI dataset root (external validation only) |

The AI-READI stages also need [AutoMorph](https://github.com/rmaphoh/AutoMorph),
with its weights, cloned into `third_party/AutoMorph`.

## Notes on the dataset

- Clinical values come from the nearest same-MRN RDV record within 183 days of
  the image. Unmatched values stay missing.
- Medication use (`Statin_Use`, `GLP1_Use`, `Antihypertensive_Use`,
  `RDV_*_Present`) is 1/0 among images with a matched record and missing
  otherwise.
- PREVENT 2023 (HbA1c model) uses `resources/prevent_hba1c_coefficients.csv`,
  taken from preventr 0.12.1. This includes the 2026-10-14 erratum that swapped
  the HbA1c-in-diabetes and HbA1c-without-diabetes coefficients. Coefficients
  are looked up by term name, not row position.
- eGFR uses the race-free 2021 CKD-EPI creatinine equation.
- ZIP-level measures (RUCA, ACS, ADI, SVI, PLACES, Chetty) describe the
  patient's area, not the individual.

## Verification against the reference run (2026-09-27)

- `make-splits` reproduces the reference fold and train/val/test assignment
  exactly (scikit-learn 1.7.2).
- `build-dataset` on the reference out-of-fold predictions reproduces every
  column of the curated `full_rag_data.xlsx` except the intentional changes
  above: the PREVENT erratum and the medication 0/1 coding.

Reference performance (patient level, n = 1,629): MAE 3.83 years,
RMSE 5.06, R² 0.890, Pearson r 0.944.

## Licenses

`model/retfound.py` adapts two helpers from RETFound (CC BY-NC 4.0).
