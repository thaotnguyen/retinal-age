"""Command-line entry point: ``python -m retinal_age_factors <stage>``.

Stages, in pipeline order (``run_all.sh`` runs the CUTI stages end to end):

    download         public contextual data (ACS, ADI, SVI, PLACES, Chetty)
    make-splits      patient-exclusive 10-fold manifests
    run-cv           train all ten folds (one subprocess per GPU)
    train-fold       train a single fold (used by run-cv)
    build-dataset    full_rag_data.csv and full_rag_data_patients.csv
    prepare-aireadi  AI-READI DICOM -> PNG (external validation)
    run-automorph    AI-READI AutoMorph quality filter
    train-final      model on all CUTI images
    external         AI-READI predictions and metrics
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="retinal_age_factors", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ["download", "make-splits", "build-dataset", "prepare-aireadi", "train-final", "external"]:
        sub.add_parser(name)
    sub.add_parser("train-fold").add_argument("--fold", type=int, required=True, choices=range(10))
    sub.add_parser("run-cv").add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    sub.add_parser("run-automorph").add_argument("--devices", nargs="+", default=["cuda:0"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Stage imports are deferred so data-only stages do not import torch.
    if args.stage == "download":
        from .data.public import download

        download()
    elif args.stage == "make-splits":
        from .model.cv import make_splits

        make_splits()
    elif args.stage == "train-fold":
        from .model.cv import train_fold

        train_fold(args.fold)
    elif args.stage == "run-cv":
        from .model.cv import run_cv

        run_cv(args.gpus)
    elif args.stage == "build-dataset":
        from .data.dataset import build_dataset

        build_dataset()
    elif args.stage == "prepare-aireadi":
        from .model.external import prepare_aireadi

        prepare_aireadi()
    elif args.stage == "run-automorph":
        from .model.external import run_automorph

        run_automorph(args.devices)
    elif args.stage == "train-final":
        from .model.external import train_final

        train_final()
    elif args.stage == "external":
        from .model.external import external

        external()


if __name__ == "__main__":
    main()
