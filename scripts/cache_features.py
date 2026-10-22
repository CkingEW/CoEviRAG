# Modified for CoEviRAG: retain the 60D cache adapter only.
import argparse

import json

from pathlib import Path

import sys

import numpy as np

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from evidence_features import NO_TOP2_SELECTOR, multiscale_features

RECALL_KS = (1, 2, 5, 10, 20, 30, 50, 100, 150, 200)

def build_features(exported, query_ids):
    if exported.shape[1] != 46:
        raise ValueError(
            "Expected 46 features (14 candidate + 5 Paragraph + 27 query), "
            f"got {exported.shape[1]}"
        )
    base = exported[:, :14]
    paragraph = exported[:, 14:19]
    legacy = np.concatenate((base, exported[:, 19:]), axis=1)
    multiscale = np.zeros((len(base), 21), dtype=np.float32)
    for query_id in np.unique(query_ids):
        mask = query_ids == query_id
        multiscale[mask] = multiscale_features(base[mask], True)
    return np.concatenate(
        (legacy[:, NO_TOP2_SELECTOR], paragraph, multiscale), axis=1
    ).astype(np.float32)
