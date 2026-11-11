"""Tissue classes for the robustness table (RM11), on real strings from the held-out samples."""

import pytest

from rna_age.train.robustness import tissue_class

CASES = {
    "Whole blood": "blood", "WBS": "blood", "mRNA extracted from whole blood": "blood", "whole blood leukocyte": "blood",
    "PBMCs": "pbmc", "peripheral blood mononuclear cells": "pbmc",
    "Peripheral blood neutrophils": "sorted_immune_cells", "memory CD4+/CD25- T cells": "sorted_immune_cells",
    "CD19+/CD27- B cells": "sorted_immune_cells", "naive helper T cells from peripheral blood": "sorted_immune_cells",
    "plasma, extracted cell-free RNA": "plasma_cfrna", "blood plasma": "plasma_cfrna",
    "Brain - Caudate (basal ganglia)": "brain_cns", "dorsolateral prefrontal cortex (DLPFC; BA8/9)": "brain_cns",
    "fusiform gyrus": "brain_cns", "spinal_cord_lumbar": "brain_cns", "AD brain": "brain_cns",
    "human meningioma": "meningioma", "nasal lavage": "airway", "bronchoalveolar lavage (BAL) cells": "airway",
    "whole lung": "lung", "Heart - Atrial Appendage": "heart", "cardiac muscle": "heart", "left atria": "heart",
    "Muscle - Skeletal": "skeletal_muscle", "musculus vastus lateralis": "skeletal_muscle",
    "Artery - Tibial": "vascular", "basilic vein": "vascular", "Cells - Cultured fibroblasts": "fibroblast",
    "Skin - Sun Exposed (Lower leg)": "skin", "Adipose - Visceral (Omentum)": "adipose", "Ileal biopsy": "gut",
    "Esophagus - Mucosa": "gut", "human islets": "pancreas", "Breast - Mammary Tissue": "breast",
    "minor salivary glands": "oral", "knee articular cartilage": "bone_joint", "liver hepatocyte cells": "liver",
    "Nerve - Tibial": "other", None: "unknown",
}  # fmt: skip


@pytest.mark.parametrize(("text", "expected"), list(CASES.items()))
def test_tissue_class(text, expected):
    assert tissue_class(text) == expected
