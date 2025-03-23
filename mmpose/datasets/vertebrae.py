# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from .datasets.base import BaseCocoStyleDataset


@DATASETS.register_module(name="VertebraeDataset")
class VertebraeDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/vertebrae.py')
