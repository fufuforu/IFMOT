# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from .deformable_detr import build as build_deformable_detr
from .ifmot import build as build_ifmot
from .motr import build as build_motr


def build_model(args):
    arch_catalog = {
        'deformable_detr': build_deformable_detr,
        'motr': build_motr,
        'ifmot': build_ifmot,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    return arch_catalog[args.meta_arch](args)
