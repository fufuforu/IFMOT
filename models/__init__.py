# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build as build_deformable_detr
from .motr import build as build_motr
from .motr_unsupervise import build as build_motr_unsupervise
from .motr_pretrain import build as build_motr_pretrain
from .motr_pretrain_v1 import build as build_motr_pretrain_v1

from .motr_mot17_aug2_v1 import build as build_motr_mot17_aug2_v1
from .motr_mot17_aug2_v2 import build as build_motr_mot17_aug2_v2
from .motr_mot17_aug2_v3 import build as build_motr_mot17_aug2_v3
from .motr_mot17_aug2_v4 import build as build_motr_mot17_aug2_v4
from .motr_mot17_aug2_v5 import build as build_motr_mot17_aug2_v5
from .motr_mot17_aug2_v6 import build as build_motr_mot17_aug2_v6
from .motr_mot17_aug2_v7 import build as build_motr_mot17_aug2_v7
from .motr_mot17_aug2_v8 import build as build_motr_mot17_aug2_v8
from .motr_mot17_aug2_v10 import build as build_motr_mot17_aug2_v10

from .motr_dance_aug2_v1 import build as build_motr_dance_aug2_v1
from .motr_dance_aug2_v2 import build as build_motr_dance_aug2_v2
from .motr_dance_aug2_v2_1 import build as build_motr_dance_aug2_v2_1
from .motr_dance_aug2_v3 import build as build_motr_dance_aug2_v3
from .motr_dance_aug2_v4 import build as build_motr_dance_aug2_v4
from .motr_dance_aug2_v5 import build as build_motr_dance_aug2_v5
from .motr_dance_aug2_v6 import build as build_motr_dance_aug2_v6
from .motr_dance_aug2_v7 import build as build_motr_dance_aug2_v7
from .motr_dance_aug2_v8 import build as build_motr_dance_aug2_v8
from .motr_dance_aug2_v9 import build as build_motr_dance_aug2_v9

from .motr_aug2_CFQM_NSM_MLC import build as build_motr_aug2_CFQM_NSM_MLC
from .motr_aug1_aug2 import build as build_motr_aug1_aug2
def build_model(args):
    arch_catalog = {
        'deformable_detr': build_deformable_detr,
        'motr': build_motr,
        'motr_unsupervise': build_motr_unsupervise,
        'motr_pretrain': build_motr_pretrain,
        'motr_pretrain_v1': build_motr_pretrain_v1,
        
        'motr_dance_aug2_v1': build_motr_dance_aug2_v1,
        'motr_mot17_aug2_v1': build_motr_mot17_aug2_v1,
        'motr_mot17_aug2_v2': build_motr_mot17_aug2_v2,
        'motr_mot17_aug2_v3': build_motr_mot17_aug2_v3,
        'motr_mot17_aug2_v4': build_motr_mot17_aug2_v4,
        'motr_mot17_aug2_v5': build_motr_mot17_aug2_v5,
        'motr_mot17_aug2_v6': build_motr_mot17_aug2_v6,
        'motr_mot17_aug2_v7': build_motr_mot17_aug2_v7,
        'motr_mot17_aug2_v8': build_motr_mot17_aug2_v8,
        'motr_mot17_aug2_v10': build_motr_mot17_aug2_v10,


        'motr_dance_aug2_v2_1': build_motr_dance_aug2_v2_1,
        'motr_dance_aug2_v2': build_motr_dance_aug2_v2,
        'motr_dance_aug2_v3': build_motr_dance_aug2_v3,
        'motr_dance_aug2_v4': build_motr_dance_aug2_v4,
        'motr_dance_aug2_v5': build_motr_dance_aug2_v5,
        'motr_dance_aug2_v6': build_motr_dance_aug2_v6,
        'motr_dance_aug2_v7': build_motr_dance_aug2_v7,
        'motr_dance_aug2_v8': build_motr_dance_aug2_v8,
        'motr_dance_aug2_v9': build_motr_dance_aug2_v9,



        'motr_aug2_CFQM_NSM_MLC': build_motr_aug2_CFQM_NSM_MLC,
        'motr_aug1_aug2': build_motr_aug1_aug2,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)

