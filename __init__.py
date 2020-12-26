# By Jet, Dec 2020
#
# A convenient wrapper to call dcp model
#
import os
import numpy as np
import open3d as o3d
from typing import Optional, Union

import torch

from .model import DCP

PointCloud = o3d.geometry.PointCloud


class DeepClosestPoint:
    def __init__(self, pointer: str = 'transformer', ckpt: Optional[str] = None, device: str = 'cuda', **kwargs):
        assert pointer in ['identity', 'transformer']
        self.model_version = 'v1' if pointer == 'identity' else 'v2'
        self.pointer = pointer

        if ckpt is None:
            ckpt = os.path.join(os.path.dirname(__file__), f'pretrained/dcp_{self.model_version}.t7')

        args = _make_dcp_args(pointer=pointer, **kwargs)

        self.model = DCP(args).to(device)
        self.model.load_state_dict(torch.load(ckpt), strict=False)

        self.device = device

    @torch.no_grad()
    def solve(self, src: Union[np.ndarray, PointCloud], dst: Union[np.ndarray, PointCloud]):
        if isinstance(src, PointCloud):
            src = np.asarray(src.points)
        if isinstance(dst, PointCloud):
            dst = np.asarray(dst.points)
        assert src.shape[-1] == dst.shape[-1] == 3
        src = src.reshape(-1, 3, 1).T
        dst = dst.reshape(-1, 3, 1).T

        src = torch.FloatTensor(src).to(self.device)
        dst = torch.FloatTensor(dst).to(self.device)

        rotation_ab, translation_ab, _, _ = self.model(src, dst)

        return rotation_ab.squeeze(0).cpu().numpy(), translation_ab.squeeze(0).cpu().numpy()


def _make_dcp_args(**kwargs):
    args = type('args', (object,), {})

    args.emb_nn = 'dgcnn'
    args.emb_dims = 512
    args.head = 'svd'
    args.cycle = False
    args.pointer = 'transformer'

    # transformer specific
    args.n_blocks = 1
    args.dropout = 0.0
    args.ff_dims = 1024
    args.n_heads = 4

    for k, v in kwargs.items():
        setattr(args, k, v)

    return args


__all__ = ['DeepClosestPoint']
