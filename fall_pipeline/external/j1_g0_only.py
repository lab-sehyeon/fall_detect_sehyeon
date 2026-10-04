"""Skeleton-only inference: frozen DSTE backbone -> J1 adapter -> G0.

No ADL, dataset-specific J1 heads, global features/scaler, G1 or G2 forward.
"""
from pathlib import Path
import torch
from torch import nn
from fall_pipeline.joint.models import J1ResidualAdapter


class J1G0Only(nn.Module):
    def __init__(self, backbone, adapter=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.adapter = J1ResidualAdapter() if adapter is None else adapter
        self.head = nn.Linear(2048,4) if head is None else head

    def forward(self, batch):
        if batch.ndim != 5 or tuple(batch.shape[1:]) != (3,64,25,2):
            raise ValueError('expected B,3,64,25,2 skeleton windows')
        jt = batch.permute(0,2,4,3,1).reshape(batch.shape[0],64,150)
        js = batch.permute(0,4,3,2,1).reshape(batch.shape[0],50,192)
        temporal,spatial = self.backbone(jt,js)
        pooled = torch.cat((temporal.amax(1),spatial.amax(1)),1)
        if pooled.shape != (len(batch),2048): raise ValueError('DSTE representation shape')
        adapted = self.adapter(pooled)
        return dict(pooled=pooled,adapted=adapted,G0=self.head(adapted))


def adapter_state(full_state):
    return {k.removeprefix('adapter.'):v for k,v in full_state.items() if k.startswith('adapter.')}


def load_model(root, spec, device):
    from fall_pipeline.fu.fall_fu_linear_eval import build_encoder
    root=Path(root)
    # Extract only the backbone: Downstream.fc is not retained or invoked.
    encoder=build_encoder(root/spec['encoder']['path'],device)
    backbone=encoder.backbone
    del encoder
    adapter=J1ResidualAdapter()
    full=torch.load(root/spec['j1']['path'],map_location='cpu',weights_only=True)
    adapter.load_state_dict(adapter_state(full),strict=True)
    del full
    head=nn.Linear(2048,4)
    head.load_state_dict(torch.load(root/spec['g0']['path'],map_location='cpu',weights_only=True),strict=True)
    model=J1G0Only(backbone,adapter,head).eval().requires_grad_(False).to(device)
    if set(dict(model.named_children())) != {'backbone','adapter','head'}:
        raise RuntimeError('unexpected active model branch')
    return model
