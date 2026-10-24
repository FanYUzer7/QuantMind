"""Test-only custom architecture, custom loss, and portable state-dict loading."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LabMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                 nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = (x - self.feature_mean) / self.feature_std
        return self.net(x).squeeze(-1)

    def predict(self, values):
        self.eval()
        array = np.asarray(values, dtype=np.float32)
        device = next(self.parameters()).device
        with torch.no_grad():
            return np.concatenate([self(torch.as_tensor(array[i:i + 8192], device=device)).cpu().numpy()
                                   for i in range(0, len(array), 8192)])


def lab_pairwise_loss(pred, target, day_codes, rank_weight=0.1):
    """SmoothL1 plus within-day pair ranking; independent of Qlib losses."""
    regression = F.smooth_l1_loss(pred, target, beta=0.1)
    terms = []
    for day in torch.unique(day_codes):
        mask = day_codes == day
        p, y = pred[mask], target[mask]
        if len(p) < 2:
            continue
        direction = torch.sign(y - torch.roll(y, 1))
        keep = direction != 0
        if keep.any():
            terms.append(F.softplus(-direction[keep] * (p - torch.roll(p, 1))[keep]).mean())
    ranking = torch.stack(terms).mean() if terms else pred.sum() * 0
    return regression + float(rank_weight) * ranking


def load_lab_model(path: Path, metadata: dict):
    arch = metadata['model_arch']
    model = LabMLP(int(arch['input_dim']), int(arch['hidden_dim']))
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model
