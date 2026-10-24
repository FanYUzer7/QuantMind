"""Native QuantMind training adapter for a genuinely custom PyTorch loop."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.shared.lab_torch_model import LabMLP, lab_pairwise_loss
from data.splits import _split_data, _prepare_arrays
from model_trainers.metrics import _compute_metrics


def train_lab_mlp(df, features, cfg, hardware=None):
    started = time.time()
    frames = dict(zip(['train', 'valid', 'test'], _split_data(df, cfg)))
    frames = {k: v.sort_values(['trade_date', 'symbol']).reset_index(drop=True) for k, v in frames.items()}
    fill_values, x, y, xv, yv, fill = _prepare_arrays(frames['train'], frames['valid'], features)
    params = cfg['model'].get('dl_params', {})
    hidden = int(params.get('hidden_size', 32))
    epochs = int(params.get('n_epochs', 3))
    rank_weight = float(params.get('rank_weight', 0.1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('This functional experiment requires real GPU execution')
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(cfg.get('seed', 42)))
    model = LabMLP(len(features), hidden).to(device)
    model.feature_mean.copy_(torch.as_tensor(x.mean(axis=0), device=device))
    std = x.std(axis=0)
    model.feature_std.copy_(torch.as_tensor(np.where(std < 1e-6, 1., std), device=device))
    initial = {k: p.detach().clone() for k, p in model.named_parameters()}
    days = pd.factorize(frames['train'].trade_date, sort=True)[0]
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.as_tensor(days, dtype=torch.long))
    loader = DataLoader(ds, batch_size=int(params.get('batch_size', 800)), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get('lr', 0.001)))
    history, best, best_state, gradient_max = [], float('inf'), None, 0.
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb, day in loader:
            xb, yb, day = xb.to(device), yb.to(device), day.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = lab_pairwise_loss(model(xb), yb, day, rank_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite custom loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(grad):
                raise RuntimeError('non-finite custom gradient')
            gradient_max = max(gradient_max, float(grad))
            optimizer.step()
            losses.append(float(loss.detach()))
        prediction = model.predict(xv)
        validation_mse = float(np.mean((prediction - yv) ** 2))
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'validation_mse': validation_mse})
        print('LAB_EPOCH', json.dumps(history[-1]), flush=True)
        if validation_mse < best:
            best, best_state = validation_mse, copy.deepcopy(model.state_dict())
    change = max(float((p.detach() - initial[k]).abs().max()) for k, p in model.named_parameters())
    assert change > 0 and gradient_max > 0
    model.load_state_dict(best_state)
    out = Path(cfg['output']['result_path']).parent
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'model.pth')
    trace = {'device': str(device), 'gpu': torch.cuda.get_device_name(0), 'network': 'LabMLP',
             'loss': 'SmoothL1 + within-day softplus pair ranking', 'rank_weight': rank_weight,
             'hidden_dim': hidden, 'epochs': epochs, 'optimizer': 'AdamW', 'weight_max_change': change,
             'gradient_norm_max': gradient_max, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
             'history': history, 'elapsed_seconds': time.time() - started,
             'label': 'lab_return: forward T+1 to T+6 return clipped to [-0.3,0.3]', 'functional_test_only': True}
    (out / 'custom_training_trace.json').write_text(json.dumps(trace, indent=2))
    predictions = df[['symbol', 'trade_date', 'label']].copy()
    predictions['pred'] = model.predict(fill(df))
    predictions['split'] = 'excluded'
    for key, frame in frames.items():
        predictions.loc[predictions.trade_date.between(frame.trade_date.min(), frame.trade_date.max()), 'split'] = key
    metrics = {key: _compute_metrics(frame, frame.label.to_numpy(dtype=np.float32), model.predict(fill(frame))) for key, frame in frames.items()}
    metadata = {'model_arch': {'input_dim': len(features), 'hidden_dim': hidden},
                'is_sequence_model': False, 'custom_loss': trace['loss'], 'functional_test_only': True,
                'feature_normalization': 'train-only mean/std stored as state_dict buffers'}
    return model, fill_values, metrics['train'], metrics['valid'], metrics['test'], predictions, frames, 'lab_mlp', metadata
