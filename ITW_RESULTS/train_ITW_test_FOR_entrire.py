import os
import re
import glob
import argparse
from datetime import datetime
from typing import List, Tuple

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# -------------------- Metrics --------------------
def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))


def eval_at_threshold(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for mel, label in tqdm(loader, desc="Evaluating", leave=False):
            mel = mel.to(device)
            out = model(mel)
            y_true.extend(label.numpy())
            y_score.extend(torch.sigmoid(out.squeeze()).cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    return acc, prec, rec, f1, auc, eer, len(y_true)


# -------------------- Dataset: FOR (for-2seconds) testing --------------------
class For2SecEvalDataset(Dataset):
    def __init__(self, for_root: str, split: str = 'testing', sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 512, hop_length: int = 80):
        self.items: List[Tuple[str, int]] = []
        # Support single split or aggregate all splits
        splits: List[str]
        if isinstance(split, str) and split.lower() == 'all':
            splits = ['training', 'validation', 'testing']
        else:
            splits = [split]
        for cur_split in splits:
            split_dir = os.path.join(for_root, cur_split)
            for label_name, label in [('real', 1), ('fake', 0)]:
                class_dir = os.path.join(split_dir, label_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith('.wav'):
                        self.items.append((os.path.join(class_dir, fname), label))
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        waveform, sr = torchaudio.load(path)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure exactly 10 seconds of audio (160000 samples at 16kHz) - same as LA training
        target_length = 10 * self.sample_rate  # 10 seconds
        if waveform.shape[0] < target_length:
            # Pad with zeros if shorter than 10 seconds
            pad_length = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            # Truncate to exactly 10 seconds if longer
            waveform = waveform[:target_length]
        
        waveform = waveform / (torch.norm(waveform) + 1e-8)
        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-6)
        return mel, torch.tensor(label, dtype=torch.long)


def make_for_loader(for_root: str, batch_size: int = 512):
    # Use the entire dataset: training + validation + testing
    ds = For2SecEvalDataset(for_root, split='all')
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# -------------------- ODE Solvers --------------------
def euler_step(func, t, y, dt):
    """Euler method for ODE solving"""
    return y + dt * func(t, y)

def rk4_step(func, t, y, dt):
    """4th order Runge-Kutta method for ODE solving"""
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt*k1/2)
    k3 = func(t + dt/2, y + dt*k2/2)
    k4 = func(t + dt, y + dt*k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def rk5_step(func, t, y, dt):
    """5th order Dormand-Prince method for ODE solving (simplified)"""
    # Simplified RK5 implementation to avoid tensor dimension issues
    k1 = func(t, y)
    k2 = func(t + dt/5, y + dt*k1/5)
    k3 = func(t + 3*dt/10, y + dt*(3*k1/40 + 9*k2/40))
    k4 = func(t + 4*dt/5, y + dt*(44*k1/45 - 56*k2/15 + 32*k3/9))
    k5 = func(t + 8*dt/9, y + dt*(19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729))
    k6 = func(t + dt, y + dt*(9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656))
    
    # Final step using 5th order coefficients
    return y + dt * (35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84)

def adaptive_step(func, t, y, dt, tol=1e-6, max_iter=5):
    """Adaptive step size ODE solver with error estimation (simplified)"""
    dt_current = dt
    for _ in range(max_iter):
        # Compute two solutions with different step sizes
        y1 = rk4_step(func, t, y, dt_current)
        y2 = rk4_step(func, t, y, dt_current/2)
        y2 = rk4_step(func, t + dt_current/2, y2, dt_current/2)
        
        # Estimate error
        error = torch.norm(y1 - y2)
        if error < tol:
            return y1, dt_current
        else:
            # Reduce step size based on error
            dt_current = dt_current * 0.5
    
    # If we reach here, use the best available solution
    return y1, dt_current


# -------------------- Model (Improved LTC with 3 cells) --------------------
class LiquidTimeConstantCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, tau_init: float = 0.1, solver_type: str = 'rk4'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver_type = solver_type
        
        # LTC parameters (inspired by the research implementation)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        
        # Time constants (learnable)
        self.log_tau = nn.Parameter(torch.full((hidden_size,), math.log(tau_init)))
        
        # Additional LTC parameters for better dynamics
        self.vleak = nn.Parameter(torch.zeros(hidden_size))
        self.gleak = nn.Parameter(torch.ones(hidden_size))
        self.cm = nn.Parameter(torch.ones(hidden_size))
        
        # Synaptic parameters
        self.mu = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.sigma = nn.Parameter(torch.ones(hidden_size, hidden_size) * 2.0)
        self.erev = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Standard LTC forward pass
        tau = torch.exp(self.log_tau).clamp(min=0.05, max=10.0)
        preact = self.W_in(x) + self.W_rec(h)
        preact = torch.clamp(preact, min=-10, max=10)
        activation = torch.tanh(preact)
        dh = (-h + activation) / tau
        dh = torch.clamp(dh, min=-5.0, max=5.0)
        h_new = h + dh
        return h_new

    def ode_func(self, t, h, x_input=None):
        """ODE function for the LTC cell with improved dynamics"""
        batch_size = h.shape[0]
        device = h.device
        
        # Clamp parameters for stability
        tau = torch.exp(self.log_tau).clamp(min=0.05, max=10.0)
        cm = self.cm.clamp(min=0.1, max=10.0)
        gleak = self.gleak.clamp(min=0.1, max=10.0)
        
        if x_input is not None:
            # Input-driven dynamics
            input_contribution = self.W_in(x_input)
        else:
            input_contribution = torch.zeros_like(h)
        
        # Simplified synaptic dynamics to avoid tensor dimension issues
        # Use a simpler approach that's more stable
        synaptic_activation = torch.tanh(self.W_rec(h))
        synaptic_current = synaptic_activation
        
        # Leak current
        leak_current = gleak * (self.vleak - h)
        
        # Total current (simplified)
        total_current = input_contribution + synaptic_current + leak_current
        
        # Membrane potential change
        dh_dt = total_current / cm
        
        return torch.clamp(dh_dt, min=-5.0, max=5.0)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, n_mels: int, cnn_hidden: int = 32):
        super().__init__()
        # 5 -> 1 -> 3 -> 3 -> 1 architecture
        self.conv1 = nn.Conv1d(n_mels, cnn_hidden, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(cnn_hidden)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_hidden)
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(cnn_hidden)
        self.act4 = nn.ReLU()
        
        self.conv5 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(cnn_hidden)
        self.act5 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        return x


class LiqNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, out_size: int, time_steps: int, tau_init: float = 0.1):
        super().__init__()
        self.conv_extractor = ConvFeatureExtractor(input_size, hidden_size)
        
        # Single LTC cell with RK4 solver
        self.ltc_cell = LiquidTimeConstantCell(hidden_size, hidden_size, tau_init=tau_init, solver_type='rk4')
        
        self.final_fc = nn.Linear(hidden_size, out_size)
        self.time_steps = time_steps
        self.hidden_size = hidden_size
        
        # ODE solver parameters
        self.dt = 0.01
        self.ode_unfolds = 2  # Reduced from 6 to 2 for stability and speed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        
        # Extract features using conv layers
        conv_features = self.conv_extractor(x)  # [batch, hidden_size, time_steps]
        
        # Initialize hidden state for single LTC cell
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Process through single LTC cell with RK4 solver
        for t in range(self.time_steps):
            # Get current time step features
            current_features = conv_features[:, :, t]
            
            # Multiple RK4 steps per time step
            for _ in range(self.ode_unfolds):
                h = rk4_step(lambda t, h: self.ltc_cell.ode_func(t, h, current_features), t, h, self.dt)
        
        out = self.final_fc(h)
        return out


# -------------------- Helpers --------------------
_PHASE2_E20_NAME = 'phase2_epoch40.pth'


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate ITW checkpoints (phase2_epoch40.pth) on entire FOR dataset at thr=0.5 and log results')
    parser.add_argument('--save_root', type=str, default='ITW_logs2', help='Directory containing state_* folders with ITW checkpoints')
    parser.add_argument('--for_root', type=str, default=r"C:\Users\Shivaay Dhondiyal\Desktop\shivaay\coding\2_projects\9_deepfake_paper\deepfake2\for-2seconds")
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve paths relative to script location
    script_dir = os.path.dirname(__file__)
    save_root_abs = args.save_root if os.path.isabs(args.save_root) else os.path.join(script_dir, args.save_root)
    for_root_abs = args.for_root if os.path.isabs(args.for_root) else os.path.join(script_dir, args.for_root)

    if not os.path.isdir(save_root_abs):
        print(f"Save root not found: {save_root_abs}")
        return

    # Prepare FOR loader (entire dataset)
    for_loader = make_for_loader(for_root_abs, batch_size=args.batch_size)

    # Prepare global log under ITW_logs2
    itw_logs_root = save_root_abs
    os.makedirs(itw_logs_root, exist_ok=True)
    out_log = os.path.join(itw_logs_root, 'ITW_eval_FOR_phase2_epoch40_results.txt')
    if not os.path.exists(out_log):
        with open(out_log, 'w', encoding='utf-8') as f:
            f.write('timestamp\tstate\tcheckpoint\tN\tthr\tacc\tprec\trec\tf1\teer\tauc\n')

    # Scan folders
    all_dirs = [d for d in sorted(os.listdir(save_root_abs)) if os.path.isdir(os.path.join(save_root_abs, d))]
    # Look for repeat* folders (ITW training folders)
    candidate_dirs = [d for d in all_dirs if d.startswith('repeat')]
    
    if not candidate_dirs:
        print(f"No repeat* folders under: {save_root_abs}")
        return

    print("Evaluating phase2_epoch40.pth on entire FOR dataset for each folder...")

    for dir_name in candidate_dirs:
        state_dir = os.path.join(save_root_abs, dir_name)
        
        # Find all .pth files in the directory
        pth_files = [f for f in os.listdir(state_dir) if f.endswith('.pth')]
        if not pth_files:
            print(f"No .pth files found in: {state_dir}")
            continue
        
        # Sort by filename to get the last one (highest epoch number)
        pth_files.sort()
        target_ckpt = os.path.join(state_dir, pth_files[-1])
        print(f"Using checkpoint: {pth_files[-1]} from {dir_name}")

        # Build model for FOR evaluation (time_steps=200 to match ITW training)
        model = LiqNNModel(input_size=128, hidden_size=32, out_size=1, time_steps=200).to(device)

        # Load weights
        ckpt = torch.load(target_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        acc, prec, rec, f1, auc, eer, n = eval_at_threshold(model, for_loader, device, threshold=0.5)
        print(f"folder={dir_name} | {os.path.basename(target_ckpt)} | N={n} | Acc@0.5={acc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f} AUC={auc:.4f} EER={eer:.4f}")
        with open(out_log, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}\t{dir_name}\t{os.path.basename(target_ckpt)}\t{n}\t0.5\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{eer:.4f}\t{auc:.4f}\n")


if __name__ == '__main__':
    main()


