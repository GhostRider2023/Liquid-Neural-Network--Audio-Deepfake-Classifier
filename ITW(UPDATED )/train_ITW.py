import os
import math
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import random

# -------------------- Dataset --------------------
class WildCSVDataset(Dataset):
    def __init__(self, root: str, entries, target_sr: int = 16000, duration_s: int = 10, n_mels: int = 128, n_fft: int = 512, hop_length: int = 80):
        self.root = root
        self.entries = entries
        self.target_sr = target_sr
        self.fixed_length = target_sr * duration_s
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=512, hop_length=80, n_mels=n_mels
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rel_file, label = self.entries[idx]
        audio_path = rel_file if os.path.isabs(rel_file) else os.path.join(self.root, rel_file)
        waveform, sr = torchaudio.load(audio_path)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sr != self.mel.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.mel.sample_rate)
            waveform = resampler(waveform)
            sr = self.mel.sample_rate
        waveform = waveform / (torch.norm(waveform) + 1e-8)
        if waveform.shape[-1] < self.fixed_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.fixed_length - waveform.shape[-1]))
        else:
            waveform = waveform[: self.fixed_length]
        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-6)
        return mel, torch.tensor(label, dtype=torch.long)

def read_meta(csv_path: str):
    import csv
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            file_val = r.get('file')
            speaker = r.get('speaker')
            label = r.get('label')
            if not file_val or not label:
                continue
            if file_val.lower().endswith('.txt'):
                continue
            rows.append((file_val, speaker, label))
    return rows

def map_label_to_int(label_str: str) -> int:
    s = label_str.strip().lower().replace('-', '')
    if s == 'spoof':
        return 0
    if s.replace(' ', '') == 'bonafide' or s in ('bonafide', 'bonafied', 'bonafide '):
        return 1
    raise ValueError(f"Unknown label: {label_str}")

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

def semi_implicit_euler(func, t, y, dt, alpha=0.5):
    """Semi-implicit Euler method for stiff ODEs"""
    # Predictor step (explicit)
    y_pred = y + dt * func(t, y)
    
    # Corrector step (implicit)
    y_corr = y + dt * (alpha * func(t, y) + (1 - alpha) * func(t + dt, y_pred))
    
    return y_corr


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

# -------------------- Metrics --------------------
def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return float(eer)

def find_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    min_idx = np.argmin(np.abs(fpr - fnr))
    optimal_threshold = thresholds[min_idx]
    return float(optimal_threshold)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for mel, label in tqdm(loader, desc="Evaluating", leave=False):
            mel = mel.to(device, non_blocking=True)
            out = model(mel)
            y_true.extend(label.numpy())
            y_score.extend(torch.sigmoid(out.squeeze()).cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thr = find_optimal_threshold(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    return acc, auc, eer, thr

def eval_at_threshold(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for mel, label in tqdm(loader, desc="Evaluating@thr", leave=False):
            mel = mel.to(device, non_blocking=True)
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
    return acc, prec, rec, f1, auc, eer

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, criterion: nn.Module, scaler: GradScaler, accumulation_steps: int = 1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step_idx, (mel, label) in enumerate(tqdm(loader, desc="Train", leave=True)):
        mel = mel.to(device, non_blocking=True)
        label = label.float().to(device, non_blocking=True)
        with autocast(enabled=torch.cuda.is_available()):
            out = model(mel)
            out = out.squeeze()
            loss = criterion(out, label)
            loss = loss / accumulation_steps
        if not torch.isfinite(loss):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        if (step_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * accumulation_steps
    return running_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description='Train LiqNN model on ITW dataset')
    parser.add_argument('--root', type=str, default=r"C:\Users\Shivaay Dhondiyal\Desktop\shivaay\coding\2_projects\9_deepfake_paper\deepfake2\release_in_the_wild")
    parser.add_argument('--csv', type=str, default='meta.csv')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--phase1_epochs', type=int, default=10)
    parser.add_argument('--phase2_epochs', type=int, default=40)
    parser.add_argument('--lr_phase1_start', type=float, default=1e-4)
    parser.add_argument('--lr_phase1_end', type=float, default=1e-5)
    parser.add_argument('--lr_phase2_start', type=float, default=1e-5)
    parser.add_argument('--lr_phase2_end', type=float, default=1e-6)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--start_repeat', type=int, default=0, help='Start index in --seeds list (0-based).')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 2025, 5555, 6789, 10007, 999, 777, 4096, 8080, 9001, 12345, 20000], help='List of seeds to run; folders will be named repeat<seed>')
    parser.add_argument('--save_root', type=str, default='deepfake3/ITW_logs2')
    parser.add_argument('--balanced_subset_size', type=int, default=None, help='Number of samples per class (spoof and bonafide). If None, uses min(spoof_count, bonafide_count). Total samples = 2 * balanced_subset_size')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load dataset
    dataset_root = os.path.abspath(args.root)
    csv_path = os.path.join(dataset_root, args.csv) if not os.path.isabs(args.csv) else args.csv
    meta = read_meta(csv_path)
    
    # Split data into train/val/test
    from sklearn.model_selection import train_test_split
    files = [m[0] for m in meta]
    labels = [map_label_to_int(m[2]) for m in meta]
    
    # Create balanced subset with equal numbers of spoof and bonafide
    print(f"Creating balanced subset...")
    
    # Separate files by label
    spoof_files = [f for f, l in zip(files, labels) if l == 0]
    bonafide_files = [f for f, l in zip(files, labels) if l == 1]
    
    print(f"Original dataset: {len(spoof_files)} spoof, {len(bonafide_files)} bonafide")
    
    # Use minimum count between spoof and bonafide, or the specified subset size if smaller
    min_count = min(len(spoof_files), len(bonafide_files))
    if args.balanced_subset_size is None:
        subset_size = min_count
    else:
        subset_size = min(args.balanced_subset_size, min_count)
    
    print(f"Using {subset_size} samples per class (min of: spoof={len(spoof_files)}, bonafide={len(bonafide_files)}, specified={args.balanced_subset_size})")
    
    # Randomly sample equal numbers from each class
    random.seed(42)  # For reproducibility
    
    spoof_subset = random.sample(spoof_files, subset_size)
    bonafide_subset = random.sample(bonafide_files, subset_size)
    
    # Combine and shuffle
    balanced_files = spoof_subset + bonafide_subset
    balanced_labels = [0] * subset_size + [1] * subset_size
    
    # Shuffle the combined data
    combined = list(zip(balanced_files, balanced_labels))
    random.shuffle(combined)
    files, labels = zip(*combined)
    files, labels = list(files), list(labels)
    
    print(f"Balanced subset: {subset_size} spoof, {subset_size} bonafide (total: {len(files)})")
    
    # Split into train+val and test first
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Split train+val into train and val
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=0.2, random_state=42, stratify=train_val_labels
    )
    
    # Create datasets
    train_entries = list(zip(train_files, train_labels))
    val_entries = list(zip(val_files, val_labels))
    test_entries = list(zip(test_files, test_labels))
    
    train_set = WildCSVDataset(dataset_root, train_entries)
    val_set = WildCSVDataset(dataset_root, val_entries)
    test_set = WildCSVDataset(dataset_root, test_entries)

    # Class balance for pos_weight (always 1.0 since we use balanced dataset)
    bonafide_count = sum(1 for lbl in train_labels + val_labels if lbl == 1)
    total_count = len(train_labels) + len(val_labels)
    spoof_count = total_count - bonafide_count
    pos_weight_value = 1.0  # Always 1.0 since we use balanced dataset
    print(f"Train+Val size: {total_count} | Bonafide: {bonafide_count} | Spoof: {spoof_count} | pos_weight: {pos_weight_value:.4f}")
    print(f"Using balanced dataset - pos_weight set to 1.0")

    def make_loaders(seed: int):
        effective_workers = 0 if os.name == 'nt' else 4
        # Ensure deterministic shuffling per repeat
        g = torch.Generator()
        g.manual_seed(seed)

        def _worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed % (2**32 - 1))

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=effective_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            generator=g,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=effective_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )
        return train_loader, test_loader

    def build_model():
        return LiqNNModel(input_size=128, hidden_size=32, out_size=1, time_steps=200).to(device)

    def run_one_seed(seed: int, ordinal_idx: int, total_runs: int):
        # Reproducible seed per run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        save_dir = os.path.join(args.save_root, f'repeat{seed}')
        os.makedirs(save_dir, exist_ok=True)
        train_loader, test_loader = make_loaders(seed)

        model = build_model()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_phase1_start)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Phase 1 scheduler
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs, eta_min=args.lr_phase1_end)

        print("="*60)
        print(f"RUN {ordinal_idx+1}/{total_runs} | seed={seed} | PHASE 1: {args.phase1_epochs} epochs (LR: {args.lr_phase1_start} -> {args.lr_phase1_end})")
        print("="*60)

        for epoch in range(1, args.phase1_epochs + 1):
            print("\n" + "-"*60)
            print(f"[Phase 1] Epoch {epoch}/{args.phase1_epochs} | lr={optimizer.param_groups[0]['lr']:.8f}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
            scheduler1.step()
            print(f"Train Loss: {train_loss:.6f}")

            # Test every eval_interval epochs at fixed threshold 0.5
            if epoch % args.eval_interval == 0:
                t_acc, t_prec, t_rec, t_f1, t_auc, t_eer = eval_at_threshold(model, test_loader, device, threshold=0.5)
                print(f"Test Acc@0.5={t_acc:.4f} | F1@0.5={t_f1:.4f} | P@0.5={t_prec:.4f} | R@0.5={t_rec:.4f} | AUC={t_auc:.4f} | EER={t_eer:.4f}")

            # Save checkpoint
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_dir, f'phase1_epoch{epoch:02d}.pth'))

        # Phase 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_phase2_start
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs, eta_min=args.lr_phase2_end)

        print("\n" + "="*60)
        print(f"RUN {ordinal_idx+1}/{total_runs} | seed={seed} | PHASE 2: {args.phase2_epochs} epochs (LR: {args.lr_phase2_start} -> {args.lr_phase2_end})")
        print("="*60)

        total_epochs = args.phase2_epochs
        for epoch in range(1, total_epochs + 1):
            print("\n" + "-"*60)
            print(f"[Phase 2] Epoch {epoch}/{total_epochs} | lr={optimizer.param_groups[0]['lr']:.8f}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
            scheduler2.step()
            print(f"Train Loss: {train_loss:.6f}")

            # Test every eval_interval epochs at fixed threshold 0.5
            absolute_epoch = args.phase1_epochs + epoch
            if absolute_epoch % args.eval_interval == 0 or epoch == total_epochs:
                t_acc, t_prec, t_rec, t_f1, t_auc, t_eer = eval_at_threshold(model, test_loader, device, threshold=0.5)
                print(f"Test Acc@0.5={t_acc:.4f} | F1@0.5={t_f1:.4f} | P@0.5={t_prec:.4f} | R@0.5={t_rec:.4f} | AUC={t_auc:.4f} | EER={t_eer:.4f}")

            # Save checkpoint
            torch.save({'epoch': args.phase1_epochs + epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_dir, f'phase2_epoch{epoch:02d}.pth'))

        # Final test pass at fixed threshold 0.5
        print("\nFINAL TEST EVALUATION @ threshold 0.5")
        t_acc, t_prec, t_rec, t_f1, t_auc, t_eer = eval_at_threshold(model, test_loader, device, threshold=0.5)
        print(f"Final Test Acc@0.5={t_acc:.4f} | F1@0.5={t_f1:.4f} | P@0.5={t_prec:.4f} | R@0.5={t_rec:.4f} | AUC={t_auc:.4f} | EER={t_eer:.4f}")

    all_seeds = args.seeds
    total_runs = len(all_seeds)
    for idx in range(args.start_repeat, total_runs):
        run_one_seed(all_seeds[idx], idx, total_runs)

if __name__ == '__main__':
    main()
