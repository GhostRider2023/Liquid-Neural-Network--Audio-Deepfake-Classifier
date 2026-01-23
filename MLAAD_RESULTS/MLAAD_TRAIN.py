import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F
from torch.amp import autocast,GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import math


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
    
    # Handle empty dataset case
    if len(y_true) == 0:
        print("Warning: Empty dataset encountered!")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Handle AUC calculation for edge cases
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError as e:
        print(f"Warning: AUC calculation failed: {e}")
        auc = 0.0
    
    # Handle EER calculation for edge cases
    try:
        eer = compute_eer(y_true, y_score)
    except (ValueError, RuntimeError) as e:
        print(f"Warning: EER calculation failed: {e}")
        eer = 0.0
    
    return acc, prec, rec, f1, auc, eer


# -------------------- Dataset: MLAAD /MAILABS --------------------


class MLAADMelDataset(Dataset):
    """MLAAD dataset processed like ASVspoof LA (Mel features, 10s audio, normalized)."""
    def __init__(self, base_dir: str, split: str = "train",
                 sample_rate: int = 16000, n_mels: int = 128,
                 n_fft: int = 512, hop_length: int = 80, duration: int = 10):
        """
        base_dir: path to MLAAD root folder (expects split/real, split/fake)
        split: 'train', 'val', or 'test'
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration  # 10s @ 16kHz
        self.items = []

        # Gather files + labels (real=1, fake=0)
        for label_name in ["real", "fake"]:
            label_dir = os.path.join(base_dir, split, label_name)
            if not os.path.isdir(label_dir):
                raise FileNotFoundError(f"Directory not found: {label_dir}")
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(".wav"):
                    label = 1 if label_name == "real" else 0
                    self.items.append((os.path.join(label_dir, fname), label))

        # Mel spectrogram transform (like ASVspoof)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, label = self.items[idx]

        # Load audio
        waveform, sr = torchaudio.load(file_path)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)  # mono

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # Pad or truncate to exactly 10s
        if waveform.shape[0] < self.target_length:
            pad_len = self.target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:self.target_length]

        # Normalize waveform (L2 norm)
        waveform = waveform / (torch.norm(waveform) + 1e-8)

        # Mel spectrogram + log scaling
        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-6)

        return mel, torch.tensor(label, dtype=torch.long)
def count_classes(dataset):
    """Count the number of samples in each class (real vs fake)."""
    real_count = 0
    fake_count = 0
    for _, label in dataset.items:
        if label == 1:  # real
            real_count += 1
        else:  # fake
            fake_count += 1
    return real_count, fake_count


def make_mlaad_loaders(base_dir: str, batch_size: int, seed: int):
    """
    Create DataLoaders for MLAAD dataset (train/val/test).
    Expects structure:
      base_dir/
        train/real, train/fake
        val/real, val/fake
        test/real, test/fake
    """
    effective_workers = 0 if os.name == 'nt' else 4

    # Ensure deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))

    # Datasets
    train_set = MLAADMelDataset(base_dir, split="train")
    val_set = MLAADMelDataset(base_dir, split="val")
    test_set = MLAADMelDataset(base_dir, split="test")

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )
    return train_loader, val_loader, test_loader


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


# -------------------- Model (Improved LTC with single cell) --------------------
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
        
        # Synaptic dynamics (simplified version of the research LTC)
        synaptic_activation = torch.sigmoid((h.unsqueeze(1) - self.mu) / self.sigma)
        synaptic_current = torch.sum(synaptic_activation * self.erev, dim=1)
        
        # Leak current
        leak_current = gleak * (self.vleak - h)
        
        # Total current
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


def train_one_epoch(model, loader, optimizer, device, criterion, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for mel, label in tqdm(loader, desc="Training"):
        mel = mel.to(device)
        label = label.to(device).float()
        
        optimizer.zero_grad()
        
        with autocast("cuda"):
            out = model(mel)
            loss = criterion(out.squeeze(), label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train LiqNN model on MLAAD dataset')
    parser.add_argument('--mlaad_root', type=str, default="/kaggle/input/m-ailabs-mlaadru-uk-pl/m_ailabs_mlaad", help='/kaggle/input/m-ailabs-mlaadru-uk-pl/m_ailabs_mlaad')
    parser.add_argument('--save_root', type=str, default='MLAAD_logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_repeat', type=int, default=0, help='Starting repeat index')
    parser.add_argument('--seeds', type=int, nargs='+', default=[5555, 6789, 10007, 999, 42, 63, 89, 101, 256, 333, 512, 777, 4096, 8080, 9001, 12345, 20000], help='List of seeds to run; folders will be named repeat<seed>')
    parser.add_argument('--phase1_epochs', type=int, default=5, help='Phase 1: epochs from 1e-3 to 1e-5')
    parser.add_argument('--phase2_epochs', type=int, default=5, help='Phase 2: epochs from 1e-5 to 1e-6')
    parser.add_argument('--lr_phase1_start', type=float, default=1e-3, help='Phase 1 start learning rate')
    parser.add_argument('--lr_phase1_end', type=float, default=1e-5, help='Phase 1 end learning rate')
    parser.add_argument('--lr_phase2_start', type=float, default=1e-5, help='Phase 2 start learning rate')
    parser.add_argument('--lr_phase2_end', type=float, default=1e-6, help='Phase 2 end learning rate')
    parser.add_argument('--eval_interval', type=int, default=5)
    
    args, _ = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve paths relative to script location
    script_dir = os.getcwd()  
    save_root_abs = args.save_root if os.path.isabs(args.save_root) else os.path.join(script_dir, args.save_root)
    mlaad_root_abs = args.mlaad_root if os.path.isabs(args.mlaad_root) else os.path.join(script_dir, args.mlaad_root)

    if not os.path.isdir(mlaad_root_abs):
        print(f"MLAAD dataset not found: {mlaad_root_abs}")
        return
    
    # Validate dataset structure
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            path = os.path.join(mlaad_root_abs, split, label)
            if not os.path.isdir(path):
                print(f"Missing directory: {path}")
                return

    # Count files
    def count_wavs(path):
        return len([f for f in os.listdir(path) if f.lower().endswith(".wav")])

    print(f"MLAAD Dataset Summary:")
    for split in ["train", "val", "test"]:
        real_count = count_wavs(os.path.join(mlaad_root_abs, split, "real"))
        fake_count = count_wavs(os.path.join(mlaad_root_abs, split, "fake"))
        print(f"  {split.capitalize()}: {real_count} real, {fake_count} fake")

    os.makedirs(save_root_abs, exist_ok=True)

    def build_model():
        return LiqNNModel(input_size=128, hidden_size=32, out_size=1, time_steps=200).to(device)

    # Training loop for each seed
    for repeat_idx, seed in enumerate(args.seeds[args.start_repeat:], start=args.start_repeat):
        print(f"\n{'='*80}")
        print(f"Repeat {repeat_idx+1}/{len(args.seeds)} | Seed: {seed}")
        print(f"{'='*80}")

        # Set seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create repeat directory
        repeat_dir = os.path.join(save_root_abs, f'repeat{seed}')
        os.makedirs(repeat_dir, exist_ok=True)
        
        # Create log file for this repeat
        log_file = os.path.join(repeat_dir, f'training_log_seed{seed}.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('epoch\tphase\tlr\ttrain_loss\tval_acc\tval_prec\tval_rec\tval_f1\tval_auc\tval_eer\n')

        # Prepare data loaders (train, val, test)
        train_loader, val_loader, test_loader = make_mlaad_loaders(mlaad_root_abs, args.batch_size, seed)

        # Build model
        model = build_model()

        # Class balancing
        real_count, fake_count = count_classes(train_loader.dataset)
        pos_weight = fake_count / real_count
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

        # Mixed precision training
        scaler = GradScaler("cuda")
        # ---------------- Phase 1 ----------------
        print(f"\n{'='*60}")
        print(f"PHASE 1: {args.phase1_epochs} epochs | LR: {args.lr_phase1_start} -> {args.lr_phase1_end}")
        print(f"{'='*60}")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_phase1_start, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs, eta_min=args.lr_phase1_end)
        
        for epoch in range(1, args.phase1_epochs + 1):
            print("\n" + "-"*60)
            print(f"Phase 1 | Epoch {epoch}/{args.phase1_epochs} | lr={optimizer.param_groups[0]['lr']:.8f}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
            scheduler.step()
            print(f"Train Loss: {train_loss:.6f}")

            val_acc, val_prec, val_rec, val_f1, val_auc, val_eer = 0, 0, 0, 0, 0, 0
            if epoch % args.eval_interval == 0:
                val_acc, val_prec, val_rec, val_f1, val_auc, val_eer = eval_at_threshold(model, val_loader, device, threshold=0.5)
                print(f"Val@0.5: Acc={val_acc:.4f} | AUC={val_auc:.4f} | EER={val_eer:.4f} | F1={val_f1:.4f} | P={val_prec:.4f} | R={val_rec:.4f}")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch}\t1\t{optimizer.param_groups[0]['lr']:.8f}\t{train_loss:.6f}\t{val_acc:.4f}\t{val_prec:.4f}\t{val_rec:.4f}\t{val_f1:.4f}\t{val_auc:.4f}\t{val_eer:.4f}\n")

        torch.save(model.state_dict(), os.path.join(repeat_dir, 'phase1_epoch10.pth'))
        print(f"Phase 1 completed. Checkpoint saved: phase1_epoch10.pth")

        # ---------------- Phase 2 ----------------
        print(f"\n{'='*60}")
        print(f"PHASE 2: {args.phase2_epochs} epochs | LR: {args.lr_phase2_start} -> {args.lr_phase2_end}")
        print(f"{'='*60}")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_phase2_start, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs, eta_min=args.lr_phase2_end)
        
        for epoch in range(1, args.phase2_epochs + 1):
            print("\n" + "-"*60)
            print(f"Phase 2 | Epoch {epoch}/{args.phase2_epochs} | lr={optimizer.param_groups[0]['lr']:.8f}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
            scheduler.step()
            print(f"Train Loss: {train_loss:.6f}")

            val_acc, val_prec, val_rec, val_f1, val_auc, val_eer = eval_at_threshold(model, val_loader, device, threshold=0.5)
            print(f"Val@0.5: Acc={val_acc:.4f} | AUC={val_auc:.4f} | EER={val_eer:.4f} | F1={val_f1:.4f} | P={val_prec:.4f} | R={val_rec:.4f}")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{args.phase1_epochs + epoch}\t2\t{optimizer.param_groups[0]['lr']:.8f}\t{train_loss:.6f}\t{val_acc:.4f}\t{val_prec:.4f}\t{val_rec:.4f}\t{val_f1:.4f}\t{val_auc:.4f}\t{val_eer:.4f}\n")

        torch.save(model.state_dict(), os.path.join(repeat_dir, 'phase2_epoch40.pth'))
        print(f"Training completed for seed {seed}. Checkpoints and logs saved in: {repeat_dir}")



if __name__ == '__main__':
    main()

import shutil

# Zip the whole working directory
shutil.make_archive('/kaggle/working/my_working_dir', 'zip', '/kaggle/working')

