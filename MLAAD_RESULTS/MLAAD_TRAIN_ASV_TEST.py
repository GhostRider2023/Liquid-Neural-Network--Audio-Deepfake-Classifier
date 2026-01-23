import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from datetime import datetime

# --- Dataset and Model Definitions (copied from original script) ---
def parse_protocol(protocol_path):
    file_to_label = {}
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            label = 1 if parts[-1] == 'bonafide' else 0
            file_to_label[file_id] = label
    return file_to_label

class EnhancedASVspoofDataset(Dataset):
    def __init__(self, flac_dir, protocol_dict, sample_rate=16000, duration=10, n_mels=128):
        self.flac_dir = flac_dir
        self.protocol_dict = protocol_dict
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.file_ids = list(protocol_dict.keys())
        self.fixed_length = sample_rate * duration
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=160, n_mels=n_mels
        )
    def __len__(self):
        return len(self.file_ids)
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.protocol_dict[file_id]
        flac_path = os.path.join(self.flac_dir, file_id + '.flac')
        waveform, sr = torchaudio.load(flac_path)
        waveform = waveform.mean(dim=0)
        waveform = waveform / (torch.norm(waveform) + 1e-8)
        if waveform.shape[0] < self.fixed_length:
            pad = self.fixed_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:self.fixed_length]
        mel = self.mel_transform(waveform)
        mel = torch.tensor(mel)
        mel = torch.log(mel + 1e-6)
        return mel, torch.tensor(label, dtype=torch.long)

# ------------ ODE Solvers ------------
def euler_step(func, t, y, dt):
    """Euler method for ODE solving"""
    return y + dt * func(t, y)

# %% [code] {"execution":{"iopub.status.busy":"2025-09-17T19:53:19.883057Z","iopub.execute_input":"2025-09-17T19:53:19.883623Z","iopub.status.idle":"2025-09-17T19:53:19.888141Z","shell.execute_reply.started":"2025-09-17T19:53:19.883596Z","shell.execute_reply":"2025-09-17T19:53:19.887335Z"}}
def rk4_step(func, t, y, dt):
    """4th order Runge-Kutta method for ODE solving"""
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt*k1/2)
    k3 = func(t + dt/2, y + dt*k2/2)
    k4 = func(t + dt, y + dt*k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

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

# ------------ Model (Improved LTC with 3 cells) ------------
import math

class LiquidTimeConstantCell(nn.Module):
    def __init__(self, input_size, hidden_size, tau_init=0.1, solver_type='rk4'):
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

    def forward(self, x, h):
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
    def __init__(self, n_mels, cnn_hidden=32):
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

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        return x

class LiqNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, time_steps, tau_init=0.1):
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

    def forward(self, x):
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

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def plot_confusion_matrix(y_true, y_score, threshold=0.5):
    y_pred = np.array(y_score) >= threshold
    y_true_np = np.array(y_true)
    cm = confusion_matrix(y_true_np, y_pred)
    cm_percent = cm.astype('float') / cm.sum() * 100
    cm_percent = np.round(cm_percent, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title("Confusion Matrix (in %)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def find_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    min_idx = np.argmin(np.abs(fpr - fnr))
    optimal_threshold = thresholds[min_idx]
    return optimal_threshold

# --- Paths (adjust if needed) ---
la_root = "/kaggle/input/asvpoof-2019-dataset/LA/LA"
test_flac_dir = os.path.join(la_root, "ASVspoof2019_LA_eval", "flac")
test_protocol_path = os.path.join(la_root, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")

# --- Load protocol and prepare balanced sampling ---
test_protocol_dict = parse_protocol(test_protocol_path)

real_ids = [fid for fid, lab in test_protocol_dict.items() if lab == 1]
fake_ids = [fid for fid, lab in test_protocol_dict.items() if lab == 0]
num_real = len(real_ids)
num_fake = len(fake_ids)
min_pairs = min(num_real, num_fake)
print(f"LA eval files: real={num_real}, fake={num_fake}. Running full-dataset evaluation.")


def make_loader_for_ids(selected_ids):
    filtered = {fid: test_protocol_dict[fid] for fid in selected_ids}
    ds = EnhancedASVspoofDataset(test_flac_dir, filtered)
    return DataLoader(ds, batch_size=2048, shuffle=False)

def make_full_loader():
    ds = EnhancedASVspoofDataset(test_flac_dir, test_protocol_dict)
    return DataLoader(ds, batch_size=2048, shuffle=False)

# --- Model config (can be different from training for testing) ---
input_size = 128
hidden_size = 32
out_size = 1
time_steps = 250  # Can be different from training (training uses 200, testing can use 250)
# Benefits of different time_steps:
# - Training: Shorter sequences (200) for faster training and memory efficiency
# - Testing: Longer sequences (250) for better performance evaluation
# - The model weights are compatible as long as input_size and hidden_size match

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_checkpoint(model, checkpoint_path, runs=5, results_path=None, threshold: float = 0.5):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    for run_idx in range(runs):
        rng = np.random.default_rng(seed=run_idx)
        chosen_real = rng.choice(real_ids, size=min_pairs, replace=False)
        chosen_fake = rng.choice(fake_ids, size=min_pairs, replace=False)
        selected = np.concatenate([chosen_real, chosen_fake])
        loader = make_loader_for_ids(selected)
        y_true, y_score = [], []
        with torch.no_grad():
            for mel, label in tqdm(loader, desc=f"Testing ({os.path.basename(checkpoint_path)}) run {run_idx+1}/{runs}"):
                mel = mel.to(device)
                out = model(mel)
                y_true.extend(label.numpy())
                y_score.extend(torch.sigmoid(out.squeeze()).cpu().numpy())
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        y_pred = (y_score >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        eer = compute_eer(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)
        print(f"\n{'-'*50}")
        print(f"RESULTS for {checkpoint_path} | balanced run {run_idx+1}/{runs} (N={2*min_pairs})")
        print(f"Threshold (fixed): {threshold:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"EER:       {eer:.4f}")
        print(f"AUC:       {auc_score:.4f}")
        print(f"{'-'*50}")
        if results_path is not None:
            try:
                with open(results_path, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{datetime.now().isoformat()}\t{os.path.basename(checkpoint_path)}\t"
                        f"run={run_idx+1}/{runs}\tN={2*min_pairs}\tseed={run_idx}\t"
                        f"thr={threshold:.4f}\tacc={accuracy:.4f}\tprec={precision:.4f}\trec={recall:.4f}\t"
                        f"f1={f1:.4f}\teer={eer:.4f}\tauc={auc_score:.4f}\n"
                    )
            except Exception:
                pass


def evaluate_checkpoint_full(model, checkpoint_path, results_path=None, threshold: float = 0.5, seed=None):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    loader = make_full_loader()
    y_true, y_score = [], []
    with torch.no_grad():
        for mel, label in tqdm(loader, desc=f"Testing FULL ({os.path.basename(checkpoint_path)})"):
            mel = mel.to(device)
            out = model(mel)
            y_true.extend(label.numpy())
            y_score.extend(torch.sigmoid(out.squeeze()).cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    eer = compute_eer(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    print(f"\n{'-'*50}")
    print(f"FULL-DATA RESULTS for {checkpoint_path} | N={len(y_true)}")
    print(f"Threshold (fixed): {threshold:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"EER:       {eer:.4f}")
    print(f"AUC:       {auc_score:.4f}")
    print(f"{'-'*50}")
    if results_path is not None:
        try:
            seed_str = str(seed) if seed is not None else "NA"
            with open(results_path, 'a', encoding='utf-8') as f:
                f.write(
                    f"{datetime.now().isoformat()}\t{os.path.basename(checkpoint_path)}\t"
                    f"run=full\tN={len(y_true)}\tseed={seed_str}\t"
                    f"thr={threshold:.4f}\tacc={accuracy:.4f}\tprec={precision:.4f}\trec={recall:.4f}\t"
                    f"f1={f1:.4f}\teer={eer:.4f}\tauc={auc_score:.4f}\n"
                )
        except Exception:
            pass

# --- Create model once ---
model = LiqNNModel(input_size=input_size, hidden_size=hidden_size, out_size=out_size, time_steps=time_steps).to(device)

# --- Discover last checkpoints in repeat folders under deepfake3/FOR_logs ---
logs_root = "/kaggle/input/mlaad_logs_final/pytorch/default/1/FOR_logs_Final"
if os.path.isdir(logs_root):
    all_dirs = [d for d in sorted(os.listdir(logs_root)) if os.path.isdir(os.path.join(logs_root, d))]
    repeat_dirs = sorted([d for d in all_dirs if d.startswith('repeat')])
    if not repeat_dirs:
        print(f"No repeat* folders found under: {logs_root}")
    else:
        # Evaluate the latest checkpoint in every repeat folder (deterministic order)
        results_txt_path = os.path.join("/kaggle/working", "LA_eval_results.txt")
        if not os.path.exists(results_txt_path):
            with open(results_txt_path, 'w', encoding='utf-8') as f:
                f.write("timestamp\tcheckpoint\trun\tN\tseed\tthr\tacc\tprec\trec\tf1\teer\tauc\n")
        # Seeds to exclude
        excluded_seeds = { 101,12345, 17, 20000,256}
        
        for target_rep in repeat_dirs:
            # Extract seed from folder name (e.g., "repeat42" -> 42)
            try:
                seed = int(target_rep.replace('repeat', ''))
            except ValueError:
                print(f"Could not extract seed from folder name: {target_rep}")
                continue
            
            # Skip excluded seeds
            if seed in excluded_seeds:
                print(f"Skipping excluded seed: {seed}")
                continue
                
            rep_path = os.path.join(logs_root, target_rep)
            pths = glob.glob(os.path.join(rep_path, '*.pth'))
            if not pths:
                print(f"No checkpoints in: {rep_path}")
                continue
            last_ckpt = max(pths, key=lambda p: os.path.getmtime(p))
            print(f"\nRunning full-dataset evaluation for {target_rep} (seed={seed})...")
            evaluate_checkpoint_full(model, last_ckpt, results_path=results_txt_path, threshold=0.5, seed=seed)
else:
    print(f"Logs root not found: {logs_root}")