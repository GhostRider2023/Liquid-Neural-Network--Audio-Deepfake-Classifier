import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from datetime import datetime
import math

# -------------------- Model Definition (Must match training) --------------------
def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt*k1/2)
    k3 = func(t + dt/2, y + dt*k2/2)
    k4 = func(t + dt, y + dt*k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

class LiquidTimeConstantCell(nn.Module):
    def __init__(self, hidden_size, tau_init=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.log_tau = nn.Parameter(torch.full((hidden_size,), math.log(tau_init)))
    def ode_func(self, t, h, current_features):
        tau = torch.exp(self.log_tau).clamp(min=0.01, max=10.0)
        dh = (-h + current_features) / tau
        return dh

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.conv5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        return x

import torch.nn.functional as F

class LiqNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, time_steps, ode_unfolds=1, dt=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.ode_unfolds = ode_unfolds
        self.dt = dt
        self.conv_extractor = ConvFeatureExtractor(input_size, hidden_size)
        self.ltc_cell = LiquidTimeConstantCell(hidden_size)
        self.final_fc = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        batch_size = x.size(0)
        conv_features = self.conv_extractor(x)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(min(self.time_steps, conv_features.size(-1))):
            current_features = conv_features[:, :, t]
            for _ in range(self.ode_unfolds):
                h = rk4_step(lambda t_ode, h_ode: self.ltc_cell.ode_func(t_ode, h_ode, current_features), 0, h, self.dt)
        return self.final_fc(h)

# -------------------- Dataset: MLAAD (MAILABS) --------------------
class MLAADMelDataset(Dataset):
    def __init__(self, base_dir, split="test", sample_rate=16000, n_mels=128, n_fft=1024, hop_length=160, duration=10):
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration
        self.items = []
        for label_name in ["real", "fake"]:
            label_dir = os.path.join(base_dir, split, label_name)
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith(".wav"):
                        label = 1 if label_name == "real" else 0
                        self.items.append((os.path.join(label_dir, fname), label))
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        waveform, sr = torchaudio.load(path)
        if waveform.dim() == 2: waveform = waveform.mean(dim=0)
        if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] < self.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[0]))
        else:
            waveform = waveform[:self.target_length]
        waveform = waveform / (torch.norm(waveform) + 1e-8)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        return mel, torch.tensor(label, dtype=torch.long)

# -------------------- Evaluation --------------------
def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))

def eval_at_threshold(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for mel, label in tqdm(loader, desc="Evaluating MLAAD"):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--mlaad_root', type=str, default=r"c:\Users\Shivaay Dhondiyal\Desktop\shivaay\coding\2_projects\9_deepfake_paper\m_ailabs_mlaad", help='MLAAD dataset root')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_log', type=str, default='MLAAD_eval_results.txt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LiqNNModel(input_size=128, hidden_size=32, out_size=1, time_steps=250).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    dataset = MLAADMelDataset(args.mlaad_root, split='test')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    acc, prec, rec, f1, auc, eer, n = eval_at_threshold(model, loader, device)
    
    timestamp = datetime.now().isoformat()
    header = "timestamp\tcheckpoint\tN\tacc\tprec\trec\tf1\teer\tauc\n"
    if not os.path.exists(args.out_log):
        with open(args.out_log, 'w') as f: f.write(header)
    
    with open(args.out_log, 'a') as f:
        f.write(f"{timestamp}\t{os.path.basename(args.checkpoint)}\t{n}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{eer:.4f}\t{auc:.4f}\n")
    
    print(f"MLAAD Eval Done: Acc={acc:.4f} EER={eer:.4f} AUC={auc:.4f}")

if __name__ == '__main__':
    main()
