import torch
import torch.nn as nn
import psutil
import time
import os

# ---------------- MODEL ----------------
class NetworkIDSModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


# ---------------- IDS ----------------
class NetworkIDS:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")

        self.model = NetworkIDSModel(checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = checkpoint["scaler"]
        self.input_dim = checkpoint["input_dim"]

    # -------- Feature Capture --------
    def _capture_features(self):
        io1 = psutil.net_io_counters()
        time.sleep(1)
        io2 = psutil.net_io_counters()

        features = [
            io2.packets_sent - io1.packets_sent,
            io2.packets_recv - io1.packets_recv,
            io2.bytes_sent - io1.bytes_sent,
            io2.bytes_recv - io1.bytes_recv,
            1  # protocol flag (dummy)
        ]

        return features + [0] * (self.input_dim - len(features))

    # -------- VPN / Foreign Network Heuristic --------
    def _vpn_or_foreign_suspected(self, traffic_volume):
        """
        Heuristic:
        - Very high packet + byte count in short time
        - Typical of VPNs / proxies / bot traffic
        """
        if traffic_volume > 150_000:
            return True
        return False

    # -------- MAIN PREDICT --------
    def predict(self):
        features = self._capture_features()
        traffic_volume = sum(features[:4])

        # ---------------- LAYER 1 ----------------
        # Normal home / office user
        if traffic_volume < 10_000:
            return 0.1  # SAFE → ALLOW

        # ---------------- LAYER 2 ----------------
        # Cloud / Render / shared network traffic
        if traffic_volume < 60_000:
            return 0.45  # WARNING (never block)

        # ---------------- LAYER 3 ----------------
        # VPN / foreign / suspicious burst
        if self._vpn_or_foreign_suspected(traffic_volume):
            return 0.95  # BLOCK

        # ---------------- ML DECISION ----------------
        X = self.scaler.transform([features])
        tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor)).item()

        return prob
