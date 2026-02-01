import torch
import torch.nn as nn
import psutil
import time

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

class NetworkIDS:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = NetworkIDSModel(checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.scaler = checkpoint["scaler"]
        self.input_dim = checkpoint["input_dim"]

    def _capture_features(self):
        io1 = psutil.net_io_counters()
        time.sleep(1)
        io2 = psutil.net_io_counters()

        features = [
            io2.packets_sent - io1.packets_sent,
            io2.packets_recv - io1.packets_recv,
            io2.bytes_sent - io1.bytes_sent,
            io2.bytes_recv - io1.bytes_recv,
            1
        ]

        return features + [0] * (self.input_dim - len(features))

    def predict(self):
        features = self._capture_features()

        # Layer‑1 rule: trusted low traffic
        if sum(features[:4]) < 5000:
            return 0.1

        X = self.scaler.transform([features])
        tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor)).item()

        return prob
