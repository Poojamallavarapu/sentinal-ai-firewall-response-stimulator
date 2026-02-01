import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.serialization import add_safe_globals

# -------------------------------------------------
# Allow sklearn objects (trusted checkpoint)
# -------------------------------------------------
add_safe_globals([TfidfVectorizer])

# -------------------------------------------------
# Prompt Injection Model (same as training)
# -------------------------------------------------
class PromptInjectionDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Prompt Detector (Layer 3)
# -------------------------------------------------
class PromptDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path

        # Load checkpoint (PyTorch 2.6+ safe loading)
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        # Load TF-IDF vectorizer and model params
        self.vectorizer = checkpoint["vectorizer"]
        self.input_dim = checkpoint["input_dim"]

        self.model = PromptInjectionDetector(self.input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print("✅ PROMPT MODEL LOADED:", self.model_path)
        print("✅ TF-IDF FEATURE SIZE:", self.input_dim)

    # -------------------------------------------------
    # Predict (returns confidence score)
    # -------------------------------------------------
    def predict(self, text: str) -> float:
        if not text or not text.strip():
            return 0.1  # low confidence for empty input

        # TF-IDF transform (same as training)
        vec = self.vectorizer.transform([text]).toarray()
        tensor = torch.tensor(vec, dtype=torch.float32)

        # Model inference
        with torch.no_grad():
            logit = self.model(tensor)
            prob = torch.sigmoid(logit).item()

        # Debug logs (optional, safe to keep)
        print("🔥 PROMPT:", text)
        print("📊 CONFIDENCE SCORE:", round(prob, 4))

        return prob
