"""Zero-shot bird classification with BioCLIP 2.5 (open_clip).

Label text follows the BioCLIP authors' convention (Imageomics/pybioclip +
Imageomics/bioclip-2): the label is the species' scientific name, optionally
joined to its common name as "<scientific> with common name <common>" (their
"sci_com" text type), and each label is wrapped in the 80 OpenAI ImageNet
templates whose text embeddings are mean-pooled per label (their
CustomLabelsClassifier). Logits are logit_scale * img . text, softmaxed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from .templates import TEMPLATE

MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"


def label_text(scientific: str, common: str | None) -> str:
    """BioCLIP 'sci_com' text type: sci name + common name when available."""
    if common:
        return f"{scientific} with common name {common}"
    return scientific


class BioCLIP:
    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; lifer requires the GPU")
        import open_clip

        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self._logit_scale = None

    def _encode_labels(self, labels: list[str], batch: int = 32) -> torch.Tensor:
        """Mean-pooled, normalized text embedding per label (n_labels, dim)."""
        feats = []
        with torch.no_grad():
            for i in range(0, len(labels), batch):
                chunk = labels[i:i + batch]
                # one batch of (n_labels x 80) sequences, then pool per label
                all_txt = [t(lab) for lab in chunk for t in TEMPLATE]
                tok = self.tokenizer(all_txt).to(self.device)
                f = self.model.encode_text(tok)
                f = F.normalize(f, dim=-1)
                f = f.view(len(chunk), len(TEMPLATE), -1).mean(dim=1)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f)
        return torch.cat(feats, dim=0)

    def precompute_labels(self, labels: list[str]) -> torch.Tensor:
        """Encode the label set once; reuse for every crop batch.

        (Re-encoding ~300 labels x 80 templates per encounter would dominate
        the run, so callers precompute and pass this into `probabilities`.)
        """
        return self._encode_labels(labels)

    def probabilities(self, crops: list[Image.Image], txt: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        """(n_crops, n_labels) class probabilities for the given crops.

        `txt` is the (n_labels, dim) matrix from `precompute_labels`.
        `temperature` divides the logits before the softmax; >1 flattens an
        overconfident distribution (the raw model saturates at ~0.99+, so
        the pipeline default is 3.0; see REPORT.md Task 02).
        """
        if self._logit_scale is None:
            with torch.no_grad():
                self._logit_scale = self.model.logit_scale.exp().to(self.device)
        txt = txt.to(self.device)
        imgs = torch.stack([self.preprocess(c.convert("RGB")) for c in crops]).to(self.device)
        with torch.no_grad():
            imgf = self.model.encode_image(imgs)
            imgf = F.normalize(imgf, dim=-1)
            logits = self._logit_scale * (imgf @ txt.T)
            if temperature != 1.0:
                logits = logits / temperature
            return F.softmax(logits, dim=-1).cpu()
