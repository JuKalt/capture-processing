"""Local-only Hugging Face Sapiens2 adapter; no downloads or remote model code."""
from pathlib import Path
import numpy as np


class Sapiens2Predictor:
    def __init__(self, model_path, device=None):
        from capture_processing.utils.semantic_parsing import verify_model_assets
        verify_model_assets(model_path)
        import torch
        import transformers
        if transformers.__version__ != "5.20.2" or torch.__version__.split("+")[0] != "2.7.1":
            raise RuntimeError("Use the pinned sapiens2 environment from scripts/setup/setup_semantic_parsing_env.sh")
        from transformers import AutoImageProcessor, Sapiens2ForSemanticSegmentation
        from capture_processing.utils.semantic_schema import FINE
        self.torch = torch
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = Sapiens2ForSemanticSegmentation.from_pretrained(
            str(model_path), local_files_only=True).to(self.device).eval()
        if self.model.config.num_labels != len(FINE):
            raise ValueError('Expected the Sapiens2 dome29 segmentation checkpoint')

    def __call__(self, image, foreground, shape):
        import torch.nn.functional as F
        from PIL import Image
        # Predict on a padded person crop, then place the posterior on the working grid.
        yy, xx = np.nonzero(foreground)
        h, w = foreground.shape
        pad = max(8, int(.05 * max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)))
        box = (max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad),
               min(w, int(xx.max()) + pad + 1), min(h, int(yy.max()) + pad + 1))
        crop = Image.fromarray(image).crop(box)
        inputs = self.processor(images=crop, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits.float()
            sh, sw = shape
            x0, y0 = int(box[0] * sw / w), int(box[1] * sh / h)
            x1, y1 = max(x0 + 1, int(np.ceil(box[2] * sw / w))), max(y0 + 1, int(np.ceil(box[3] * sh / h)))
            p = F.interpolate(logits, size=(y1 - y0, x1 - x0), mode='bilinear',
                              align_corners=False).softmax(1)[0].permute(1, 2, 0).cpu().numpy()
        result = np.zeros((sh, sw, 29), dtype=np.float32)
        result[..., 0] = 1
        result[y0:y1, x0:x1] = p
        return result
