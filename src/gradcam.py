"""Grad-CAM: visualise which retinal regions drove the predicted grade.

Gradient-weighted Class Activation Mapping backpropagates the score for a
chosen class to the last convolutional feature map. Channels are weighted by
their average gradient, summed, and passed through ReLU, leaving a coarse
heatmap over the areas that pushed the prediction up.

For DR this is the difference between "the model says grade 3" and "the model
says grade 3 because of these haemorrhages in the inferior temporal quadrant"
-- which is what makes the output defensible to a clinician.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src import config as C


class GradCAM:
    """Grad-CAM for any model exposing a `cam_target_layer` property."""

    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.target_layer = target_layer or model.cam_target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register()

    def _register(self):
        def forward_hook(module, inp, out):
            self.activations = out
            # Hooking the tensor itself is more reliable than a module
            # backward hook, which misbehaves for modules reused in a graph.
            if out.requires_grad:
                self._handles.append(
                    out.register_hook(lambda grad: setattr(self, "gradients", grad))
                )

        self._handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, input_tensor: torch.Tensor,
                 class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
        """Return (heatmap in [0,1], predicted class, softmax probabilities)."""
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM captured no gradients. Check the target layer."
            )

        # Weight each channel by its globally averaged gradient.
        grads = self.gradients[0]            # (C, H, W)
        acts = self.activations[0]           # (C, H, W)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=0))

        cam = cam.detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, class_idx, probs


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM map over the original image as a JET overlay."""
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if image_rgb.dtype != np.uint8:
        image_rgb = np.uint8(255 * np.clip(image_rgb, 0, 1))
    blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return blended


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation so the tensor can be shown as an image."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(C.STD) + np.array(C.MEAN)
    return np.uint8(255 * np.clip(img, 0, 1))
