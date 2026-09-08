"""Stage 4/5/6: CNN backbones, feature extraction and the classification head.

Two families are provided so the results chapter can quantify what transfer
learning is actually worth on this dataset:

  * BaselineCNN     -- built from scratch, no external data
  * timm backbones  -- ImageNet-pretrained EfficientNet / ResNet, fine-tuned
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src import config as C


# --------------------------------------------------------------------------
# Activations
#
# The activation is the non-linearity inside the network -- without one, a
# stack of convolutions collapses into a single linear operation no matter how
# deep it is. This is a different job from the optimizer, which decides how
# weights are updated; every run uses one of each.
# --------------------------------------------------------------------------
ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.01, inplace=True),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(inplace=True),      # a.k.a. Swish, used by EfficientNet
    "elu": lambda: nn.ELU(inplace=True),
    "mish": lambda: nn.Mish(inplace=True),
}


def make_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise KeyError(f"Unknown activation '{name}'. Options: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


# --------------------------------------------------------------------------
# Baseline: a plain CNN trained from scratch
# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv -> Norm -> Act -> Conv -> Norm -> Act -> MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, activation: str = "relu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            make_activation(activation),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            make_activation(activation),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """Five-stage convolutional network, ~2.5M parameters.

    Channel width doubles at each stage while spatial resolution halves, the
    standard VGG-style pattern. Global average pooling replaces the usual
    flatten+dense layers, which keeps the parameter count low -- important
    when training from scratch on only ~2.5k images.
    """

    def __init__(self, num_classes: int = C.NUM_CLASSES, dropout: float = 0.4,
                 activation: str = C.ACTIVATION):
        super().__init__()
        self.activation = activation
        self.features = nn.Sequential(
            ConvBlock(3, 32, activation),     # 224 -> 112
            ConvBlock(32, 64, activation),    # 112 -> 56
            ConvBlock(64, 128, activation),   # 56  -> 28
            ConvBlock(128, 256, activation),  # 28  -> 14
            ConvBlock(256, 512, activation),  # 14  -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            make_activation(activation),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming's gain is derived for the ReLU family; other
                # activations fall back to the closest supported nonlinearity.
                nonlinearity = ("leaky_relu" if self.activation == "leaky_relu"
                                else "relu")
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity=nonlinearity)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    @property
    def cam_target_layer(self) -> nn.Module:
        """Last activation of the final conv stage -- what Grad-CAM hooks into."""
        return self.features[-1].block[-2]


# --------------------------------------------------------------------------
# Transfer learning wrapper
# --------------------------------------------------------------------------
class TransferModel(nn.Module):
    """ImageNet-pretrained backbone with a fresh DR classification head."""

    def __init__(self, arch: str, num_classes: int = C.NUM_CLASSES,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        # num_classes=0 gives us the pooled feature vector (stage 5) rather
        # than the original 1000-way ImageNet head.
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=0
        )
        n_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x) -> torch.Tensor:
        """Stage 5 output: the high-level feature vector before classification."""
        return self.backbone(x)

    @property
    def cam_target_layer(self) -> nn.Module:
        """Final feature map of the backbone, chosen per architecture family."""
        if hasattr(self.backbone, "conv_head"):        # EfficientNet
            return self.backbone.conv_head
        if hasattr(self.backbone, "layer4"):           # ResNet
            return self.backbone.layer4[-1]
        # Generic fallback: last module that produces a 4D feature map.
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError(f"No Grad-CAM target layer found for {self.arch}")
        return convs[-1]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_model(model_key: str = C.DEFAULT_MODEL, pretrained: bool | None = None,
                activation: str | None = None) -> nn.Module:
    """Instantiate a model from its key in config.MODELS."""
    if model_key not in C.MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Options: {list(C.MODELS)}")
    cfg = C.MODELS[model_key]
    arch = cfg["arch"]
    use_pretrained = cfg["pretrained"] if pretrained is None else pretrained

    if arch == "baseline_cnn":
        return BaselineCNN(activation=activation or C.ACTIVATION)
    # Pretrained backbones keep their own activations -- swapping them would
    # invalidate the ImageNet weights they were trained with.
    return TransferModel(arch, pretrained=use_pretrained)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device() -> torch.device:
    """Prefer Apple Metal, fall back to CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
