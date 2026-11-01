"""A small PatchCore (Roth et al., CVPR 2022) that runs on a CPU.

Backbone: ImageNet-pretrained ResNet-18, features from layer2 and layer3.
Each image becomes 28 x 28 = 784 patch vectors of 384 numbers.
Memory bank: a greedy coreset (1 percent) of all training patches.
Image score: the largest distance from any patch to its nearest memory-bank patch.
score() is differentiable, so FGSM and PGD can attack it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from common import normalize


class Backbone(nn.Module):
    """ResNet-18 up to layer3. Input: normalised image. Output: (layer2, layer3) feature maps."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        # Frozen: the pretrained weights are never changed.
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        return f2, f3


def embed(f2, f3):
    """Feature maps -> patch vectors (B, 784, 384). 3x3 averaging gives each patch some local context."""
    f2 = F.avg_pool2d(f2, 3, 1, 1)
    f3 = F.avg_pool2d(f3, 3, 1, 1)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([f2, f3], 1).flatten(2).transpose(1, 2)


def nearest_distances(patches, bank, chunk=4096):
    """Distance from every patch to its nearest memory-bank vector. (B, P, D) -> (B, P)."""
    b, p, d = patches.shape
    flat = patches.reshape(-1, d)
    out = [torch.cdist(flat[i:i + chunk], bank).min(1).values for i in range(0, len(flat), chunk)]
    return torch.cat(out).reshape(b, p)


def greedy_coreset(feats, ratio, proj_dim=128, seed=0):
    """Pick ratio * N patches that cover the feature space (greedy k-centre on a random projection)."""
    n = max(1, int(len(feats) * ratio))
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(feats.shape[1], proj_dim, generator=g) / proj_dim ** 0.5
    z = feats @ proj
    idx = [int(torch.randint(len(z), (1,), generator=g))]
    mind = ((z - z[idx[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        idx.append(i)
        mind = torch.minimum(mind, ((z - z[i]) ** 2).sum(1))
    return feats[torch.tensor(idx)].clone()


class PatchCore:
    def __init__(self, backbone=None, bank=None):
        self.backbone = backbone if backbone is not None else Backbone()
        self.bank = bank

    @torch.no_grad()
    def patch_features(self, x01, batch=8):
        out = []
        for i in range(0, len(x01), batch):
            f2, f3 = self.backbone(normalize(x01[i:i + batch]))
            out.append(embed(f2, f3))
        return torch.cat(out)

    def fit(self, x01, coreset_ratio=0.01):
        feats = self.patch_features(x01).reshape(-1, 384)
        self.n_train_patches = len(feats)
        self.bank = greedy_coreset(feats, coreset_ratio)
        return self

    def score(self, x01):
        """Image anomaly scores (B,). Differentiable with respect to x01."""
        f2, f3 = self.backbone(normalize(x01))
        return nearest_distances(embed(f2, f3), self.bank).max(1).values

    @torch.no_grad()
    def score_batched(self, x01, batch=8):
        return torch.cat([self.score(x01[i:i + batch]) for i in range(0, len(x01), batch)])


def fgsm(model, x01, labels, eps, mode="worst_case", batch=8):
    """One-step FGSM in pixel space (0..1), clipped back to valid pixels.

    worst_case : good images pushed up (false alarms), defective pushed down (missed defects)
    evasion    : only defective images are changed, pushed towards "normal"
    naive_up   : every image pushed to look more anomalous
    """
    if eps == 0:
        return x01.clone()
    out = []
    for i in range(0, len(x01), batch):
        x = x01[i:i + batch].clone().requires_grad_(True)
        y = labels[i:i + batch].float()
        if mode == "worst_case":
            direction = 1 - 2 * y            # good: +1, defect: -1
        elif mode == "evasion":
            direction = -y                   # good: 0 (untouched), defect: -1
        elif mode == "naive_up":
            direction = torch.ones_like(y)
        else:
            raise ValueError(mode)
        (model.score(x) * direction).sum().backward()
        out.append((x01[i:i + batch] + eps * x.grad.sign()).clamp(0, 1).detach())
    return torch.cat(out)


def pgd(model, x01, labels, eps, steps=10, batch=8):
    """Projected Gradient Descent (Madry et al., 2018): many small worst-case steps, kept within eps."""
    if eps == 0:
        return x01.clone()
    alpha = eps / 4
    out = []
    for i in range(0, len(x01), batch):
        x0 = x01[i:i + batch]
        direction = 1 - 2 * labels[i:i + batch].float()      # good: +1, defect: -1
        x = x0.clone()
        for _ in range(steps):
            x.requires_grad_(True)
            (model.score(x) * direction).sum().backward()
            with torch.no_grad():
                x = x + alpha * x.grad.sign()
                x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0, 1)
        out.append(x.detach())
    return torch.cat(out)
