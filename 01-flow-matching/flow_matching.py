"""Flow matching on a 2D spiral: the five code blocks shown in episode 1."""
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Data
def make_spiral(n, turns=1.5, radius=3.0):
    s = torch.sqrt(torch.rand(n)) * 2 * math.pi * turns
    arm = (torch.rand(n) < 0.5).float() * math.pi
    x1 = torch.stack([s * torch.cos(s + arm), s * torch.sin(s + arm)], dim=1)
    return x1 / (2 * math.pi * turns) * radius + 0.05 * torch.randn(n, 2)


# 2. Network
class FlowMatcher(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.embed_x = nn.Linear(2, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, 2),
        )

    def embed_t(self, t):
        freqs = 1000 ** (torch.arange(0, self.dim, 2) / self.dim)
        return torch.cat([(t * freqs).sin(), (t * freqs).cos()], dim=-1)

    def forward(self, x, t):
        return self.mlp(self.embed_x(x) + self.embed_t(t))


# 3. Loss
def flow_matching_loss(model, x1):
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1), 1)
    xt = t * x1 + (1 - t) * x0
    v = model(xt, t)
    return torch.mean((v - (x1 - x0)) ** 2)


# 4. Training
print("training for 2000 epochs ...")

torch.manual_seed(0)
data = make_spiral(1000)
model = FlowMatcher()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

losses = []
for epoch in range(2000):
    total = 0
    for (x1,) in loader:
        loss = flow_matching_loss(model, x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    losses.append(total / len(loader))


# 5. Sampler
@torch.no_grad()
def sample(model, n=1000, steps=50):
    x = torch.randn(n, 2)
    dt = 1 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt)
        x = x + dt * model(x, t)
    return x


# Not shown in the video: print the result.
samples = sample(model)
print(f"final epoch loss: {losses[-1]:.4f}")
print(f"mean of last 100 epochs: {sum(losses[-100:]) / 100:.4f}")
print(f"generated {len(samples)} points with 50 Euler steps (plots are in notebook.ipynb)")
