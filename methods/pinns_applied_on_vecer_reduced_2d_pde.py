from __future__ import annotations

import csv
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from methods.vecer_dimension_reduction import vecer_fixed_asian_call_adaptive


@dataclass
class Config:

    T: float = 1.0            # maturity
    z_min: float = -1.0       # left end of the z domain (u ~ 0 there); z0 = 1 - K/S0
    z_max: float = 1.0        # right end; must be >= 1 so the right boundary value is exact
    r_max: float = 0.5
    sigma_max: float = 1.0

    # network
    width: int = 160          # nodes per hidden layer
    depth: int = 4            # number of hidden layers
    normalize_inputs: bool = True   # map each input to [-1, 1]

    # optimisation
    epochs: int = 50_000
    lr: float = 1e-3          # cosine annealed to 0
    n_pde: int = 1000         # collocation points per iteration
    n_bc: int = 100           # points per Terminal Condition / Boundary Condition face per iteration
    w_pde: float = 1.0        # weight w on the PDE loss

    # misc
    seed: int = 0
    dtype: str = "float32"
    log_every: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.z_max < 1.0:
            raise ValueError("z_max must be >= 1, otherwise the right boundary value is not exact")


def _bounds(cfg: Config, device=None, dtype=None):
    """Lower and upper ends of the input box, columns (t, z, r, sigma)."""
    lower = torch.tensor([0.0, cfg.z_min, 0.0, 0.0], device=device, dtype=dtype)
    upper = torch.tensor([cfg.T, cfg.z_max, cfg.r_max, cfg.sigma_max], device=device, dtype=dtype)
    return lower, upper


class PINN(nn.Module):
    """MLP u_theta(t, z, r, sigma) with GELU activations."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        layers, d_in = [], 4
        for _ in range(cfg.depth):
            layers += [nn.Linear(d_in, cfg.width), nn.GELU()]
            d_in = cfg.width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

        lower, upper = _bounds(cfg)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x columns: t, z, r, sigma
        if self.cfg.normalize_inputs:
            x = 2.0 * (x - self.lower) / (self.upper - self.lower) - 1.0
        return self.net(x)


def q(t: torch.Tensor, cfg: Config) -> torch.Tensor:
    return 1.0 - t / cfg.T


def bc_value_right(t, z, r, cfg):
    """Exact solution for z >= q(t): u = e^{-r tau} z + (1 - e^{-r tau}(1 + r tau)) / (r T)."""
    tau = cfg.T - t
    disc = torch.exp(-r * tau)
    small = r.abs() < 1e-6
    r_safe = torch.where(small, torch.ones_like(r), r)
    a = torch.where(small,
                    r * tau**2 / (2.0 * cfg.T),                         # limit r -> 0
                    (1.0 - disc * (1.0 + r * tau)) / (r_safe * cfg.T))
    return disc * z + a


def sample_box(n: int, cfg: Config, device, dtype) -> torch.Tensor:
    """Uniform points in [0,T] x [z_min,z_max] x [0,r_max] x [0,sigma_max]."""
    lower, upper = _bounds(cfg, device, dtype)
    return lower + torch.rand(n, 4, device=device, dtype=dtype) * (upper - lower)


def sample_data(cfg: Config, device, dtype):
    """Terminal condition and the two boundary conditions, n_bc points each."""
    xs, ys = [], []
    n = cfg.n_bc

    # terminal condition t = T:  u = z^+
    x = sample_box(n, cfg, device, dtype)
    x[:, 0] = cfg.T
    xs.append(x)
    ys.append(torch.relu(x[:, 1:2]))

    # z = z_min:  u = 0 (approximate)
    x = sample_box(n, cfg, device, dtype)
    x[:, 1] = cfg.z_min
    xs.append(x)
    ys.append(torch.zeros(n, 1, device=device, dtype=dtype))

    # z = z_max:  exact linear solution
    x = sample_box(n, cfg, device, dtype)
    x[:, 1] = cfg.z_max
    xs.append(x)
    ys.append(bc_value_right(x[:, 0:1], x[:, 1:2], x[:, 2:3], cfg))

    return torch.cat(xs), torch.cat(ys)


def pde_residual(model: PINN, x: torch.Tensor) -> torch.Tensor:
    """u_t + r (q - z) u_z + 1/2 sigma^2 (q - z)^2 u_zz at points x."""
    x = x.clone().requires_grad_(True)
    u = model(x)
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t, u_z = g[:, 0:1], g[:, 1:2]
    u_zz = torch.autograd.grad(u_z.sum(), x, create_graph=True)[0][:, 1:2]

    t, z, r, sigma = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    d = q(t, model.cfg) - z
    return u_t + r * d * u_z + 0.5 * sigma**2 * d**2 * u_zz


def train(cfg: Config, out_dir: str | Path) -> PINN:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)

    model = PINN(cfg).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.0)

    with open(out / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params}  epochs={cfg.epochs}")
    t0 = time.time()

    with open(out / "loss_log.csv", "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "loss", "loss_pde", "loss_data", "lr", "seconds"])

        for epoch in range(1, cfg.epochs + 1):
            x_pde = sample_box(cfg.n_pde, cfg, device, dtype)
            x_d, y_d = sample_data(cfg, device, dtype)

            loss_pde = pde_residual(model, x_pde).pow(2).mean()
            loss_data = (model(x_d) - y_d).pow(2).mean()
            loss = cfg.w_pde * loss_pde + loss_data

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
                secs = time.time() - t0
                lr_now = sched.get_last_lr()[0]
                writer.writerow([epoch, loss.item(), loss_pde.item(), loss_data.item(), lr_now, secs])
                log.flush()
                print(f"{epoch:>8d}  loss={loss.item():.3e}  pde={loss_pde.item():.3e}  "
                      f"data={loss_data.item():.3e}  lr={lr_now:.2e}  t={secs:.0f}s")

    torch.save(model.state_dict(), out / "model.pt")
    print(f"saved to {out}/  (training time {time.time() - t0:.0f}s)")
    return model


def load_model(run_dir: str | Path, device: str | None = None) -> tuple[PINN, Config]:
    run = Path(run_dir)
    with open(run / "config.json") as f:
        cfg = Config(**json.load(f))
    if device is not None:
        cfg.device = device
    elif not torch.cuda.is_available():
        cfg.device = "cpu"
    model = PINN(cfg).to(device=cfg.device, dtype=getattr(torch, cfg.dtype))
    model.load_state_dict(torch.load(run / "model.pt", map_location=cfg.device))
    model.eval()
    return model, cfg


# Pricing and Greeks in the original variables


def _value(model, cfg, S, K, r, sigma, t, I):
    """V = S * u(t, z) with z = q(t) + (I/T - K)/S. All arguments are tensors of shape (n,)."""
    z = q(t, cfg) + (I / cfg.T - K) / S
    x = torch.stack([t, z, r, sigma], dim=1)

    lower, upper = _bounds(cfg, x.device, x.dtype)
    outside = ((x < lower) | (x > upper)).any(dim=0)
    if outside.any():
        names = [nm for nm, o in zip(["t", "z", "r", "sigma"], outside.tolist()) if o]
        warnings.warn(f"inputs outside the training box: {names}; the result is an extrapolation")
    return S * model(x).squeeze(1)


def _tensors(cfg, S, K, r, sigma, t, I, requires_grad=False):
    dtype = getattr(torch, cfg.dtype)
    S = torch.as_tensor(np.asarray(S, dtype=float), dtype=dtype, device=cfg.device).reshape(-1)
    args = [S] + [torch.full_like(S, float(v)) for v in (K, r, sigma, t, I)]
    if requires_grad:
        for a in args:
            a.requires_grad_(True)
    return args


def price(model, cfg, S, K, r, sigma, t=0.0, I=0.0):
    """Price of the fixed-strike Asian call.

    S      spot price (scalar or array)
    K      strike
    r, sigma  rate and volatility (must lie in the training box)
    t      current time, 0 <= t <= T (default: inception)
    I      running integral of S from 0 to t, i.e. t * (average so far); 0 at inception
    """
    with torch.no_grad():
        return _value(model, cfg, *_tensors(cfg, S, K, r, sigma, t, I)).cpu().numpy()


def greeks(model, cfg, S, K, r, sigma, t=0.0, I=0.0):
    """Price and Greeks by automatic differentiation (arguments as in price).

        delta = dV/dS, gamma = d2V/dS2, vega = dV/dsigma, rho = dV/dr (per 1.00),
        dV_dt = dV/dt with I fixed, theta = dV/dt + S dV/dI (time passes, S unchanged),
        delta_I = dV/dI, dual_delta = dV/dK.
    The derivatives are taken directly with respect to S, K, I, t, r and sigma, so the
    change of variables to z is handled by autograd.
    """
    S_, K_, r_, sig_, t_, I_ = _tensors(cfg, S, K, r, sigma, t, I, requires_grad=True)
    V = _value(model, cfg, S_, K_, r_, sig_, t_, I_)
    dS, dK, dr, dsig, dt, dI = torch.autograd.grad(V.sum(), [S_, K_, r_, sig_, t_, I_],
                                                   create_graph=True)
    d2S = torch.autograd.grad(dS.sum(), S_)[0]

    out = {
        "price": V,
        "delta": dS,
        "gamma": d2S,
        "dV_dt": dt,
        "theta": dt + S_ * dI,
        "vega": dsig,
        "rho": dr,
        "delta_I": dI,
        "dual_delta": dK,
    }
    return {name: value.detach().cpu().numpy() for name, value in out.items()}


def fd_reference(S, K, T, r, sigma, M=800, N=200):
    """Price from the finite-difference Vecer solver, for comparison (t = 0 only)."""
    return np.array([vecer_fixed_asian_call_adaptive(float(s), K, T, r, sigma, M, N)
                     for s in np.atleast_1d(S)])