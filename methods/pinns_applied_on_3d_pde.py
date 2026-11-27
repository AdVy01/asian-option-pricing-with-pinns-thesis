from __future__ import annotations

import csv
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

# Strike used in training. Fixed: other strikes come from rescaling in price().
K_TRAIN = 1.0

@dataclass
class Config:

    T: float = 1.0            # maturity
    S_max: float = 3.0
    I_max: float = 1.0        # must satisfy I_max / T >= K for Boundary Condition 3 to be exact
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

class PINN(nn.Module):
    """MLP V_theta(S, I, t, r, sigma) with GELU activations."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        layers, d_in = [], 5
        for _ in range(cfg.depth):
            layers += [nn.Linear(d_in, cfg.width), nn.GELU()]
            d_in = cfg.width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

        # input scaling to [-1, 1]; lower bounds are all 0
        upper = torch.tensor([cfg.S_max, cfg.I_max, cfg.T, cfg.r_max, cfg.sigma_max])
        self.register_buffer("upper", upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x columns: S, I, t, r, sigma
        if self.cfg.normalize_inputs:
            x = 2.0 * x / self.upper - 1.0
        return self.net(x)

def annuity(r: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """(1 - exp(-r tau)) / r, numerically safe for r -> 0 (limit = tau)."""
    small = r.abs() < 1e-6
    r_safe = torch.where(small, torch.ones_like(r), r)
    return torch.where(small, tau * (1.0 - 0.5 * r * tau), -torch.expm1(-r * tau) / r_safe)


def payoff(I: torch.Tensor, cfg: Config) -> torch.Tensor:
    return torch.relu(I / cfg.T - K_TRAIN)


def bc_value_S0(I, t, r, cfg):
    """S = 0: the asset stays at 0, so the average is frozen (exact)."""
    return torch.exp(-r * (cfg.T - t)) * payoff(I, cfg)


def bc_value_Smax(S, I, t, r, cfg):
    """S = S_max: option treated as deep in the money (approximate).

    Park et al. write S_max (1 - e^{-r(T-t)}) / r; the 1/T factor below makes
    it correct for T != 1 and is identical for their T = 1.
    """
    tau = cfg.T - t
    disc = torch.exp(-r * tau)
    return torch.relu(disc * (I / cfg.T - K_TRAIN) + S * annuity(r, tau) / cfg.T)


def bc_value_Imax(S, I, t, r, cfg):
    """I = I_max >= K T: the option is certain to finish in the money (exact)."""
    tau = cfg.T - t
    disc = torch.exp(-r * tau)
    return disc * (I / cfg.T - K_TRAIN) + S * annuity(r, tau) / cfg.T


def sample_box(n: int, cfg: Config, device, dtype) -> torch.Tensor:
    """Uniform points in [0,S_max] x [0,I_max] x [0,T] x [0,r_max] x [0,sigma_max]."""
    u = torch.rand(n, 5, device=device, dtype=dtype)
    upper = torch.tensor([cfg.S_max, cfg.I_max, cfg.T, cfg.r_max, cfg.sigma_max],
                         device=device, dtype=dtype)
    return u * upper


def sample_data(cfg: Config, device, dtype):
    """Terminal condition and the three boundary faces, n_bc points each."""
    xs, ys = [], []
    n = cfg.n_bc

    # (i) terminal condition t = T
    x = sample_box(n, cfg, device, dtype); x[:, 2] = cfg.T
    xs.append(x); ys.append(payoff(x[:, 1:2], cfg))

    # (ii) S = 0
    x = sample_box(n, cfg, device, dtype); x[:, 0] = 0.0
    xs.append(x); ys.append(bc_value_S0(x[:, 1:2], x[:, 2:3], x[:, 3:4], cfg))

    # (iii) S = S_max
    x = sample_box(n, cfg, device, dtype); x[:, 0] = cfg.S_max
    xs.append(x); ys.append(bc_value_Smax(x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], cfg))

    # (iv) I = I_max
    x = sample_box(n, cfg, device, dtype); x[:, 1] = cfg.I_max
    xs.append(x); ys.append(bc_value_Imax(x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], cfg))

    return torch.cat(xs), torch.cat(ys)


def pde_residual(model: PINN, x: torch.Tensor) -> torch.Tensor:
    """V_t + 1/2 sigma^2 S^2 V_SS + r S V_S + S V_I - r V at points x."""
    x = x.clone().requires_grad_(True)
    V = model(x)
    g = torch.autograd.grad(V.sum(), x, create_graph=True)[0]      # all first derivatives
    V_S, V_I, V_t = g[:, 0:1], g[:, 1:2], g[:, 2:3]
    V_SS = torch.autograd.grad(V_S.sum(), x, create_graph=True)[0][:, 0:1]

    S, r, sigma = x[:, 0:1], x[:, 3:4], x[:, 4:5]
    return V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S + S * V_I - r * V


def train(cfg: Config, out_dir: str | Path) -> PINN:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)

    model = PINN(cfg).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.0)

    with open(out / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    log = open(out / "loss_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "loss", "loss_pde", "loss_data", "lr", "seconds"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params}  epochs={cfg.epochs}")
    t0 = time.time()

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

    log.close()
    torch.save(model.state_dict(), out / "model.pt")
    print(f"saved to {out}/  (training time {time.time() - t0:.0f}s)")
    return model


def load_model(run_dir: str | Path, device: str | None = None) -> tuple[PINN, Config]:
    run = Path(run_dir)
    with open(run / "config.json") as f:
        saved = json.load(f)
    saved.pop("K", None)
    cfg = Config(**saved)
    if device is not None:
        cfg.device = device
    elif not torch.cuda.is_available():
        cfg.device = "cpu"
    model = PINN(cfg).to(device=cfg.device, dtype=getattr(torch, cfg.dtype))
    model.load_state_dict(torch.load(run / "model.pt", map_location=cfg.device))
    model.eval()
    return model, cfg


def _inputs(S, K, r, sigma, cfg, t=0.0, I=0.0):
    dtype = getattr(torch, cfg.dtype)
    S = torch.as_tensor(S, dtype=dtype, device=cfg.device).reshape(-1, 1)
    n = S.shape[0]
    full = lambda v: torch.full((n, 1), float(v), dtype=dtype, device=cfg.device)
    x = torch.cat([S / K, full(I / K), full(t), full(r), full(sigma)], dim=1)

    # warn if asked to price outside the region the network was trained on
    upper = torch.tensor([cfg.S_max, cfg.I_max, cfg.T, cfg.r_max, cfg.sigma_max],
                         dtype=dtype, device=cfg.device)
    outside = ((x < 0) | (x > upper)).any(dim=0)
    if outside.any():
        names = [nm for nm, o in zip(["S/K", "I/K", "t", "r", "sigma"], outside.tolist()) if o]
        warnings.warn(f"inputs outside the training box: {names}; the result is an extrapolation")
    return x


def price(model, cfg, S, K, r, sigma, t=0.0, I=0.0):
    """Price of the fixed-strike Asian call.

    S      spot price (scalar or array)
    K      strike (any value; rescaling is done here)
    r, sigma  rate and volatility (must lie in the training box)
    t      current time, 0 <= t <= T (default: inception)
    I      running integral of S from 0 to t, i.e. t * (average so far); 0 at inception
    """
    x = _inputs(S, K, r, sigma, cfg, t, I)
    with torch.no_grad():
        return (K * model(x)).squeeze(1).cpu().numpy()


def price_and_delta(model, cfg, S, K, r, sigma, t=0.0, I=0.0):
    """Price and delta dV/dS via automatic differentiation (arguments as in price)."""
    x = _inputs(S, K, r, sigma, cfg, t, I).requires_grad_(True)
    V = model(x)
    dV = torch.autograd.grad(V.sum(), x)[0][:, 0]
    return (K * V).squeeze(1).detach().cpu().numpy(), dV.detach().cpu().numpy()


def greeks(model, cfg, S, K, r, sigma, t=0.0, I=0.0):
    """Price and Greeks by automatic differentiation of the network (arguments as in price).

    The network u takes x = (S/K, I/K, t, r, sigma) and V = K * u(x). The chain rule
    then gives (u_S = du/d(S/K), u_I = du/d(I/K), ...):

        delta      dV/dS        = u_S
        gamma      d2V/dS2      = u_SS / K
        dV_dt      dV/dt        = K u_t        (partial derivative, I held fixed: the V_t
                                                term of the PDE; per year)
        theta      dV/dt + S dV/dI             (value change per year as time passes with S
                                                unchanged; I then grows at rate S, since
                                                dI = S dt; /365 for per day)
        vega       dV/dsigma    = K u_sigma    (per 1.00 of volatility; /100 for per 1%)
        rho        dV/dr        = K u_r        (per 1.00 of rate; /100 for per 1%)
        delta_I    dV/dI        = u_I          (sensitivity to the running integral I)
        dual_delta dV/dK        = u - (S/K) u_S - (I/K) u_I   (from V being homogeneous of
                                                               degree 1 in S, I, K)

    Vega and rho exist only because r and sigma are network inputs. Sensitivities to
    parameters that are not inputs (maturity T, a dividend yield) are not available.

    Returns a dict of numpy arrays, one value per entry of S.
    """
    x = _inputs(S, K, r, sigma, cfg, t, I).requires_grad_(True)
    u = model(x)
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]      # all first derivatives
    u_SS = torch.autograd.grad(g[:, 0].sum(), x)[0][:, 0]

    u = u.squeeze(1)
    u_S, u_I, u_t, u_r, u_sigma = (g[:, i] for i in range(5))
    S_over_K, I_over_K = x[:, 0], x[:, 1]

    out = {
        "price": K * u,
        "delta": u_S,
        "gamma": u_SS / K,
        "dV_dt": K * u_t,
        "theta": K * u_t + K * S_over_K * u_I,      # = dV/dt + S dV/dI
        "vega": K * u_sigma,
        "rho": K * u_r,
        "delta_I": u_I,
        "dual_delta": u - S_over_K * u_S - I_over_K * u_I,
    }
    return {name: value.detach().cpu().numpy() for name, value in out.items()}