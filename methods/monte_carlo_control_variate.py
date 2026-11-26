from functools import cached_property

import numpy as np
from scipy.stats import norm


class AsianOptionMC:
    """Arithmetic Asian option priced by Monte Carlo.

    option_type : 'call' or 'put'
    S0          : initial stock/index level (> 0)
    strike      : strike price (> 0)
    T           : time to maturity in years (> 0)
    M           : number of monitoring dates (time steps)
    r           : constant risk-free short rate
    div         : continuous dividend yield
    sigma       : volatility
    simulations : number of simulated paths
    seed        : seed of the random generator (same seed -> same paths)
    batch_size  : paths simulated at once; default keeps each batch near
                  10 million numbers (~80 MB) so large M does not run out of memory
    """

    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, simulations,
                 seed=100, batch_size=None, average='discrete'):
        if option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        if average not in ('discrete', 'continuous'):
            raise ValueError("average must be 'discrete' or 'continuous'")
        self.average = average
        self.option_type = option_type
        self.S0 = float(S0)
        self.strike = float(strike)
        self.T = float(T)
        self.M = int(M)
        self.r = float(r)
        self.div = float(div)
        self.sigma = float(sigma)
        self.simulations = int(simulations)
        self.seed = seed

        if self.S0 <= 0 or self.strike <= 0 or self.T <= 0:
            raise ValueError('S0, strike and T must be positive')
        if self.sigma < 0 or self.div < 0:
            raise ValueError('sigma and div must be non-negative')
        if self.M < 1 or self.simulations < 2:
            raise ValueError('need M >= 1 and at least 2 simulations')

        self.time_unit = self.T / self.M
        self.discount = np.exp(-self.r * self.T)
        self.batch_size = int(batch_size) if batch_size else max(1, 10_000_000 // self.M)

    @property
    def GeometricAsianOption(self):
        """Exact price of the geometric-average Asian option (closed form)."""
        M, T, sigma = self.M, self.T, self.sigma
        var = sigma**2 * T * (M + 1) * (2 * M + 1) / (6 * M**2)       # Var[log G]
        muT = 0.5 * var + (self.r - self.div - 0.5 * sigma**2) * T * (M + 1) / (2 * M)
        # muT = log(E[G] / S0)
        d1 = (np.log(self.S0 / self.strike) + muT + 0.5 * var) / np.sqrt(var)
        d2 = d1 - np.sqrt(var)
        forward_G = self.S0 * np.exp(muT)
        if self.option_type == 'call':
            return self.discount * (forward_G * norm.cdf(d1) - self.strike * norm.cdf(d2))
        return self.discount * (self.strike * norm.cdf(-d2) - forward_G * norm.cdf(-d1))


    def _simulate_batch(self, rng, n):
        """n paths at the M monitoring dates, shape (n, M)."""
        drift = (self.r - self.div - 0.5 * self.sigma**2) * self.time_unit
        vol = self.sigma * np.sqrt(self.time_unit)
        log_paths = np.cumsum(drift + vol * rng.standard_normal((n, self.M)), axis=1)
        return self.S0 * np.exp(log_paths)

    @cached_property
    def price_path(self):
        """All simulated paths, shape (simulations, M). Only for inspection or
        plotting: it stores everything, so avoid it for very large runs."""
        return self._simulate_batch(np.random.default_rng(self.seed), self.simulations)


    def _payoff(self, average):
        if self.option_type == 'call':
            return self.discount * np.maximum(average - self.strike, 0.0)
        return self.discount * np.maximum(self.strike - average, 0.0)


    @cached_property
    def _payoffs(self):
        """Discounted arithmetic and geometric payoffs of every path.
        Simulated once, in batches, and reused by all results below."""
        rng = np.random.default_rng(self.seed)
        arith = np.empty(self.simulations)
        geom = np.empty(self.simulations)
        for start in range(0, self.simulations, self.batch_size):
            n = min(self.batch_size, self.simulations - start)
            paths = self._simulate_batch(rng, n)
            if self.average == 'discrete':
                arithmetic_mean = paths.mean(axis=1)
            else:   # trapezoid rule on [0, T], including S0
                arithmetic_mean = (0.5 * self.S0 + paths[:, :-1].sum(axis=1)
                                   + 0.5 * paths[:, -1]) / self.M
            arith[start:start + n] = self._payoff(arithmetic_mean)
            geom[start:start + n] = self._payoff(np.exp(np.log(paths).mean(axis=1)))
        return arith, geom
    

    @property
    def MCPayoff(self):
        """Discounted arithmetic-average payoffs, one per path (chosen average)."""
        return self._payoffs[0]


    def _estimate(self, samples):
        mean = samples.mean()
        half_width = 1.96 * samples.std(ddof=1) / np.sqrt(self.simulations)
        return mean, mean - half_width, mean + half_width
    
    
    @property
    def value(self):
        """Plain Monte Carlo estimate."""
        return self._estimate(self.MCPayoff)

    @property
    def value_with_control_variate(self):
        """Estimate with the geometric Asian option as control variate (beta = 1)."""
        arith, geom = self._payoffs
        return self._estimate(arith + self.GeometricAsianOption - geom)
