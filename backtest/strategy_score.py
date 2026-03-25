import math


class StrategyScore:
    """Weighted multi-metric fitness score for strategy ranking (SQX-style).

    Each metric is normalized to [0, 1] and combined using configurable weights.
    Supports two modes:
      - rank()  : population-relative min-max normalization (for optimization sweeps)
      - score() : absolute scoring with predefined caps (for single-strategy evaluation)
    """

    DEFAULT_WEIGHTS = {
        "return_dd_ratio": 0.50,
        "profit_factor": 0.20,
        "num_trades": 0.15,
        "sharpe_ratio": 0.15,
    }

    DEFAULT_CAPS = {
        "return_dd_ratio": 10.0,
        "profit_factor": 5.0,
        "num_trades": 500,
        "sharpe_ratio": 5.0,
    }

    def __init__(self, weights=None, caps=None, profit_factor_cap=10.0):
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self.caps = caps or dict(self.DEFAULT_CAPS)
        self.profit_factor_cap = profit_factor_cap

    def _clamp_pf(self, value):
        """Cap profit_factor to handle inf values."""
        if math.isinf(value):
            return self.profit_factor_cap
        return min(value, self.profit_factor_cap)

    def rank(self, metrics_list):
        """Rank strategies using population-relative min-max normalization.

        Args:
            metrics_list: list of metric dicts (each must contain the keys in self.weights).

        Returns:
            New list of dicts sorted by fitness_score descending, with fitness_score added.
        """
        if not metrics_list:
            return []

        # Work on copies to avoid mutating originals
        items = [dict(m) for m in metrics_list]

        # Clamp profit_factor before normalization
        for m in items:
            m["profit_factor"] = self._clamp_pf(m["profit_factor"])

        # Min-max normalize each metric across the population
        for key in self.weights:
            values = [m[key] for m in items]
            lo, hi = min(values), max(values)
            for m in items:
                m[f"_n_{key}"] = (m[key] - lo) / (hi - lo) if hi > lo else 1.0

        # Weighted sum
        for m in items:
            m["fitness_score"] = round(
                sum(self.weights[k] * m[f"_n_{k}"] for k in self.weights), 4
            )
            for k in self.weights:
                del m[f"_n_{k}"]

        items.sort(key=lambda m: m["fitness_score"], reverse=True)
        return items

    def score(self, metrics):
        """Score a single strategy using predefined caps for normalization.

        Args:
            metrics: dict containing the metric keys in self.weights.

        Returns:
            float fitness score in [0, 1].
        """
        result = 0.0
        for key, weight in self.weights.items():
            value = metrics[key]
            if key == "profit_factor":
                value = self._clamp_pf(value)
            cap = self.caps[key]
            normalized = min(value / cap, 1.0) if cap > 0 else 0.0
            result += weight * normalized
        return round(result, 4)
