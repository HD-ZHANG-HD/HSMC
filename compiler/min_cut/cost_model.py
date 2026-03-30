
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Tuple

from .profiler_db import BenchmarkRecord, ConversionRecord, Domain, ProfilerDB


EstimationStrategy = Literal["exact", "nearest", "linear", "size_scaling", "auto"]


def _numel(shape: Iterable[int]) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return int(n)


def _shape_distance(a_in: Tuple[int, ...], a_out: Tuple[int, ...], b_in: Tuple[int, ...], b_out: Tuple[int, ...]) -> float:
    a = _numel(a_in) + _numel(a_out)
    b = _numel(b_in) + _numel(b_out)
    if a <= 0 or b <= 0:
        return float(abs(a - b))
    return abs(math.log(float(a) / float(b)))


@dataclass(frozen=True)
class CostEstimate:
    latency_ms: float
    strategy_used: str


class CostModel:
    """Explicit and testable cost model for node and conversion latency."""

    def __init__(self, db: ProfilerDB, default_strategy: EstimationStrategy = "auto") -> None:
        self.db = db
        self.default_strategy = default_strategy

    def estimate_node_cost(
        self,
        op_type: str,
        domain: Domain,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        strategy: EstimationStrategy | None = None,
        method: str | None = None,
    ) -> CostEstimate:
        mode = strategy or self.default_strategy
        exact = self.db.find_exact_operator_record(op_type, domain, input_shape, output_shape, method=method)
        if exact is not None:
            return CostEstimate(latency_ms=exact.total_latency_ms, strategy_used="exact")

        candidates = self.db.get_operator_records(op_type, domain, method=method)
        if not candidates:
            raise ValueError(f"No profiler records for op={op_type}, domain={domain}, method={method or '*'}")

        if mode == "exact":
            raise ValueError(
                f"Exact record not found for op={op_type}, domain={domain}, method={method or '*'}, "
                f"shapes={input_shape}->{output_shape}"
            )
        if mode == "nearest":
            return self._nearest(candidates, input_shape, output_shape)
        if mode == "linear":
            return self._linear_fit(candidates, input_shape, output_shape)
        if mode == "size_scaling":
            return self._size_scaling(candidates, input_shape, output_shape)
        if mode == "auto":
            if len(candidates) >= 2:
                return self._linear_fit(candidates, input_shape, output_shape)
            return self._nearest(candidates, input_shape, output_shape)
        raise ValueError(f"Unsupported estimation strategy: {mode}")

    def estimate_conversion_cost(
        self,
        tensor_shape: Tuple[int, ...],
        from_domain: Domain,
        to_domain: Domain,
        strategy: EstimationStrategy | None = None,
        method: str | None = None,
        layout_family: str | None = None,
    ) -> CostEstimate:
        if from_domain == to_domain:
            return CostEstimate(latency_ms=0.0, strategy_used="same_domain")
        mode = strategy or self.default_strategy
        exact = self.db.find_exact_conversion_record(
            from_domain,
            to_domain,
            tensor_shape,
            method=method,
            layout_family=layout_family,
        )
        if exact is not None:
            return CostEstimate(latency_ms=exact.total_latency_ms, strategy_used="exact")

        candidates = self.db.get_conversion_records(
            from_domain,
            to_domain,
            method=method,
            layout_family=layout_family,
        )
        if not candidates:
            raise ValueError(
                f"No conversion records for {from_domain}->{to_domain}, method={method or '*'}, layout={layout_family or '*'}"
            )

        if mode == "exact":
            raise ValueError(
                f"Exact conversion record not found for {from_domain}->{to_domain} "
                f"shape={tensor_shape}, method={method or '*'}, layout={layout_family or '*'}"
            )
        if mode in ("nearest", "auto"):
            nearest = min(candidates, key=lambda r: _shape_distance(tensor_shape, tensor_shape, r.tensor_shape, r.tensor_shape))
            return CostEstimate(latency_ms=nearest.total_latency_ms, strategy_used="nearest")
        if mode == "size_scaling":
            nearest = min(candidates, key=lambda r: _shape_distance(tensor_shape, tensor_shape, r.tensor_shape, r.tensor_shape))
            base_size = max(1, _numel(nearest.tensor_shape))
            query_size = max(1, _numel(tensor_shape))
            scaled = nearest.total_latency_ms * (float(query_size) / float(base_size))
            return CostEstimate(latency_ms=max(0.0, scaled), strategy_used="size_scaling")
        if mode == "linear":
            xs = [float(_numel(rec.tensor_shape)) for rec in candidates]
            ys = [float(rec.total_latency_ms) for rec in candidates]
            return CostEstimate(latency_ms=max(0.0, self._fit_predict(xs, ys, float(_numel(tensor_shape)))), strategy_used="linear")
        raise ValueError(f"Unsupported conversion estimation strategy: {mode}")

    def _nearest(self, candidates: list[BenchmarkRecord], input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> CostEstimate:
        nearest = min(
            candidates,
            key=lambda r: _shape_distance(input_shape, output_shape, r.input_shape, r.output_shape),
        )
        return CostEstimate(latency_ms=nearest.total_latency_ms, strategy_used="nearest")

    def _size_scaling(self, candidates: list[BenchmarkRecord], input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> CostEstimate:
        nearest = min(
            candidates,
            key=lambda r: _shape_distance(input_shape, output_shape, r.input_shape, r.output_shape),
        )
        base_size = max(1, _numel(nearest.input_shape) + _numel(nearest.output_shape))
        query_size = max(1, _numel(input_shape) + _numel(output_shape))
        scaled = nearest.total_latency_ms * (float(query_size) / float(base_size))
        return CostEstimate(latency_ms=max(0.0, scaled), strategy_used="size_scaling")

    def _linear_fit(self, candidates: list[BenchmarkRecord], input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> CostEstimate:
        xs = [float(_numel(rec.input_shape) + _numel(rec.output_shape)) for rec in candidates]
        ys = [float(rec.total_latency_ms) for rec in candidates]
        xq = float(_numel(input_shape) + _numel(output_shape))
        predicted = self._fit_predict(xs, ys, xq)
        return CostEstimate(latency_ms=max(0.0, predicted), strategy_used="linear")

    @staticmethod
    def _fit_predict(xs: list[float], ys: list[float], xq: float) -> float:
        if len(xs) != len(ys) or not xs:
            raise ValueError("Invalid data for linear fit")
        if len(xs) == 1:
            return ys[0]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x == 0.0:
            return mean_y
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = cov / var_x
        intercept = mean_y - slope * mean_x
        return slope * xq + intercept
