
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple


Domain = Literal["HE", "MPC"]


def _as_shape(value: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in value)


def _as_domain(value: str) -> Domain:
    domain = str(value).upper()
    if domain not in {"HE", "MPC"}:
        raise ValueError(f"Unsupported domain in profiler record: {domain}")
    return domain  # type: ignore[return-value]


@dataclass(frozen=True)
class BenchmarkRecord:
    op_type: str
    domain: Domain
    method: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    local_compute_ms: float
    comm_bytes: int
    comm_rounds: int
    total_latency_ms: float
    metadata: Dict[str, Any]

    @property
    def latency_ms(self) -> float:
        return self.total_latency_ms


@dataclass(frozen=True)
class ConversionRecord:
    from_domain: Domain
    to_domain: Domain
    method: str
    layout_family: str
    tensor_shape: Tuple[int, ...]
    local_compute_ms: float
    comm_bytes: int
    comm_rounds: int
    total_latency_ms: float
    metadata: Dict[str, Any]

    @property
    def direction(self) -> str:
        return f"{self.from_domain}_to_{self.to_domain}"

    @property
    def latency_ms(self) -> float:
        return self.total_latency_ms


class ProfilerDB:
    """In-memory microbenchmark database for compiler cost estimation."""

    SUPPORTED_DOMAINS = {"HE", "MPC"}

    def __init__(self, records: List[BenchmarkRecord], conversion_records: List[ConversionRecord]) -> None:
        self.records = records
        self.conversion_records = conversion_records
        self._op_index: Dict[Tuple[str, Domain, str], List[BenchmarkRecord]] = {}
        self._conv_index: Dict[Tuple[Domain, Domain, str, str], List[ConversionRecord]] = {}
        for rec in records:
            self._op_index.setdefault((rec.op_type, rec.domain, rec.method), []).append(rec)
            self._op_index.setdefault((rec.op_type, rec.domain, "*"), []).append(rec)
        for rec in conversion_records:
            self._conv_index.setdefault((rec.from_domain, rec.to_domain, rec.method, rec.layout_family), []).append(rec)
            self._conv_index.setdefault((rec.from_domain, rec.to_domain, "*", rec.layout_family), []).append(rec)
            self._conv_index.setdefault((rec.from_domain, rec.to_domain, rec.method, "*"), []).append(rec)
            self._conv_index.setdefault((rec.from_domain, rec.to_domain, "*", "*"), []).append(rec)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "ProfilerDB":
        payload = json.loads(Path(json_path).read_text())
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProfilerDB":
        records: List[BenchmarkRecord] = []
        conversion_records: List[ConversionRecord] = []
        for rec in payload.get("records", []):
            domain = _as_domain(rec.get("backend", rec.get("domain", "")))
            local_compute_ms = float(rec.get("local_compute_ms", rec.get("latency_ms", 0.0)))
            total_latency_ms = float(rec.get("total_latency_ms", rec.get("latency_ms", local_compute_ms)))
            records.append(
                BenchmarkRecord(
                    op_type=str(rec["op_type"]),
                    domain=domain,
                    method=str(rec.get("method", "method_default")),
                    input_shape=_as_shape(rec["input_shape"]),
                    output_shape=_as_shape(rec["output_shape"]),
                    local_compute_ms=local_compute_ms,
                    comm_bytes=int(rec.get("comm_bytes", 0)),
                    comm_rounds=int(rec.get("comm_rounds", 0)),
                    total_latency_ms=total_latency_ms,
                    metadata=dict(rec.get("metadata", {})),
                )
            )
        for rec in payload.get("conversion_records", []):
            from_domain = _as_domain(rec.get("from_domain", str(rec.get("direction", "HE_to_MPC")).split("_to_")[0]))
            to_domain = _as_domain(rec.get("to_domain", str(rec.get("direction", "HE_to_MPC")).split("_to_")[1]))
            local_compute_ms = float(rec.get("local_compute_ms", rec.get("latency_ms", 0.0)))
            total_latency_ms = float(rec.get("total_latency_ms", rec.get("latency_ms", local_compute_ms)))
            conversion_records.append(
                ConversionRecord(
                    from_domain=from_domain,
                    to_domain=to_domain,
                    method=str(rec.get("method", "method_default")),
                    layout_family=str(rec.get("layout_family", "generic")),
                    tensor_shape=_as_shape(rec["tensor_shape"]),
                    local_compute_ms=local_compute_ms,
                    comm_bytes=int(rec.get("comm_bytes", 0)),
                    comm_rounds=int(rec.get("comm_rounds", 0)),
                    total_latency_ms=total_latency_ms,
                    metadata=dict(rec.get("metadata", {})),
                )
            )
        return cls(records=records, conversion_records=conversion_records)

    def get_operator_records(self, op_type: str, domain: Domain, method: str | None = None) -> List[BenchmarkRecord]:
        return list(self._op_index.get((op_type, domain, method or "*"), []))

    def find_exact_operator_record(
        self,
        op_type: str,
        domain: Domain,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        method: str | None = None,
    ) -> BenchmarkRecord | None:
        for rec in self.get_operator_records(op_type, domain, method=method):
            if rec.input_shape == input_shape and rec.output_shape == output_shape:
                return rec
        return None

    def get_conversion_records(
        self,
        from_domain: Domain,
        to_domain: Domain,
        method: str | None = None,
        layout_family: str | None = None,
    ) -> List[ConversionRecord]:
        return list(self._conv_index.get((from_domain, to_domain, method or "*", layout_family or "*"), []))

    def find_exact_conversion_record(
        self,
        from_domain: Domain,
        to_domain: Domain,
        tensor_shape: Tuple[int, ...],
        method: str | None = None,
        layout_family: str | None = None,
    ) -> ConversionRecord | None:
        for rec in self.get_conversion_records(from_domain, to_domain, method=method, layout_family=layout_family):
            if rec.tensor_shape == tensor_shape:
                return rec
        return None
