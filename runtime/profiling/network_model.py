
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    bandwidth_bytes_per_sec: float
    rtt_ms: float

    def describe(self) -> str:
        return (
            f"bandwidth={self.bandwidth_bytes_per_sec:.0f}Bps "
            f"rtt_ms={self.rtt_ms:.3f}"
        )


class NetworkModel:
    @staticmethod
    def estimate_latency(
        local_ms: float,
        comm_bytes: int,
        comm_rounds: int,
        config: NetworkConfig,
    ) -> float:
        transfer_time_ms = (float(comm_bytes) / float(config.bandwidth_bytes_per_sec)) * 1000.0
        rtt_cost_ms = float(comm_rounds) * float(config.rtt_ms)
        return float(local_ms) + transfer_time_ms + rtt_cost_ms
