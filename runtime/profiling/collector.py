
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .schema import ConversionProfileRecord, OperatorProfileRecord


@dataclass
class ProfilingCollector:
    operator_records: list[OperatorProfileRecord] = field(default_factory=list)
    conversion_records: list[ConversionProfileRecord] = field(default_factory=list)

    def add_operator_record(self, record: OperatorProfileRecord) -> None:
        self.operator_records.append(record)

    def add_conversion_record(self, record: ConversionProfileRecord) -> None:
        self.conversion_records.append(record)

    def export_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "domains": ["HE", "MPC"],
            "records": [record.to_dict() for record in self.operator_records],
            "conversion_records": [record.to_dict() for record in self.conversion_records],
        }

    def export_json(self) -> str:
        return json.dumps(self.export_payload(), indent=2)
