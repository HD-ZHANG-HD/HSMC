from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ExistingCodebasePaths:
    mpcformer_models_py: Path
    ezpc_bolt_bert_cpp: Path
    ezpc_bolt_linear_cpp: Path
    ezpc_bolt_nonlinear_cpp: Path


def discover_existing_paths(project_root: Path) -> ExistingCodebasePaths:
    return ExistingCodebasePaths(
        mpcformer_models_py=project_root / "he_compiler/MPCFormer/src/benchmark/models.py",
        ezpc_bolt_bert_cpp=project_root / "he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/bert.cpp",
        ezpc_bolt_linear_cpp=project_root / "he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/linear.cpp",
        ezpc_bolt_nonlinear_cpp=project_root / "he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp",
    )


def operator_source_map(paths: ExistingCodebasePaths) -> Dict[str, str]:
    return {
        "Embedding": str(paths.mpcformer_models_py),
        "Linear_QKV": str(paths.mpcformer_models_py),
        "Attention_QK_MatMul": str(paths.mpcformer_models_py),
        "Softmax": str(paths.ezpc_bolt_nonlinear_cpp),
        "Attention_V_MatMul": str(paths.ezpc_bolt_linear_cpp),
        "Out_Projection": str(paths.ezpc_bolt_linear_cpp),
        "Residual_Add": str(paths.mpcformer_models_py),
        "LayerNorm": str(paths.ezpc_bolt_nonlinear_cpp),
        "FFN_Linear_1": str(paths.ezpc_bolt_linear_cpp),
        "GeLU": str(paths.ezpc_bolt_nonlinear_cpp),
        "FFN_Linear_2": str(paths.ezpc_bolt_linear_cpp),
    }
