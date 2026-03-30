from __future__ import annotations

import numpy as np

from compiler.min_cut.runtime_plan_adapter import resolve_conversion_method
from ir import BertBlockConfig, build_bert_block_graph
from runtime import BackendType, ExecutionContext, TensorValue, conversion_manager


def _sample_tensor(shape: tuple[int, ...]) -> np.ndarray:
    numel = int(np.prod(shape))
    data = np.linspace(-0.5, 0.5, num=numel, dtype=np.float64)
    return data.reshape(shape)


def main() -> None:
    graph = build_bert_block_graph(BertBlockConfig(batch_size=1, seq_len=8, hidden_size=768, intermediate_size=3072))
    node_map = {node.node_id: node for node in graph.nodes}
    checked = []

    for edge in graph.edges:
        src_op = node_map[edge.src].op_type
        dst_op = node_map[edge.dst].op_type
        method = resolve_conversion_method(src_op, dst_op, edge.tensor_shape)
        if method != "method_sci_restricted":
            raise AssertionError(
                f"Expected general restricted conversion on {edge.src}->{edge.dst} ({src_op}->{dst_op}), got {method}"
            )

        base = _sample_tensor(edge.tensor_shape)

        ctx_he_to_mpc = ExecutionContext(params={"conversion_sci_seed": 7})
        out_mpc = conversion_manager.convert(
            TensorValue(np.array(base, copy=True), BackendType.HE, {"edge": f"{edge.src}->{edge.dst}"}),
            BackendType.MPC,
            ctx_he_to_mpc,
            method_name=method,
        )
        assert out_mpc.domain == BackendType.MPC
        assert tuple(np.asarray(out_mpc.data).shape) == edge.tensor_shape
        assert out_mpc.meta["conversion_protocol"]["kind"] == "sci_restricted"

        ctx_mpc_to_he = ExecutionContext(params={"conversion_sci_seed": 7})
        out_he = conversion_manager.convert(
            TensorValue(np.array(base, copy=True), BackendType.MPC, {"edge": f"{edge.src}->{edge.dst}"}),
            BackendType.HE,
            ctx_mpc_to_he,
            method_name=method,
        )
        assert out_he.domain == BackendType.HE
        assert tuple(np.asarray(out_he.data).shape) == edge.tensor_shape
        assert out_he.meta["conversion_protocol"]["kind"] == "sci_restricted"

        checked.append(
            {
                "edge": f"{edge.src}->{edge.dst}",
                "ops": f"{src_op}->{dst_op}",
                "shape": list(edge.tensor_shape),
                "layout_family": out_mpc.meta["layout_family"],
                "method": method,
            }
        )

    print("PASS: generalized BERT-edge HE<->MPC conversion covers all adjacent operators")
    for item in checked:
        print(
            f"  {item['edge']} | {item['ops']} | shape={item['shape']} | "
            f"layout={item['layout_family']} | method={item['method']}"
        )


if __name__ == "__main__":
    main()
