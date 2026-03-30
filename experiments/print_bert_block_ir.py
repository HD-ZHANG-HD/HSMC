from __future__ import annotations

from ir import BertBlockConfig, build_bert_block_graph


def main() -> None:
    graph = build_bert_block_graph(
        BertBlockConfig(
            batch_size=1,
            seq_len=8,
            hidden_size=768,
            intermediate_size=3072,
            num_heads=12,
            graph_id="bert_block_demo",
        )
    )

    print(f"graph_id={graph.graph_id}")
    print("[nodes]")
    for node in graph.nodes:
        print(
            f"  {node.node_id}: {node.op_type} "
            f"in={list(node.input_shape)} out={list(node.output_shape)} attrs={node.attributes}"
        )

    print("[edges]")
    for edge in graph.edges:
        print(f"  {edge.src} -> {edge.dst} shape={list(edge.tensor_shape)}")


if __name__ == "__main__":
    main()
