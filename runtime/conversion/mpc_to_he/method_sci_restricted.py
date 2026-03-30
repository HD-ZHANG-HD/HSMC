from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.layout.bert_edge_packing import (
    build_bert_edge_packing_contract,
    prepare_mpc_tensor_for_he_bert_edge,
)
from ...capabilities import CapabilityStatus
from ...types import BackendType, ExecutionContext, TensorValue
from ..adapters import ConversionContract, MpcToHeAdapter
from ..types import ConversionMethodSpec


METHOD_NAME = "method_sci_restricted"


def _mask(ell: int) -> int:
    return (1 << ell) - 1


def _quantize_to_ring(x: np.ndarray, ell: int, scale: int) -> np.ndarray:
    q = np.round(np.asarray(x, dtype=np.float64).reshape(-1) * float(1 << scale)).astype(np.int64)
    return (q & _mask(ell)).astype(np.uint64)


def _decode_from_ring(x: np.ndarray, ell: int, scale: int, shape: tuple[int, ...]) -> np.ndarray:
    signed = x.astype(np.int64)
    sign_cut = 1 << (ell - 1)
    signed = np.where(signed >= sign_cut, signed - (1 << ell), signed)
    return (signed.astype(np.float64) / float(1 << scale)).reshape(shape)


@dataclass
class SciRestrictedMpcToHeAdapter(MpcToHeAdapter):
    def convert(self, tensor: TensorValue, meta: ConversionContract, ctx: ExecutionContext) -> TensorValue:
        contract = build_bert_edge_packing_contract(meta.tensor_shape, max_tokens=4096)
        packed = prepare_mpc_tensor_for_he_bert_edge(np.asarray(tensor.data, dtype=np.float64), contract)
        q = _quantize_to_ring(packed, meta.ring_bits, meta.scale_bits)
        rng = np.random.default_rng(int(ctx.params.get("conversion_sci_seed", 0)) + 1)
        server_plain_share = rng.integers(0, 1 << meta.ring_bits, size=q.size, dtype=np.uint64)
        client_encrypted_share = (q - server_plain_share) & np.uint64(_mask(meta.ring_bits))
        reconstructed = _decode_from_ring(
            (server_plain_share + client_encrypted_share) & np.uint64(_mask(meta.ring_bits)),
            meta.ring_bits,
            meta.scale_bits,
            contract.tensor_shape,
        )
        ctx.trace.append(
            "[conversion_mpc_to_he_sci_restricted] "
            f"shape={list(contract.tensor_shape)} layout={contract.layout_name} family={contract.layout_family} "
            f"ring_bits={meta.ring_bits} scale_bits={meta.scale_bits}"
        )
        payload = dict(tensor.meta)
        payload.update(contract.as_meta())
        payload["conversion_contract"] = meta.as_meta()
        payload["conversion_protocol"] = {
            "kind": "sci_restricted",
            "reference": "BLB SCI bert_bolt mpc_to_ckks",
            "direction": "MPC_to_HE",
            "server_plain_checksum": int(np.sum(server_plain_share, dtype=np.uint64)),
            "client_cipher_checksum": int(np.sum(client_encrypted_share, dtype=np.uint64)),
        }
        payload["sci_ciphertext_simulation"] = {
            "server_plain_share": server_plain_share.reshape(contract.tensor_shape),
            "client_encrypted_share": client_encrypted_share.reshape(contract.tensor_shape),
        }
        return TensorValue(reconstructed, BackendType.HE, payload)


def _build_contract(tensor: TensorValue, ctx: ExecutionContext) -> ConversionContract:
    shape = tuple(np.asarray(tensor.data).shape)
    contract = build_bert_edge_packing_contract(shape, max_tokens=int(ctx.params.get("conversion_max_tokens", 4096)))
    return ConversionContract(
        direction="MPC_to_HE",
        tensor_shape=shape,
        layout_family=contract.layout_family,
        layout_name=contract.layout_name,
        ring_bits=int(ctx.params.get("conversion_mpc_to_he_ring_bits", ctx.params.get("conversion_ring_bits", 40))),
        scale_bits=int(ctx.params.get("conversion_mpc_to_he_scale_bits", ctx.params.get("conversion_scale_bits", 20))),
        assumptions=(
            "Single-process simulation of BLB/SCI mpc_to_ckks semantics.",
            "Explicit BERT-edge packing is applied before simulated HE re-encoding.",
            "Supported layout families: bert_hidden_state, bert_ffn_intermediate, bert_attention_scores, bert_packed_qkv.",
        ),
        unsupported_cases=(
            "Arbitrary non-BERT tensor layouts are unsupported.",
            "Operator-specific cross-packing beyond shape-preserving BERT edge layouts is unsupported.",
        ),
    )


_ADAPTER = SciRestrictedMpcToHeAdapter()


def convert_mpc_to_he_sci_restricted(tensor: TensorValue, ctx: ExecutionContext) -> TensorValue:
    contract = _build_contract(tensor, ctx)
    return _ADAPTER.convert(tensor, contract, ctx)


METHOD_SPEC = ConversionMethodSpec(
    src_domain=BackendType.MPC,
    dst_domain=BackendType.HE,
    method_name=METHOD_NAME,
    status=CapabilityStatus.RESTRICTED_INTEGRATED,
    fn=convert_mpc_to_he_sci_restricted,
)
