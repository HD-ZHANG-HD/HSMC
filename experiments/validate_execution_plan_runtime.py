from __future__ import annotations

import numpy as np

from runtime import (
    BackendType,
    ExecutionContext,
    ExecutionPlan,
    OperatorRegistry,
    OperatorStep,
    TensorValue,
    ConversionStep,
    execute,
)


def _make_op(op_name: str, backend: BackendType, scale: float):
    def fn(inputs, ctx: ExecutionContext):
        ctx.trace.append(f"{op_name}@{backend.value}/method_default")
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(x * scale, backend, {"shape": list(x.shape)})

    return fn


def main() -> None:
    registry = OperatorRegistry()
    registry.register("MockHeOp", BackendType.HE, _make_op("MockHeOp", BackendType.HE, 2.0))
    registry.register("MockMpcOp", BackendType.MPC, _make_op("MockMpcOp", BackendType.MPC, 3.0))

    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    tensors = {"input": TensorValue(x, BackendType.HE)}
    plan = ExecutionPlan(
        steps=[
            OperatorStep(
                op_type="MockHeOp",
                method="method_default",
                backend=BackendType.HE,
                inputs=["input"],
                outputs=["he_out"],
            ),
            ConversionStep(
                from_domain=BackendType.HE,
                to_domain=BackendType.MPC,
                tensor="he_out",
                method="method_default",
                output_tensor="mpc_in",
            ),
            OperatorStep(
                op_type="MockMpcOp",
                method="method_default",
                backend=BackendType.MPC,
                inputs=["mpc_in"],
                outputs=["final_out"],
            ),
        ]
    )

    ctx = ExecutionContext()
    outputs = execute(plan, tensors, ctx=ctx, registry=registry)

    final_out = outputs["final_out"]
    assert final_out.domain == BackendType.MPC
    assert np.asarray(final_out.data).shape == x.shape
    assert any(line.startswith("CONVERT HE_to_MPC@method_default") for line in ctx.trace)

    execute_idx = ctx.trace.index("EXECUTE MockHeOp@HE/method_default")
    convert_idx = next(i for i, line in enumerate(ctx.trace) if line.startswith("CONVERT HE_to_MPC@method_default"))
    mpc_idx = ctx.trace.index("EXECUTE MockMpcOp@MPC/method_default")
    assert execute_idx < convert_idx < mpc_idx

    mismatch_plan = ExecutionPlan(
        steps=[
            OperatorStep(
                op_type="MockMpcOp",
                method="method_default",
                backend=BackendType.MPC,
                inputs=["input"],
                outputs=["bad_out"],
            )
        ]
    )
    try:
        execute(mismatch_plan, {"input": TensorValue(x, BackendType.HE)}, registry=registry)
    except ValueError as exc:
        assert "domain mismatch" in str(exc)
    else:
        raise AssertionError("Expected runtime.execute to reject missing explicit conversion")

    print("PASS: runtime executes explicit execution plans without dynamic routing")


if __name__ == "__main__":
    main()
