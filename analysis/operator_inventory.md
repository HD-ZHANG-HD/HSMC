# Step 1 - Operator Inventory from Existing Implementations

## MPCFormer (`he_compiler/MPCFormer/src/benchmark/models.py`)

- `Embedding`
  - `BertEmbeddings.forward`: token projection + positional embedding + normalization.
- `Linear_QKV`
  - `BertSelfAttention.forward`: `query`, `key`, `value` linear projections.
- `Attention_QK_MatMul`
  - `BertSelfAttention.forward`: `query_layer.matmul(key_layer.transpose(-1, -2))`.
- `Softmax`
  - `BertSelfAttention.forward`: `self.smax(attention_scores)`.
- `Attention_V_MatMul`
  - `BertSelfAttention.forward`: `attention_probs.matmul(value_layer)`.
- `Out_Projection`
  - `BertSelfOutput.forward`: `self.dense(hidden_states)`.
- `Residual_Add`
  - `BertSelfOutput.forward` and `BertOutput.forward`: residual additions.
- `LayerNorm`
  - `BertEmbeddings.forward`, `BertSelfOutput.forward`, `BertOutput.forward`.
- `FFN_Linear_1`
  - `BertIntermediate.forward`: first FFN dense.
- `GeLU`
  - `BertIntermediate.forward`: `activation_quad` / activation function.
- `FFN_Linear_2`
  - `BertOutput.forward`: second FFN dense.

## EzPC BOLT HE-MPC (`he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt`)

- `Linear_QKV`
  - `bert.cpp` calls `lin.linear_1(...)` (HE path, QKV packed).
- `Attention_QK_MatMul`
  - `linear.cpp` `bert_cipher_cipher_cross_packing(...)`.
- `Softmax`
  - `bert.cpp` calls `nl.softmax(...)` (MPC nonlinear).
- `Attention_V_MatMul`
  - `bert.cpp` mixed flow around `lin.softmax_v(...)` / `lin.bert_softmax_V(...)`.
- `Out_Projection`
  - `bert.cpp` "Linear #2", `lin.linear_2(...)`.
- `Residual_Add`
  - `bert.cpp` residual add before first layer norm (`ln_input_row[i] += h1_cache_12[i]`).
- `LayerNorm`
  - `bert.cpp` calls `nl.layer_norm(...)` twice per layer.
- `FFN_Linear_1`
  - `bert.cpp` "Linear #3", `lin.linear_2(...)` with FFN-1 params.
- `GeLU`
  - `bert.cpp` calls `nl.gelu(...)`.
- `FFN_Linear_2`
  - `bert.cpp` "Linear #4", `lin.linear_2(...)`.
- Conversions already present in legacy flow:
  - `he_to_ss_server/client`: HE -> MPC-share domain.
  - `ss_to_he_server/client`: MPC-share -> HE domain.

## Mapping Note

`Embedding` is explicit in MPCFormer and can be added to EzPC BOLT by introducing a pre-attention embedding operator module ahead of `linear_1` input preparation.
