# Min-Cut Domain Assignment Prototype

This folder contains a self-contained compiler prototype for binary operator placement over a transformer-style operator graph.

Domains are fixed to:

- `HE`
- `MPC`

The optimizer minimizes:

- node execution latency
- plus conversion latency when graph edges cross domains

using an s-t min-cut formulation.

## Files

- `profiler_db.py`
  - loads microbenchmark JSON data
  - supports operator and conversion lookups
- `cost_model.py`
  - estimates node and conversion latency from profiler records
  - supports explicit strategies (`exact`, `nearest`, `linear`, `size_scaling`, `auto`)
- `domain_assignment.py`
  - defines graph data structures
  - constructs min-cut graph
  - solves binary HE/MPC assignment
- `plan_builder.py`
  - builds execution plan from graph + assignment
  - inserts explicit conversion operations
- `demo.py`
  - end-to-end runner over synthetic test cases
- `test/*.json`
  - synthetic microbenchmark and graph datasets

## Run Demo

From `he_compiler/operator_execution_framework`:

```bash
python -m compiler.min_cut.demo
```

or:

```bash
cd compiler/min_cut
python demo.py
```

## Input/Output Summary

### Input

- microbenchmark JSON with operator latencies by `(op_type, domain, shape)`
- operator graph JSON with nodes and edges

### Output

- per-node domain assignment (`HE` or `MPC`)
- explicit execution plan steps:
  - operator execution
  - conversion insertion when needed
- estimated total latency with cost breakdown
- baseline comparison:
  - all-HE
  - all-MPC

