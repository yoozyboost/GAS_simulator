# GAS Simulator (Python)

A Python simulator for running Grover Adaptive Search (GAS) efficiently across diverse scenarios.
It supports both circuit-based simulation (Qiskit) and fast circuit-free backends for systematic benchmarking.

## Overview

This repository provides a scenario-driven GAS simulator.
You write a simulation scenario in YAML and run the entry script to execute one or more experiments in a batch.

## Features

- Scenario-driven execution via YAML configuration files
- Supported algorithm families
  - QD-GAS (quantum dictionary based)
  - QSP-GAS (quantum signal processing based)
- Execution modes
  - Circuit mode: build and simulate quantum circuits (Qiskit)
  - No-circuit mode: circuit-free sampling engines for faster runs
- Problem types
  - `binary`: variables in {0, 1}
  - `spin`: variables in {+1, -1}, mapped from bits by `0 -> +1`, `1 -> -1`
- Initial states
  - `uniform` (alias: `hadamard`)
  - `dicke` with parameter `k`
  - `w_state` (alias: `w`)
- Batch execution with result saving and convergence plots

## Requirements

- Python 3.10 or newer

### Python packages

Dependencies are listed in `requirements.txt`.
Typical packages include:

- qiskit
- qiskit-aer
- numpy
- scipy
- sympy
- matplotlib
- pyyaml
- tqdm
- pytest
- pyqsp==0.2.0

## Installation

From the repository root (the directory that contains `GAS_Simulator/`):

```bash
cd GAS_Simulator
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

## Usage

### 1. Write a simulation scenario in YAML

Create a YAML file under `GAS_Simulator/config/`, or start from an existing example.

Example: QSP-GAS, spin variables, no-circuit backend.

```yaml
experiment_name: qsp_nocircuit_spin_example

problem:
  n_key: 4
  objective_function: -x0 * x1 + 2 * x1 + 1.5 * x1 * x2 * x3 + x3 - 1
  variable_type: spin

algorithm:
  method: qsp
  initial_state: uniform
  qsp_params:
    qsp_degree: 9

simulation:
  convergence_backend: nocircuit
  backend: statevector
  max_iterations: 20
  seed: 1

execution:
  num_trials: 100

circuit_assets:
  build: false
```

### 2. Run `main.py`

Edit `GAS_Simulator/main.py` and set `config_list` to the YAML files you want to run, then execute:

```bash
python main.py
```

A batch run directory is created under:

- `GAS_Simulator/results/BatchRun_YYYYMMDD_HHMMSS/`

### 3. Compare multiple scenarios in one batch

Run multiple YAML configs in a single batch to compare convergence behavior and query efficiency under different settings.
Typical comparisons include:

- QD-GAS vs. QSP-GAS
- circuit vs. nocircuit backends
- different objective precisions (for QD, via `n_val` or no-circuit Fejér modeling)
- different initial states (uniform, Dicke, W)
- spin vs. binary variable types

Batch runs create aggregated plots and summaries in the `BatchRun_*` directory, allowing side-by-side evaluation across scenarios.

## Configuration Guide

### Objective function

- Use a SymPy-compatible expression over variables `x0, x1, ..., x{n_key-1}`
- Operators: `+`, `-`, `*`, `**`, parentheses

Examples:

- `-x0 * x1 + 2 * x1 + 1.5 * x1 * x2 * x3 + x3 - 1`

### Variable types

- `binary`: each `xi` is substituted by `0` or `1`
- `spin`: each measured bit is mapped as `0 -> +1`, `1 -> -1`

### Initial states

Set `algorithm.initial_state`:

- `uniform` (or `hadamard`)
- `dicke` with `algorithm.state_prep_params.k`
- `w_state` (or `w`)

Example: Dicke state.

```yaml
algorithm:
  method: qd
  initial_state: dicke
  state_prep_params:
    k: 2
```

### Circuit vs. no-circuit execution

Set `simulation.convergence_backend`:

- `circuit`
  - Constructs a quantum circuit and simulates it to obtain samples
- `nocircuit`
  - Uses a circuit-free engine to sample efficiently

For circuit mode, set `simulation.backend`:

- `statevector`: statevector-based simulation
- `qasm`: shot-based simulation (Aer)

### QD-GAS options

Common QD-specific knobs:

- `problem.n_val`: number of value-register qubits used to encode objective values (precision control)
- `simulation.qd_nocircuit_mode` (when `convergence_backend: nocircuit`)
  - `fejer`: Fejér-kernel based finite-precision modeling of value-register discretization

### QSP-GAS options

Common QSP-specific knobs:

- `algorithm.qsp_params.qsp_degree`: degree parameter for the QSP sequence

## Supported Algorithms and Scenarios

- Algorithms: QD-GAS, QSP-GAS
- Execution: circuit mode, no-circuit mode
- Problem types: spin, binary
- Initial states: uniform, Dicke, W state

## Circuit diagrams and cost metrics

In circuit mode, the simulator can export circuit diagrams and report circuit cost metrics.
This is useful for comparing not only convergence performance but also resource requirements.

Typical outputs include:

- circuit diagrams for the main components (state preparation, oracle, Grover iterate)
- gate counts (including multi-qubit gates)
- depth-related metrics
- estimates of resource usage (for example, ancilla usage if applicable)

These artifacts are saved alongside the experiment outputs when circuit asset export is enabled in the configuration.

## Outputs

Each experiment typically produces an output directory containing:

- `settings.yaml`
- convergence statistics (CSV) and plots
- additional plots (for example, query CDF) when enabled
- circuit diagrams and circuit cost summaries when circuit asset export is enabled

Batch runs typically produce aggregated comparison plots under:

- `results/BatchRun_YYYYMMDD_HHMMSS/`

Console output typically includes, per experiment:

- backend mode (`circuit` or `nocircuit`)
- final best objective (mean over trials)
- success probability and average queries to success (when observed)


## Repository Layout (core)

- `main.py`: entry point for batch execution
- `config/`: YAML scenarios
- `src/`: simulator implementation
  - `core/`: GAS model and circuit construction
  - `engines/`: circuit-free engines
  - `state_prep/`: initial state preparation
  - `oracles/`: oracle builders (QD and QSP)
- `results/`: outputs and plots

## License

This project is released under the MIT License. See `LICENSE`.

It includes modified portions of software originally developed by Ishikawa Laboratory (MIT License).
See `THIRD_PARTY_NOTICES.md` for details.
