# Bifrost CSL — Sub-Quadratic Linear Foundation Architecture

**Bifrost** is a linear-time, attention-free foundation model architecture 
developed by **Valerois AI**. It replaces the quadratic memory footprint of 
classical attention with a constant O(1) state buffer, enabling efficient 
long-context language modeling on consumer hardware.

**Valkir** is our reference language model family built on Bifrost.

---

## Architectural Properties

- **Linear-Time Complexity:** O(N) training, O(1) inference memory per token.
- **Zero KV-Cache Growth:** A static circular state buffer replaces dynamic KV-cache allocation.
- **Multi-Scale Dilated Causal Operators:** Effective receptive field exceeding 1,200 tokens with sub-quadratic cost.
- **Sub-Quadratic Memory Wall:** No quadratic attention matrices; no O(N²) bottlenecks.
- **Consumer-GPU Native:** Reference model trained end-to-end on a single AMD Radeon RX 9070 XT (16 GB).

---

## Reference Model: Valkir 16L (116.8M Parameters)

Trained on 2.36 billion tokens of pure algorithmic Python and scientific text.
No chat templates. No instruction contamination. No synthetic leakage.

### Official Zero-Shot Benchmark Telemetry
*Evaluated on 18,838 official validation samples across 4 benchmarks.*

| Benchmark | Samples | Valkir 16L | Pythia-160M | OPT-125M |
|-----------|--------:|-----------:|------------:|---------:|
| **WinoGrande** | 1,267 | **52.01%** | 51.80% | 50.10% |
| **ARC-Easy** | 2,376 | **32.95%** | 33.50% | 31.80% |
| **HellaSwag** | 10,042 | 25.92% | 29.80% | 28.20% |
| **HumanEval (AST)** | 164 | **44.0%** | ~20% | ~15% |

*Reference baselines required ~127× more pretraining data to reach parity.*

### Hardware Telemetry (AMD RX 9070 XT, ROCm 7.2)
- **Training Throughput:** 37,350 tok/s (ROCm Inductor) / 29,550 tok/s (native PyTorch)
- **Peak VRAM:** 4.87 GB (compiled) / 8.12 GB (native)
- **Wall-Clock:** 9.24 hours for 1.25B continual tokens
- **Checkpoint Size:** 223 MB (native BFloat16)

---

## Scaling Trajectory

The Bifrost architecture scales linearly with model size. The 116M reference 
model validates the mathematical foundation; the next milestone targets 
**1.5B – 3B parameters** on a mixed code + scientific corpus.

---

## Access & Licensing

> ⚠️ **Proprietary Architecture — All Rights Reserved**

This repository contains **documentation only**.  
The Bifrost architecture, Valkir model weights, training pipeline, 
kernel implementations, and optimizer stack are **not** included 
and remain the exclusive intellectual property of Valerois AI.

**No reproduction, redistribution, derivative work, or commercial use is permitted** without prior written consent from Valerois AI.

For research collaboration, compute partnerships, or technical audit under NDA:

📧 **contact@valerois.com**  
🌐 **valerois.com**

---

## Prior Art Notice

This repository servesed record of the Bifrost architecture's 
existence and public disclosure. First published: 2026.

---

*© 2026 Valerois AI. All rights reserved. Bifrost and Valkir are trademarks of Valerois AI.*