---
title: "Fused3S: Fast Sparse Attention on Tensor Cores"
authors: "Zitong Li, Aparna Chandramowlishwaran"
periodical: "ICS'25"
periodical_full: "ICS 2025"
link: "https://doi.org/10.1145/3721145.3730430"
code: "https://github.com/HPCForge/Fused3S"
date: 2025-08-22
---

## Motivation

Attention has become fundamental in machine learning models from transformers to graph neural networks (GNNs). However, its computational cost remains a bottleneck as we scale in sequence length and graph size. While dense and block-sparse attention have benefited from hardware-aware algorithm design (e.g., FlashAttention), **sparse attention**—essential for graph-based learning and dynamic sparsity patterns—remains under-optimized on modern hardware accelerators.

This inefficiency is especially pronounced on GPUs with tensor cores, which deliver peak throughput for dense matrix multiplications with strict operand shapes. Sparse operations involve irregular memory accesses and unstructured computation, making them poorly suited for current tensor core design. As a result, **tensor cores remain largely underutilized for sparse workloads**.

Prior efforts fall into two categories:

1. **Individual kernel optimizations** — improving SDDMM and/or SpMM in isolation, but incurring unnecessary data movement when intermediate results are materialized in global memory.
2. **Kernel fusion** — reducing memory traffic by combining operations, but existing fused kernels target only CPUs or CUDA cores, leaving tensor core acceleration untapped.

**No existing work fuses the 3S operations while targeting tensor cores** — until Fused3S.

## The 3S Computational Pattern

The 3S pattern computes sparse attention as:

**O = softmax(QK<sup>T</sup> ⊙ A) V**

where Q, K, V, O ∈ ℝ<sup>N×d</sup> are dense matrices and A ∈ ℝ<sup>N×N</sup> is a sparse matrix defining attention patterns (e.g., adjacency or masking). This decomposes into three operations:

1. **SDDMM** — Compute attention scores S = QK<sup>T</sup> ⊙ A, where the dense product is computed only for non-zeros in A.
2. **Softmax** — Normalize scores row-wise: E = softmax(S).
3. **SpMM** — Aggregate output: O = EV.

This pattern appears across Graph Attention Networks (GAT), Graph Transformers (GT), and Sparse Transformers — all sharing the same 3S bottleneck on modern hardware.

## Key Contributions

### 1. Binary Sparse Block (BSB) Format

We introduce the **Binary Sparse Block (BSB)** format to efficiently map a sparse matrix onto tensor cores. BSB extends prior tensor-core-aware formats but reduces overhead by encoding sparsity with a fixed-size bitmap instead of integer indices.

<figure class="figure-medium">
<img src="/images/fused3s/sparseRepresentation.png" alt="Binary Sparse Block (BSB) format" />
<figcaption>Binary Sparse Block (BSB) format. The sparse matrix is divided into row windows, compacted by removing zero-only columns, then tiled into tensor core blocks (TCBs). Each TCB's sparsity pattern is stored as a compact bitmap.</figcaption>
</figure>

The construction proceeds as:
- Divide the sparse matrix into **row windows** of size *r*.
- Within each row window, **eliminate columns** containing only zeros to increase compute density.
- Partition the compacted row window into **tensor core blocks (TCBs)** of shape *r × c* aligned with MMA tile sizes (e.g., 16 × 8).
- Store a **bitmap** encoding the sparsity pattern in each TCB (128 bits for a 16×8 block), eliminating indexing overhead.

### 2. Fused On-Chip Algorithm

Fused3S fuses SDDMM, softmax, and SpMM into a **single GPU kernel** to reuse intermediate results in registers and shared memory, avoiding costly global memory round-trips.

<figure class="figure-medium">
<img src="/images/fused3s/tbNodeVsEdgeParallel.png" alt="Node-parallel vs edge-parallel strategies" />
<figcaption>Comparison of node-parallel (top) and edge-parallel (bottom) strategies. In node-parallel, each thread block owns all data needed for its rows, avoiding inter-block synchronization.</figcaption>
</figure>

We adopt **node-parallel fusion** where each thread block handles a row window, keeping all softmax and SpMM data local. To address load imbalance in graphs with irregular degree distributions, we apply **row window reordering** — sorting row windows by decreasing TCB count so that the heaviest work is scheduled first when more parallelism is available.

### 3. Warp Partitioning and Register Remapping

Within each thread block, we use a **split-column** warp partitioning strategy where each warp computes independent tiles of the output, eliminating inter-warp synchronization.

<figure class="figure-medium">
<img src="/images/fused3s/warpParallel.png" alt="Warp partitioning strategies" />
<figcaption>Split-column (top) vs. split-row (bottom) warp partitioning. In split-column, each warp independently computes a tile of S and O without inter-warp synchronization.</figcaption>
</figure>

We further optimize memory access through **register remapping** — permuting column layouts of K and V to enable 128-bit coalesced loads instead of scattered 32-bit loads. We use the PTX `mma` interface to load operands directly from HBM into registers, bypassing shared memory for single-use data.

<figure class="figure-medium">
<img src="/images/fused3s/RegRemap.png" alt="Register remapping optimization" />
<figcaption>Register remapping in SDDMM (left) and SpMM (right). Top: original scattered access layouts. Bottom: permuted layouts enabling coalesced 128-bit loads.</figcaption>
</figure>

## Results

We evaluate on NVIDIA **A30** (Ampere, 56 SMs) and **H100** (Hopper, 132 SMs) GPUs across 15 single-graph datasets and batched graphs from LRGB and OGB benchmarks.

### 3S Kernel Performance

**On single graphs**, Fused3S consistently outperforms all baselines:

<figure>
<img src="/images/fused3s/speedup_full_GH200.png" alt="Kernel speedup on H100" />
<figcaption>3S kernel performance on single graphs, H100. Fused3S achieves 2.8×, 2.2×, 1.6×, 4.4× and 14.7× geometric mean speedup over the baselines.</figcaption>
</figure>

<figure>
<img src="/images/fused3s/speedup_full_A30.png" alt="Kernel speedup on A30" />
<figcaption>3S kernel performance on single graphs, A30. Fused3S achieves 2.7×, 1.7×, 1.5×, 2.2×, and 12.3× geometric mean speedup over the baselines.</figcaption>
</figure>

**On batched graphs**, the gains are even more pronounced:

<figure>
<img src="/images/fused3s/speedup_batched_GH200.png" alt="Kernel speedup on batched graphs, H100" />
<figcaption>3S kernel performance on batched graphs, H100. Fused3S achieves up to 16.3× geometric mean speedup.</figcaption>
</figure>

<figure>
<img src="/images/fused3s/speedup_batched_A30.png" alt="Kernel speedup on batched graphs, A30" />
<figcaption>3S kernel performance on batched graphs, A30. Fused3S achieves up to 14× geometric mean speedup.</figcaption>
</figure>

By avoiding materialization of the intermediate score matrix, Fused3S also **reduces memory consumption**, enabling execution on large graphs where other methods run out of memory.

### End-to-End Graph Transformer Inference

Integrated into a 10-layer Graph Transformer model, Fused3S accelerates end-to-end inference:

<figure>
<img src="/images/fused3s/graph_transformer_speedup_A30.png" alt="GT inference speedup on A30" />
<figcaption>Graph Transformer inference on A30. Fused3S achieves 1.55×, 1.29×, 1.10×, and 3.08× speedup over the baselines.</figcaption>
</figure>

<figure>
<img src="/images/fused3s/graph_transformer_speedup_H100.png" alt="GT inference speedup on H100" />
<figcaption>Graph Transformer inference on H100. Fused3S achieves 1.56×, 1.05×, 1.15× and 5.36× speedup over the baselines.</figcaption>
</figure>

## BibTeX

```bibtex
@inproceedings{li2025fused3s,
  title     = {Fused3S: Fast Sparse Attention on Tensor Cores},
  author    = {Li, Zitong and Chandramowlishwaran, Aparna},
  booktitle = {Proceedings of the 2025 International Conference on Supercomputing (ICS)},
  year      = {2025},
  doi       = {10.1145/3721145.3730430}
}
```
