# Multimodal SLM Research for PDF Extraction (Updated)

## Executive Summary

Based on technical analysis for **CPU-only + 8GB RAM** environments processing Brazilian fiscal documents.

---

## Primary Recommendation: Qwen2-VL-2B-Instruct

| Aspect           | Details                   |
| ---------------- | ------------------------- |
| **Parameters**   | 2.2 Billion               |
| **RAM (Q4_K_M)** | ~1.7-2.5GB (Safe for 8GB) |
| **Key Feature**  | Dynamic Resolution        |
| **Use Case**     | DANFE, Boletos, Recibos   |

### Why Qwen2-VL-2B

1. **Dynamic Resolution**: Handles A4 documents with small text (6-8pt fonts in tax tables)
2. **M-ROPE**: Understands 2D spatial layout (correlates columns in tables)
3. **Native Portuguese**: Efficient tokenization
4. **llama.cpp support**: CPU inference via GGUF quantization

### Setup Commands

```bash
# 1. Create models directory
mkdir models

# 2. Download GGUF model (~1.7GB)
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct-GGUF \
    qwen2-vl-2b-instruct-q4_k_m.gguf --local-dir models

# 3. Set environment variables
set QWEN2VL_MODEL_PATH=models/qwen2-vl-2b-instruct-q4_k_m.gguf
```

---

## Secondary Options

| Model         | Size | RAM    | Best For               |
| ------------- | ---- | ------ | ---------------------- |
| SmolDocling   | 256M | ~2GB   | Fast OCR fallback      |
| MiniCPM-V 2.6 | 8B   | ~5.2GB | High precision (risky) |
| Florence-2    | 0.7B | ~1.5GB | Pre-processing only    |

---

## Current Implementation

### Files Created

- `qwen2vl_voter.py` - Qwen2-VL via llama-cpp-python
- `smol_docling_voter.py` - SmolDocling fallback
- `ensemble_extractor.py` - Regex + GLiNER + VLM + OCR

### Test Results (24 iterations)

| Metric       | Start | Now        |
| ------------ | ----- | ---------- |
| Error Rate   | 42%   | **27.49%** |
| Perfect      | 0%    | **17.7%**  |
| Amount Error | 35%   | **13.6%**  |

---

## Hybrid Architecture (Recommended)

```
PDF Input
    │
    ├─► Regex Voter (30+ patterns)
    │       ↓ confidence
    ├─► GLiNER (150 tokens, sliding windows)
    │       ↓ if <60% confidence
    ├─► Qwen2-VL-2B (visual understanding)
    │       ↓ for boletos
    └─► zbar/pyzbar (barcode decoding)
```

> [!IMPORTANT]
> For boletos, use VLM to **locate** barcode area, then `pyzbar` to **decode**.
> VLMs hallucinate long numeric sequences.
