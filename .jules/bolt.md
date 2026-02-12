## 2026-02-12 - [Batch Embedding Optimization]
**Learning:** Sequential processing of embeddings in Vision-Language Models like ColPali is a major bottleneck due to the blocking overhead of per-item GPU memory cleanup (~0.08s per call) and inefficient GPU utilization.
**Action:** Always implement batch embedding generation and move memory cleanup to the batch level to maximize throughput during document ingestion.
