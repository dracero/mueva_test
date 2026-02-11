## 2025-05-14 - Redundant Memory Cleanup Overhead
**Learning:** Frequent calls to `gc.collect()` and `torch.cuda.empty_cache()` (even if no GPU is present) introduce significant latency (~0.08s per call in this environment). In a RAG indexing loop with hundreds of items, this adds up to several seconds of pure overhead.
**Action:** Move memory cleanup from per-item methods to the batch level (e.g., after processing a full batch of 8-16 items).

## 2025-05-14 - Batch Embedding Generation
**Learning:** ColPali models and processors natively support batching. Moving from single-item embedding generation to batch processing reduces model invocation overhead and enables better GPU/CPU utilization.
**Action:** Implement `_batch` methods for embeddings and refactor ingestion loops to iterate in steps of `Config.BATCH_SIZE`.
