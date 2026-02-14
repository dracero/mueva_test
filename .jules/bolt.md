## 2025-05-15 - [Batch Embedding & Redundant Search Optimization]
**Learning:** Indexing performance was significantly limited by sequential embedding generation and excessive GPU cache clearing. Additionally, the query flow had a ~50% overhead due to a redundant search call in the compatibility wrapper.
**Action:** Always implement batch methods for model inference (ColPali) and perform memory cleanup (gc/cuda) at the batch level rather than per-item. Ensure agent state results are reused in wrappers to avoid redundant retrieval.
