---
title: Vector Databases - Internals, Indexes & Distributed Architecture
description: tA Complete Deep-Dive Into HNSW, IVF, PQ, Embeddings, Metadata Filtering & Multi-Node Search Systemso
---

# **Vector Databases: Internals, Indexes & Distributed Architecture**

### *A Complete Deep-Dive Into HNSW, IVF, PQ, Embeddings, Metadata Filtering & Multi-Node Search Systems*

Vector databases have exploded in popularity because large language models (LLMs) turned **embeddings** into the new “universal representation” for text, images, video, audio, code, protein sequences — anything that can be encoded into high-dimensional vector space.

In 2024–25, vector database questions became a **core part of HLD interviews** at FAANG, MAMAA, and Indian product companies:

* “Explain how HNSW works.”
* “How does a vector database scale horizontally?”
* “How do you combine ANN search with metadata filters?”
* “How does PQ compress vector space?”
* “Why are insertions slow in HNSW?”
* “What index would you pick for 100M vectors?”
* “How does Pinecone/Milvus shard data?”
* “What if you need freshness + fast ingestion?”

This article teaches everything you need for interviews **and actual engineering work**:
From embeddings → indexing → compression → filters → distributed search → ingestion → consistency → system design.

This is your interview guide on vector databases.

---

# **Table of Contents**

1. Why Vector Databases Exist
2. Embeddings 101 — The Semantic Foundation
3. The Challenge: High-Dimensional Nearest Neighbor Search
4. Exact Search vs Approximate Search
5. ANN Indexes: Categories & Trade-offs
6. **HNSW Deep Dive (Most Important)**
7. **IVF / IVF-Flat / IVF-PQ**
8. **PQ, OPQ, Scalar Quantization**
9. **Hybrid Index Architectures (HNSW-PQ, IVF-HNSW)**
10. Storage Engine Design for Vector Databases
11. WAL, segments, delta logs & compaction
12. Distributed Architecture (Sharding, Replication, Routing)
13. Metadata Filtering + Vector Search Fusion
14. Real-Time Ingestion Challenges
15. Memory, Latency, Recall Tuning
16. Case Studies: FAISS, Pinecone, Milvus, Weaviate, OpenSearch
17. Interview Mental Models
18. Summary

---

# **1. Why Vector Databases Exist**

LLMs turned all content into vector form.

Instead of keyword matching (`BM25`, inverted indexes), embeddings allow engines to search for meaning:

* “dog” and “puppy” → close in vector space
* “machine learning engineer” and “AI developer” → similar
* “refund request” and “money back issue” → close

Embeddings extract semantic information that keyword search simply cannot.

This creates a new problem:

### **How do you search millions or billions of 768-dimensional vectors?**

Naively, by computing:

```
distance(query, vector_i)
```

for every vector.
This is **O(N × d)** → too slow beyond 100K–1M vectors.

Vector databases exist to:

* store embeddings
* index them efficiently
* perform approximate nearest neighbor (ANN) search
* scale horizontally
* support filters
* handle ingestion
* provide consistency

They are specialized search engines — not general databases.

---

# **2. Embeddings 101 — The Semantic Foundation**

An embedding is a **d-dimensional float vector**:

```
v = [0.12, -0.83, 0.44, …]
```

Common embeddings:

* 384 dimensions (MiniLM)
* 768 dimensions (BERT)
* 1024–4096 dimensions (OpenAI / Llama)
* 8k dimensions (image embeddings)

Distance metrics:

1. **Cosine similarity** (most common)
2. **L2 distance (Euclidean)**
3. **Inner product (dot-product)**

Cosine similarity requires normalization:

```
v_normalized = v / ||v||
```

Distance and similarity are interchangeable:

```
cosine_similarity = dot(q, v)
l2_distance = ||q - v||
```

Vector DBs convert similarity queries into **top-K nearest neighbor searches**.

---

# **3. The Challenge: High-Dimensional Nearest Neighbor Search**

Exact search is easy:

```
for v in all_vectors:
    compute distance(q, v)
return top K results
```

But high-dimensional spaces suffer from:

### **The Curse of Dimensionality**

* Distances become less meaningful
* Indexes like KD-trees collapse
* Partitioning becomes ineffective
* Clustering becomes expensive

Thus the shift to **Approximate Nearest Neighbor (ANN)**:

> Return results that are *very close* to the nearest ones, but not necessarily the exact top-K.

ANN trades off accuracy for speed → this is acceptable for most LLM applications.

---

# **4. Exact Search vs Approximate Search**

| Property | Exact          | Approximate (ANN) |
| -------- | -------------- | ----------------- |
| Accuracy | 100%           | ~90–99%           |
| Latency  | Slow           | Fast              |
| Memory   | High           | Medium/High       |
| Use Case | Small datasets | >1M vectors       |

Exact search uses:

* Flat index (brute force)
* GPU acceleration (FAISS GPU)

ANN uses:

* HNSW
* IVF
* PQ
* ScaNN
* ANNoy
* DiskANN

Vector databases almost always choose ANN.

---

# **5. ANN Index Families (The Big Picture)**

Three major families dominate ANN indexing:

```
Graph-based:    HNSW, NSG
Tree/Cluster:   IVF, KMeans, LSH
Quantization:   PQ, OPQ, SQ
```

They differ in accuracy, speed, memory footprint, and ingestion cost.

---

# **6. HNSW Deep Dive (Most Important for Interviews)**

### *Hierarchical Navigable Small Worlds — The King of ANN Indexes*

HNSW is used by:

* Pinecone
* Weaviate
* Qdrant
* Vespa
* Milvus
* Many enterprise vector search engines

It is widely considered the **best all-around ANN index**.

---

## **6.1 Intuition: A Multi-Level Navigable Graph**

HNSW builds a graph where:

* nodes = vectors
* edges = proximity links
* upper layers = fewer nodes, long-range links
* bottom layer = dense links, local neighbors

Diagram:

```
Level 3:     A ---- B
             |      |
Level 2:   C ---- D ---- E
           |       \
Level 1: F -- G -- H -- I -- J
           \    \      \ 
Level 0:  (dense graph with nearest neighbors)
```

Searching works by:

1. Start at top layer
2. Greedily move to neighbor closer to query
3. Drop to next layer
4. Repeat
5. Use BFS/priority queue at bottom layer

Search complexity is **O(log N)**.

---

## **6.2 Key Parameters**

### **M**

Max number of neighbors.

Higher M → higher accuracy, more memory.

### **efConstruction**

Quality of graph during build.

Higher → better recall, slower build.

### **efSearch**

Search beam width.

Higher → better recall, slower search.

---

## **6.3 HNSW Strengths**

* Best-in-class recall/latency trade-off
* Incremental inserts supported
* Very strong locality-sensitive navigation
* Executes on CPU, no GPU required

---

## **6.4 HNSW Weaknesses**

* Memory heavy (graph edges ~ 64–128 bytes per edge)
* Insertions expensive
* Deletions complicated (lazy deletion + rebuild)
* Harder to persist to disk (needs compact segments)
* Poor for streaming ingestion >10–50K writes/sec

Thus many vector DBs use:

* HNSW for in-memory serving
* On-disk indexes (IVF-PQ, DiskANN) for cold storage

---

# **7. IVF (Inverted File Index) & IVF-PQ**

IVF is the **coarse quantization** approach.

---

## **7.1 Concept**

1. Run k-means clustering on all vectors.
2. Assign each vector to its nearest centroid.
3. Search only in a few nearest clusters.

Visual:

```
100M vectors
↓
1000 clusters
↓
At query time: search only 4 clusters (4/1000 = 0.4%)
```

Huge speedup.

---

## **7.2 IVF-Flat**

IVF without compression.

Inside each cluster, you search raw vectors.

Fast but heavy on memory.

---

## **7.3 IVF-PQ (Product Quantization)**

PQ compresses vectors:

* Split vector into subspaces
* Quantize each subspace
* Store compact codes (8–16 bytes per vector instead of 512–4096 bytes)

Example:

```
128-dim vector
↓ split into 8 segments
each segment encoded to 1 byte
↓
8 bytes total storage
```

Reduction from **512 bytes → ~8 bytes**.

This allows 1B vectors to fit in memory.

Downside: lower recall.

---

# **8. Product Quantization (PQ) — Compression for Massive Scale**

PQ is critical for large-scale vector databases.

### **PQ breaks vectors into subspaces:**

```
128 dims → 8 groups of 16 dims
```

### **Each group quantized:**

```
v = [segment1][segment2]…[segment8]
```

Each segment encoded as 1 byte.

### **Distance computed using lookup tables**

Distance lookup speed is extremely fast.

PQ variants:

* **PQ** — basic
* **OPQ** — rotates vector space to improve quantization
* **SQ** — scalar quantization (1D quantization per dimension)

PQ allows FAISS to search **billion-scale** datasets on a single machine.

---

# **9. Hybrid Indexes (HNSW + PQ, IVF + HNSW)**

Real systems combine indexes:

* **HNSW for top layer**, PQ for bottom layer
* **IVF + HNSW refined search**
* **HNSW graph over PQ vectors**

Hybrid systems provide:

* High recall
* Low memory
* Fast latency
* Good ingestion performance

This is how Milvus and Pinecone optimize real workloads.

---

# **10. Storage Engine Internals**

Unlike relational DBs, vector DBs must store:

* raw vectors
* compressed vectors (PQ)
* metadata
* filters (inverted index)
* index files (graph, cluster assignments)
* WAL
* snapshot files

### **Segment Files (LSM-inspired)**

Most systems (Milvus, Pinecone) use segments:

```
Segment 1: 500K vectors + index
Segment 2: 500K vectors + index
Segment 3: new writes (in-memory)
```

When a segment grows:

* sealed
* persisted
* indexed via HNSW / IVF
* merged later

This solves the "HNSW is slow to insert" problem.

---

# **11. WAL, Delta Logs & Compaction**

Vector DBs maintain durability with:

* Write-Ahead Logs
* Segment logs
* Batch writes
* Periodic snapshotting

Process:

```
writes → WAL
small in-memory index
flush → segment file
segment indexing → background
compaction → merge old segments
```

This mirrors RocksDB but adapted for ANN indexing.

---

# **12. Distributed Architecture (Sharding, Replication, Routing)**

Modern vector DBs must scale horizontally.

The architecture typically looks like:

```
Query Node / Coordinator
       ↓
Shards (Segment replicas)
       ↓
Index Nodes (HNSW/IVF/PQ)
```

---

## **12.1 Sharding Strategies**

### **A. Hash by ID**

Uniform distribution, simple.

### **B. Hash by vector**

Rare, because embeddings don’t hash uniformly.

### **C. Cluster-based sharding (IVF centroids)**

Shards correspond to cluster partitions.

### **D. Range-based (for metadata hybrid DBs)**

Used in Weaviate hybrid search.

---

## **12.2 Replication**

Replication modes:

* **sync** — strong consistency
* **async** — eventual consistency
* **multi-primary** — conflict resolution needed

Often replicates entire segment files.

---

## **12.3 Query Routing**

Query steps:

1. Coordinator determines which shards are relevant
2. Sends query vector to top candidate shards
3. Each shard returns top-K partial results
4. Coordinator merges into global top-K

Diagram:

```
Query
 ↓
Coordinator
 ↓            ↓             ↓
Shard 1     Shard 2      Shard 3
 ↓            ↓             ↓
Top-K1      Top-K2       Top-K3
 ↓ merge
Final Top-K
```

---

# **13. Metadata Filtering + ANN Fusion**

One of the hardest problems in vector databases.

Example:

```
Find products semantically similar to Q where:
brand = "Nike"
price between 2000 and 4000
category = "Shoes"
```

ANN indexes don’t handle filters well.
Vector DBs solve this using **hybrid indexes**:

---

## **Solution Approaches**

### **1. Pre-filtering (Filter → ANN)**

Filter narrow set → ANN search on remaining.

Good when filters are highly selective.

---

### **2. Post-filtering (ANN → Filter)**

ANN returns candidates → filter them.

May degrade recall if filtered too aggressively.

---

### **3. Inverted index + ANN hybrid**

Systems like Weaviate store:

* Vector index (HNSW)
* Inverted index for metadata

Hybrid score computed using weighted fusion.

---

### **4. Pinecone Sparse-Dense Hybrid**

Dense vector + sparse keyword vector (BM25-like):

```
score = w_dense * sim_dense + w_sparse * sim_sparse
```

This is extremely effective for RAG pipelines.

---

# **14. Real-Time Ingestion Challenges**

HNSW is slow to insert into because:

* Graph rewiring required
* Each insertion updates multiple edges
* efConstruction parameter heavily affects time

Solutions:

### **A. Buffer writes into log segments**

Then build HNSW offline.

### **B. Use IVF-PQ for write-friendly ingestion**

### **C. Store writes in RAM index (small HNSW)**

Merge later.

### **D. Append-only segments + periodic rebuild**

Like LSM compaction.

---

# **15. Memory, Latency & Recall Tuning**

### **Memory Footprint = Vectors + Index**

HNSW is memory heavy:

```
vector_size + M * neighbor_size * bytes
```

PQ compresses vector size massively.

---

## **Recall Tuning**

### **HNSW tunables**:

* efSearch
* efConstruction
* M

### **IVF tunables**:

* nprobe (# clusters searched)

Higher values → better recall, slower search.

---

## **Latency Optimization**

* SIMD instructions for distance calculation
* cache-aware block layout
* prefetching
* GPU-based search
* reducing dimensionality via PCA

---

# **16. Case Studies**

## **16.1 FAISS (Meta)**

* Best GPU-accelerated ANN library
* IVF-PQ, HNSW, Flat, OPQ
* Billion-scale search on a single node

---

## **16.2 Pinecone**

* Fully managed vector DB
* Built on HNSW + proprietary segmenting
* Adaptive capacity
* Multi-tenant isolation
* Sparse-dense hybrid retrieval

---

## **16.3 Milvus**

* Most advanced open-source vector DB
* Supports HNSW, IVF, IVF-PQ, DiskANN
* Distributed metadata server
* LSM-tree-like segment design

---

## **16.4 Weaviate**

* Hybrid search (vector + keyword)
* HNSW for vector search
* Inverted index for metadata
* Excellent for enterprise use cases

---

## **16.5 OpenSearch KNN**

* FAISS/HNSW under the hood
* Integrates with existing search pipeline

---

# **17. Interview Mental Models**

### **Q: Why ANN instead of exact search?**

High-dimensional exact search collapses after 100K vectors.

### **Q: When do you choose HNSW?**

When recall & latency are priority and ingestion is moderate.

### **Q: When IVF-PQ?**

When scaling to 100M–10B vectors with memory constraints.

### **Q: How do you scale vector search horizontally?**

Sharding + distributed top-K merging.

### **Q: Why are inserts slow in HNSW?**

Graph rewiring and greedy search to find neighbors.

### **Q: How do filters work with vectors?**

Hybrid indexing → pre-filter or post-filter strategies.

### **Q: How do vector DBs persist data?**

Segments + WAL + compaction.

### **Q: Which index for streaming ingestion?**

IVF-flat or HNSW + segment buffering.

---

# **18. Summary (The one-sentence version)**

> A vector database is a specialized distributed search engine that combines embeddings, ANN indexes (HNSW/IVF/PQ), metadata filters, write-ahead logging, segment storage, and multi-node top-K routing to deliver low-latency semantic search at massive scale.
