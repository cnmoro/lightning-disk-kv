# ⚡ lightning-disk-kv

This project is an absurdly fast, sharded Key-Value storage engine designed for high-throughput Python applications. 

It is a drop-in solution for machine learning pipelines that need to store millions of embeddings (or other data type) samples efficiently. It solves the **Global Interpreter Lock (GIL)** bottleneck by offloading hashing, serialization, and disk I/O to parallel Rust threads.

### 🚀 Key Features

*   **True Parallelism:** Writes to multiple LMDB shards simultaneously using all CPU cores.
*   **Zero-Copy Vectors:** Specialized "Fast Path" for `numpy` arrays that writes raw bytes to disk (no pickling).
*   **Generic Storage:** Capable of storing arbitrary Python objects (Strings, Dicts, Lists) via optimized parallel pickling.
*   **Crash Safe:** Based on LMDB (Lightning Memory-Mapped Database), offering proven reliability.
*   **Redis Compatible:** Includes a wrapper that mimics the `redis-py` API for easy integration.

---

## 📦 Installation

### Option A: Install via Pip (Recommended)
```bash
pip install lightning_disk_kv
```

### Option B: Build from Source
If you are modifying the Rust code or building for a specific architecture:
```bash
# Requires Rust and Maturin
maturin develop --release
```

---

## ⚡ Usage Guide

### 1. Initialization
Initialize the database by specifying a base directory. The storage engine automatically handles sharding (splitting data across multiple files) to maximize write speed.

```python
from lightning_disk_kv import LDKV

# Initialize with 5 shards.
# 'map_size' is the maximum virtual memory size. 
# It does NOT consume this amount of RAM immediately.
# Default is ~1TB, which is safe for 64-bit systems.
db = LDKV(
    base_path="./my_database", 
    num_shards=5, 
    map_size=100 * 1024**3  # 100 GB limit
)
```

### 2. Storing Vectors (The "Fast Path")
Use `store_vectors` when dealing with Numpy embeddings. This bypasses Python's overhead entirely by reading memory directly from C-pointers.

**Requirement:** Data must be `np.float32`.

```python
import numpy as np

# Create dummy data
ids = [1, 2, 3]
vectors = np.random.rand(3, 128).astype(np.float32)

# Store in parallel
db.store_vectors(vectors, ids)

# Retrieve
# Returns a list of numpy arrays, or None if the ID doesn't exist
results = db.get_vectors([1, 999])

print(results[0].shape)  # (128,)
print(results[1])        # None
```

### 3. Storing Objects (The "Generic Path")
Use `store_data` for strings, dictionaries, images, or lists. While this uses `pickle` internally, the serialization and disk writing happen in parallel threads, making it significantly faster than standard loops.

```python
ids = [100, 101]
data = [
    "A simple string", 
    {"key": "value", "meta": [1, 2, 3]}
]

db.store_data(data, ids)

results = db.get_data([100])
print(results[0]) # "A simple string"
```

### 4. Redis Compatibility API
We provide a `redis-py` compatible wrapper. This allows you to use `lightning-disk-kv` as an embedded, persistent Redis replacement without running a separate server process.

```python
from lightning_redis import LDKV_RedisCompat

# Initialize (replaces host/port with a file path)
r = LDKV_RedisCompat(base_path="./redis_data", decode_responses=True)

# Basic Key-Value
r.set('foo', 'bar')
print(r.get('foo'))  # 'bar'

# TTL (Time To Live) - key automatically removed after 5 seconds
r.set('temp_key', 'hidden', ex=5)

# Atomic Counters
r.incr('visitor_count', amount=1)

# Hash Maps
r.hset('user:100', mapping={'name': 'Alice', 'role': 'admin'})
print(r.hgetall('user:100')) # {'name': 'Alice', 'role': 'admin'}
```

### 5. Management & Syncing

```python
# Check total number of items across all shards
count = db.get_data_count()
print(f"Total items: {count}")

# Delete items
db.delete_data([1, 100])

# Force flush to disk
# The engine uses OS buffers for maximum speed. 
# Call .sync() to ensure data is physically written to the drive.
db.sync()
```

---

## ⚠️ Configuration & Safety

### Understanding `map_size`
LMDB uses a memory map. You must set `map_size` larger than the maximum data you ever intend to store. 
*   **Don't worry about RAM:** Setting this to 1TB does not use 1TB of RAM. It simply reserves virtual address space.
*   **Error handling:** If you exceed this limit, you will get a `MapFull` error.

### Durability vs. Speed
To achieve maximum throughput, `lightning_disk_kv` sets the `MDB_NOSYNC` flag by default.
*   **Application Crash:** Data is safe.
*   **OS Crash / Power Cut:** Data currently in the OS buffer (last few seconds) might be lost.
*   **Best Practice:** If data durability is critical (e.g., you can't re-generate the data), call `db.sync()` periodically or after a large bulk insert.

---

## 🛠 Building from Source (Advanced)

If you cannot install via pip, you must compile the Rust backend manually.

1.  **Install Rust:**
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
2.  **Install the builder:**
    ```bash
    pip install maturin
    ```
3.  **Compile:**
    Navigate to the project root and run:
    ```bash
    maturin develop --release
    ```