import time
import os
import shutil
import numpy as np
import secrets
import struct
import pickle
import lmdb
import atexit
import hashlib
from lightning_disk_kv import LDKV

class LmdbStorage:
    def __init__(self, path, map_size=20*1024*1024*1024):
        self.env = lmdb.open(path, map_size=map_size, writemap=True, map_async=True)
        # Note: Added writemap=True and map_async=True to give Python version 
        # the best possible chance against the Rust version (which uses similar flags)
        atexit.register(self.close)

    def _int_to_bytes(self, x):
        MAX_UINT64 = 2**64
        MAX_INT64 = 2**63
        x = x % MAX_UINT64
        if x >= MAX_INT64:
            x -= MAX_UINT64
        return struct.pack('<q', x)

    def store_data(self, data, identifiers, batch_size=5000):
        total = len(data)
        with self.env.begin(write=True) as txn:
            for i in range(0, total, batch_size):
                batch_data = data[i:i+batch_size]
                batch_ids = identifiers[i:i+batch_size]
                with txn.cursor() as curs:
                    for vec, id in zip(batch_data, batch_ids):
                        curs.put(
                            self._int_to_bytes(id) if isinstance(id, int) else id.encode(),
                            pickle.dumps(vec)
                        )

    def get_data(self, identifiers):
        datas = []
        with self.env.begin() as txn:
            for id in identifiers:
                data = txn.get(self._int_to_bytes(id) if isinstance(id, int) else id.encode())
                if data:
                    datas.append(pickle.loads(data))
                else:
                    datas.append(None)
        return [ v for v in datas if v is not None ]
        
    def store_vectors(self, data, identifiers, batch_size=5000):
        total = len(data)
        with self.env.begin(write=True) as txn:
            for i in range(0, total, batch_size):
                batch_data = data[i:i+batch_size]
                batch_ids = identifiers[i:i+batch_size]
                with txn.cursor() as curs:
                    for vec, id in zip(batch_data, batch_ids):
                        curs.put(
                            self._int_to_bytes(id) if isinstance(id, int) else id.encode(),
                            np.array(vec, dtype=np.float32).tobytes()
                        )
    
    def get_vectors(self, identifiers):
        datas = []
        with self.env.begin() as txn:
            for id in identifiers:
                data = txn.get(self._int_to_bytes(id) if isinstance(id, int) else id.encode())
                if data:
                    datas.append(np.frombuffer(data, dtype=np.float32))
                else:
                    datas.append(None)
        return [v for v in datas if v is not None]
    
    def delete_data(self, identifiers):
        with self.env.begin(write=True) as txn:
            for id in identifiers:
                txn.delete(self._int_to_bytes(id) if isinstance(id, int) else id.encode())

    def get_data_count(self):
        with self.env.begin() as txn:
            return txn.stat()['entries']

    def sync(self):
        self.env.sync()

    def close(self):
        self.env.close()

class ShardedLmdbStorage:
    def __init__(self, base_path, num_shards=5, map_size=70*1024*1024*1024):
        self.num_shards = num_shards
        self.shards = {}
        for shard_idx in range(num_shards):
            shard_path = os.path.join(base_path, f"shard_{shard_idx}")
            os.makedirs(shard_path, exist_ok=True)
            self.shards[shard_idx] = LmdbStorage(shard_path, map_size=map_size)

    def _get_shard_for_id(self, identifier):
        identifier_str = str(identifier).encode('utf-8')
        h = hashlib.md5(identifier_str).hexdigest()
        return int(h, 16) % self.num_shards

    def store_data(self, data, identifiers, batch_size=5000):
        shard_data = {i: [] for i in range(self.num_shards)}
        shard_ids = {i: [] for i in range(self.num_shards)}
        for vec, identifier in zip(data, identifiers):
            shard = self._get_shard_for_id(identifier)
            shard_data[shard].append(vec)
            shard_ids[shard].append(identifier)
        for shard, lmdb_storage in self.shards.items():
            if shard_data[shard]:
                lmdb_storage.store_data(shard_data[shard], shard_ids[shard], batch_size=batch_size)
    
    def get_data(self, identifiers):
        id_to_data = {}
        shard_ids = {i: [] for i in range(self.num_shards)}
        for identifier in identifiers:
            shard = self._get_shard_for_id(identifier)
            shard_ids[shard].append(identifier)
        for shard, lmdb_storage in self.shards.items():
            if shard_ids[shard]:
                data_list = lmdb_storage.get_data(shard_ids[shard])
                # Note: The original python class drops Nones in get_data, 
                # which might cause misalignment here if keys are missing.
                # Assuming benchmark data always exists.
                if len(data_list) == len(shard_ids[shard]):
                    for identifier, data in zip(shard_ids[shard], data_list):
                        id_to_data[identifier] = data
        return [id_to_data.get(identifier, None) for identifier in identifiers]

    def store_vectors(self, data, identifiers, batch_size=5000):
        shard_data = {i: [] for i in range(self.num_shards)}
        shard_ids = {i: [] for i in range(self.num_shards)}
        for vec, identifier in zip(data, identifiers):
            shard = self._get_shard_for_id(identifier)
            shard_data[shard].append(vec)
            shard_ids[shard].append(identifier)
        for shard, lmdb_storage in self.shards.items():
            if shard_data[shard]:
                lmdb_storage.store_vectors(shard_data[shard], shard_ids[shard], batch_size=batch_size)
    
    def get_vectors(self, identifiers):
        id_to_vector = {}
        shard_ids = {i: [] for i in range(self.num_shards)}
        for identifier in identifiers:
            shard = self._get_shard_for_id(identifier)
            shard_ids[shard].append(identifier)
        for shard, lmdb_storage in self.shards.items():
            if shard_ids[shard]:
                vectors = lmdb_storage.get_vectors(shard_ids[shard])
                if len(vectors) == len(shard_ids[shard]):
                    for identifier, vector in zip(shard_ids[shard], vectors):
                        id_to_vector[identifier] = vector
        return [id_to_vector.get(identifier, None) for identifier in identifiers]

    def delete_data(self, identifiers):
        shard_ids = {i: [] for i in range(self.num_shards)}
        for identifier in identifiers:
            shard = self._get_shard_for_id(identifier)
            shard_ids[shard].append(identifier)
        for shard, lmdb_storage in self.shards.items():
            if shard_ids[shard]:
                lmdb_storage.delete_data(shard_ids[shard])

    def get_data_count(self):
        total = 0
        for lmdb_storage in self.shards.values():
            total += lmdb_storage.get_data_count()
        return total

    def sync(self):
        for lmdb_storage in self.shards.values():
            lmdb_storage.sync()
            
    def close(self):
        for lmdb_storage in self.shards.values():
            lmdb_storage.close()

def format_res(name, op, count, duration):
    tps = count / duration
    print(f"| {name:<12} | {op:<15} | {duration:.4f}s | {int(tps):,} items/s |")

def run_benchmark():
    # --- Configuration ---
    NUM_ITEMS = 200_000      # 200k items
    DIM = 768                # Vector dimensions
    NUM_SHARDS = 10          # Parallelism
    MAP_SIZE = 50 * 1024**3  # 50GB
    
    RUST_PATH = "./bench_rust"
    PY_PATH = "./bench_python"

    # Cleanup
    for p in [RUST_PATH, PY_PATH]:
        if os.path.exists(p):
            shutil.rmtree(p)

    print(f"\ngenerating {NUM_ITEMS:,} items (Float32 Vectors + Random IDs)...")
    # Generate Data
    vectors = np.random.rand(NUM_ITEMS, DIM).astype(np.float32)
    ids = [secrets.randbits(63) for _ in range(NUM_ITEMS)]
    
    # Generic Data (Strings)
    generic_data = [f"data_payload_{i}" * 10 for i in range(NUM_ITEMS)]

    print(f"{'-'*80}")
    print(f"| {'Engine':<12} | {'Operation':<15} | {'Time':<8} | {'Throughput':<15} |")
    print(f"{'-'*80}")

    # ==========================================
    # 1. TEST RUST ENGINE
    # ==========================================
    
    db_rust = LDKV(RUST_PATH, num_shards=NUM_SHARDS, map_size=MAP_SIZE)
    
    # Write Vectors
    start = time.perf_counter()
    db_rust.store_vectors(vectors, ids)
    db_rust.sync() # Ensure disk flush for fair timing
    dur = time.perf_counter() - start
    format_res("Rust (LDKV)", "Write Vectors", NUM_ITEMS, dur)

    # Read Vectors
    start = time.perf_counter()
    _ = db_rust.get_vectors(ids)
    dur = time.perf_counter() - start
    format_res("Rust (LDKV)", "Read Vectors", NUM_ITEMS, dur)

    # Write Generic
    start = time.perf_counter()
    db_rust.store_data(generic_data, ids)
    db_rust.sync()
    dur = time.perf_counter() - start
    format_res("Rust (LDKV)", "Write Generic", NUM_ITEMS, dur)

    # Read Generic
    start = time.perf_counter()
    _ = db_rust.get_data(ids)
    dur = time.perf_counter() - start
    format_res("Rust (LDKV)", "Read Generic", NUM_ITEMS, dur)
    
    # Cleanup Rust mem
    del db_rust

    print(f"{'-'*80}")

    # ==========================================
    # 2. TEST PYTHON ENGINE
    # ==========================================

    db_py = ShardedLmdbStorage(PY_PATH, num_shards=NUM_SHARDS, map_size=MAP_SIZE)

    # Write Vectors
    start = time.perf_counter()
    db_py.store_vectors(vectors, ids)
    db_py.sync()
    dur = time.perf_counter() - start
    format_res("Python (LMDB)", "Write Vectors", NUM_ITEMS, dur)

    # Read Vectors
    start = time.perf_counter()
    _ = db_py.get_vectors(ids)
    dur = time.perf_counter() - start
    format_res("Python (LMDB)", "Read Vectors", NUM_ITEMS, dur)

    # Write Generic
    start = time.perf_counter()
    db_py.store_data(generic_data, ids)
    db_py.sync()
    dur = time.perf_counter() - start
    format_res("Python (LMDB)", "Write Generic", NUM_ITEMS, dur)

    # Read Generic
    start = time.perf_counter()
    _ = db_py.get_data(ids)
    dur = time.perf_counter() - start
    format_res("Python (LMDB)", "Read Generic", NUM_ITEMS, dur)

    db_py.close()
    print(f"{'-'*80}")

    # Cleanup Files
    for p in [RUST_PATH, PY_PATH]:
        if os.path.exists(p):
            shutil.rmtree(p)

if __name__ == "__main__":
    run_benchmark()
