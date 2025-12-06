import numpy as np
import time
import shutil
import os
from lightning_disk_kv import LDKV
import secrets

# Clean previous runs
DB_PATH = "./db_rust_test"
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

# Initialize
# 5 shards, 100GB map size
storage = LDKV(DB_PATH, num_shards=20, map_size=100 * 1024**3)

# ---------------------------------------------------------
# 1. Store Vectors (Fastest Path)
# ---------------------------------------------------------
N = 1_000_000
DIM = 3072
print(f"--- Generating {N} Vectors ---")
vec_data = np.random.rand(N, DIM).astype(np.float32)

# generate random 64-bit integer IDs
vec_ids = [secrets.randbits(63) for _ in range(N)]

print("Storing Vectors...")
start = time.time()
storage.store_vectors(vec_data, vec_ids)
print(f"Stored in {time.time() - start:.4f}s")

# ---------------------------------------------------------
# 2. Get Vectors
# ---------------------------------------------------------
print("Retrieving Vectors...")
start = time.time()
# Fetch 0, 10, and a non-existent ID
retrieved_vecs = storage.get_vectors([0, 10, 99999999])
print(f"Retrieved in {time.time() - start:.4f}s")
print(f"Shape of ID 0: {retrieved_vecs[0].shape}")
print(f"ID 99999999 is: {retrieved_vecs[2]}") # Should be None

# ---------------------------------------------------------
# 3. Store Generic Data (Strings, Lists, Objects)
# ---------------------------------------------------------
print("\n--- Testing Generic Data Storage ---")
obj_ids = [1000001, 1000002, 1000003]
obj_data = [
    "I am a simple string",
    {"key": "I am a dictionary", "val": [1, 2, 3]},
    ("I", "am", "a", "tuple")
]

start = time.time()
storage.store_data(obj_data, obj_ids)
print(f"Stored Generic Data in {time.time() - start:.4f}s")

# ---------------------------------------------------------
# 4. Get Generic Data
# ---------------------------------------------------------
retrieved_objs = storage.get_data(obj_ids)
print(f"Retrieved String: {retrieved_objs[0]}")
print(f"Retrieved Dict: {retrieved_objs[1]}")

# ---------------------------------------------------------
# 5. Get Count & Delete
# ---------------------------------------------------------
count = storage.get_data_count()
print(f"\nTotal items in DB: {count}") # Should be 100,000 + 3

print("Deleting the dictionary (ID 1000002)...")
storage.delete_data([1000002])

count_after = storage.get_data_count()
print(f"Total items after delete: {count_after}")

# Verify delete
check = storage.get_data([1000002])
print(f"Trying to get deleted ID: {check[0]}") # Should be None

# Sync to disk
storage.sync()