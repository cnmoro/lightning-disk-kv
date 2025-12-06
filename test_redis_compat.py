import time
import os
import shutil
from lightning_redis import LightningRedis

# Setup
DB_PATH = "./redis_compat_test"
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

# Initialize "Redis"
# We act just like redis-py, but with a file path instead of host/port
r = LightningRedis(base_path=DB_PATH, decode_responses=True)

print("--- Testing Basic KV ---")
r.set('foo', 'bar')
val = r.get('foo')
print(f"SET foo=bar, GET foo -> {val}")
assert val == 'bar'

print("\n--- Testing TTL (Expiry) ---")
r.set('temp', 'I will disappear', ex=1)
print(f"Immediate get: {r.get('temp')}")
time.sleep(1.2)
print(f"Get after 1.2s: {r.get('temp')}")
assert r.get('temp') is None

print("\n--- Testing NX (Not Exists) ---")
r.set('uniq', '1')
res = r.set('uniq', '2', nx=True) # Should fail
print(f"Set existing with nx=True result: {res}")
print(f"Value is still: {r.get('uniq')}")
assert res is None
assert r.get('uniq') == '1'

print("\n--- Testing MGET/MSET ---")
r.mset({'a': '1', 'b': '2', 'c': '3'})
vals = r.mget(['a', 'b', 'missing'])
print(f"MGET a, b, missing -> {vals}")
assert vals == ['1', '2', None]

print("\n--- Testing Counters ---")
r.set('count', 10)
r.incr('count')
r.incr('count', 5)
r.decr('count')
print(f"Counter (10 + 1 + 5 - 1) -> {r.get('count')}")
assert int(r.get('count')) == 15

print("\n--- Testing Hashes ---")
r.hset('user:100', mapping={'name': 'John', 'age': '30'})
r.hset('user:100', 'city', 'New York')
user = r.hgetall('user:100')
print(f"HGETALL user:100 -> {user}")
assert user['city'] == 'New York'
assert user['name'] == 'John'

print("\n--- Testing Bytes (Binary Safety) ---")
r_bin = LightningRedis(base_path=DB_PATH, decode_responses=False)
r_bin.set('binary', b'\x00\xFF\x10')
res = r_bin.get('binary')
print(f"Binary stored: {res}")
assert res == b'\x00\xFF\x10'

print("\n--- Persistence Check ---")
r.save()
del r
del r_bin

# Re-open
r2 = LightningRedis(base_path=DB_PATH, decode_responses=True)
print(f"Recovered 'foo' from disk: {r2.get('foo')}")
assert r2.get('foo') == 'bar'

print("\n✅ All Redis compatibility tests passed.")