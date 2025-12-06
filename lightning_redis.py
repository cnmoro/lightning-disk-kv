import time
import os
import shutil
from typing import Union, Optional, Any
from lightning_disk_kv import LDKV

class LDKV_RedisCompat:
    """
    A drop-in replacement for redis.Redis that uses lightning-disk-kv 
    (LMDB sharded backend) for storage.
    
    Features:
    - Persists to disk automatically.
    - Supports TTL (expiry).
    - Thread-safe (GIL released during disk I/O).
    - Mimics redis-py API (get, set, mget, delete, hget, etc).
    """

    def __init__(self, 
                 base_path: str = "./redis_data", 
                 num_shards: int = 16, 
                 map_size: int = 100 * 1024**3,
                 decode_responses: bool = False,
                 **kwargs):
        """
        Args:
            base_path: Directory to store DB files (replaces host/port).
            num_shards: Parallelism level.
            map_size: Max database size in bytes (virtual memory).
            decode_responses: If True, returns strings. If False, returns bytes.
        """
        self.base_path = base_path
        self.decode_responses = decode_responses
        
        # Initialize the Rust backend
        self._db = LDKV(base_path, num_shards, map_size)

    def _encode_key(self, key: Any) -> str:
        """LDKV requires keys to be int or str. Redis uses bytes usually."""
        if isinstance(key, int):
            return key
        if isinstance(key, bytes):
            # We use latin1 to map bytes 1-to-1 to unicode code points
            # ensuring we can support arbitrary binary keys
            return key.decode('latin1')
        return str(key)

    def _encode_val(self, val: Any) -> Any:
        """Redis stores bytes. We accept bytes/str/int/float and store as is."""
        if isinstance(val, (int, float)):
            return str(val).encode('utf-8')
        if isinstance(val, str):
            return val.encode('utf-8')
        return val

    def _decode_val(self, val: Any) -> Union[str, bytes, None]:
        if val is None:
            return None
        if self.decode_responses and isinstance(val, bytes):
            return val.decode('utf-8')
        return val

    def _pack(self, value, ex: Optional[int] = None):
        """Wraps value with expiry metadata."""
        expiry = time.time() + ex if ex else None
        return {'d': self._encode_val(value), 'x': expiry}

    def _unpack(self, key, payload):
        """Unwraps value and checks expiry."""
        if payload is None:
            return None
        
        # Check expiry
        if payload['x'] and time.time() > payload['x']:
            self._db.delete_data([key]) # Lazy delete
            return None
            
        return self._decode_val(payload['d'])

    # ------------------------------------------------------------------
    # Basic K/V Operations
    # ------------------------------------------------------------------

    def set(self, name, value, ex=None, px=None, nx=False, xx=False):
        """
        Set the value at key ``name`` to ``value``.
        """
        key = self._encode_key(name)
        
        # Handle expiration math
        expiry_seconds = ex
        if px:
            expiry_seconds = px / 1000.0
            
        # Handle NX/XX constraints
        if nx or xx:
            exists = self.exists(name)
            if nx and exists: return None
            if xx and not exists: return None

        payload = self._pack(value, expiry_seconds)
        self._db.store_data([payload], [key])
        return True

    def get(self, name):
        """Return the value at key ``name``, or None if the key doesn't exist."""
        key = self._encode_key(name)
        payload = self._db.get_data([key])[0]
        return self._unpack(key, payload)

    def mget(self, keys, *args):
        """Returns a list of values ordered identically to ``keys``."""
        if args:
            keys = list(keys) + list(args)
        
        encoded_keys = [self._encode_key(k) for k in keys]
        payloads = self._db.get_data(encoded_keys)
        
        return [self._unpack(k, p) for k, p in zip(encoded_keys, payloads)]

    def mset(self, mapping):
        """Sets key/values based on a mapping. mapping is a dict of key-value pairs."""
        keys = []
        payloads = []
        for k, v in mapping.items():
            keys.append(self._encode_key(k))
            payloads.append(self._pack(v))
        
        self._db.store_data(payloads, keys)
        return True

    def delete(self, *names):
        """Delete one or more keys specified by ``names``."""
        keys = [self._encode_key(n) for n in names]
        self._db.delete_data(keys)
        return len(keys)

    def exists(self, *names):
        """Returns the number of ``names`` that exist."""
        count = 0
        keys = [self._encode_key(n) for n in names]
        payloads = self._db.get_data(keys)
        for k, p in zip(keys, payloads):
            if self._unpack(k, p) is not None:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def incr(self, name, amount=1):
        """Increments the value of key ``name`` by ``amount``."""
        current = self.get(name)
        if current is None:
            val = 0
        else:
            try:
                val = int(current)
            except ValueError:
                raise TypeError("value is not an integer or out of range")
        
        val += amount
        self.set(name, val)
        return val

    def decr(self, name, amount=1):
        return self.incr(name, -amount)

    # ------------------------------------------------------------------
    # Hashes (Simulated via Dicts)
    # ------------------------------------------------------------------

    def hset(self, name, key=None, value=None, mapping=None):
        """Set key to value within hash ``name``."""
        db_key = self._encode_key(name)
        
        existing = self._db.get_data([db_key])[0]
        data = self._unpack(db_key, existing) if existing else {}
        
        if not isinstance(data, dict):
            if data is None: data = {}
            else: raise TypeError("WRONGTYPE Operation against a key holding the wrong kind of value")

        updated_count = 0
        
        if mapping:
            for k, v in mapping.items():
                data[k] = self._encode_val(v)
                updated_count += 1
        if key is not None and value is not None:
            data[key] = self._encode_val(value)
            updated_count += 1
            
        payload = self._pack(data)
        self._db.store_data([payload], [db_key])
        return updated_count

    def hget(self, name, key):
        """Return the value of ``key`` within the hash ``name``."""
        all_data = self.hgetall(name)
        return all_data.get(self._encode_val(key)) if all_data else None

    def hgetall(self, name):
        """Return a Python dict of the hash's name/value pairs."""
        val = self.get(name)
        if val is None: return {}
        if not isinstance(val, dict):
             raise TypeError("WRONGTYPE Operation against a key holding the wrong kind of value")
        
        if self.decode_responses:
            return {
                (k.decode('utf-8') if isinstance(k, bytes) else k): 
                (v.decode('utf-8') if isinstance(v, bytes) else v)
                for k, v in val.items()
            }
        return val

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def flushdb(self):
        """Delete all keys in the current database."""
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        self._db = LDKV(self.base_path, self._db.num_shards)
        return True

    def save(self):
        """Force sync to disk."""
        self._db.sync()
        return True
    
    def dbsize(self):
        return self._db.get_data_count()

    def close(self):
        self._db.sync()