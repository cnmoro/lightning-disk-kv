use pyo3::prelude::*;
use pyo3::types::{PyList, PyBytes, PyNone, PyString};
use numpy::{PyReadonlyArray2, ToPyArray};
use lmdb::{Environment, Transaction, WriteFlags, Database};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::path::Path;
use std::fs;
use byteorder::{ByteOrder, LittleEndian};

/// Custom error handling
#[derive(Debug)]
enum StorageError {
    Lmdb(lmdb::Error),
    Io(std::io::Error),
    Py(PyErr),
}

impl From<lmdb::Error> for StorageError {
    fn from(err: lmdb::Error) -> Self { StorageError::Lmdb(err) }
}
impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self { StorageError::Io(err) }
}
impl From<PyErr> for StorageError {
    fn from(err: PyErr) -> Self { StorageError::Py(err) }
}
impl From<StorageError> for PyErr {
    fn from(err: StorageError) -> PyErr {
        match err {
            StorageError::Lmdb(e) => pyo3::exceptions::PyIOError::new_err(format!("LMDB Error: {}", e)),
            StorageError::Io(e) => pyo3::exceptions::PyIOError::new_err(format!("IO Error: {}", e)),
            StorageError::Py(e) => e,
        }
    }
}

/// Helper struct to hold processed key information decoupled from Python objects
struct PreparedKey {
    shard_idx: usize,
    key_bytes: Vec<u8>
}

#[pyclass]
struct RsLmdbStorage {
    shards: Vec<Arc<Environment>>,
    dbs: Vec<Database>,
    num_shards: usize,
}

impl RsLmdbStorage {
    /// Helper to process a Python identifier (int or str) into shard index and byte key.
    /// Matches the Python logic:
    /// - Int: Little Endian 64-bit, Shard = abs(id) % N
    /// - Str: UTF-8 Bytes, Shard = int(md5(str), 16) % N
    fn process_key(obj: &PyAny, num_shards: usize, _idx: usize) -> PyResult<PreparedKey> {
        if let Ok(val) = obj.extract::<i64>() {
            let shard_idx = (val.abs() as usize) % num_shards;
            let mut key_bytes = [0u8; 8];
            LittleEndian::write_i64(&mut key_bytes, val);
            Ok(PreparedKey {
                shard_idx,
                key_bytes: key_bytes.to_vec()
            })
        } else if let Ok(val) = obj.downcast::<PyString>() {
            let s = val.to_string_lossy();
            let bytes = s.as_bytes();
            
            // Replicate Python: int(hashlib.md5(str).hexdigest(), 16) % num_shards
            let digest = md5::compute(bytes);
            // Treat the 16-byte MD5 digest as a u128 (Big Endian per hex string standard)
            let hash_int = u128::from_be_bytes(digest.0);
            let shard_idx = (hash_int % (num_shards as u128)) as usize;

            Ok(PreparedKey {
                shard_idx,
                key_bytes: bytes.to_vec()
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Unsupported key type. Must be int or str."))
        }
    }
}

#[pymethods]
impl RsLmdbStorage {
    #[new]
    #[pyo3(signature = (base_path, num_shards=5, map_size=1099511627776))]
    fn new(base_path: String, num_shards: usize, map_size: usize) -> PyResult<Self> {
        let mut shards = Vec::with_capacity(num_shards);
        let mut dbs = Vec::with_capacity(num_shards);

        for i in 0..num_shards {
            let path_str = format!("{}/shard_{}", base_path, i);
            let path = Path::new(&path_str);
            fs::create_dir_all(path)?;

            let env = Environment::new()
                .set_map_size(map_size)
                .set_max_dbs(1)
                .set_flags(lmdb::EnvironmentFlags::WRITE_MAP | lmdb::EnvironmentFlags::NO_SYNC) 
                .open(path)
                .map_err(StorageError::from)?;

            let db = env.open_db(None).map_err(StorageError::from)?;

            shards.push(Arc::new(env));
            dbs.push(db);
        }

        Ok(RsLmdbStorage { shards, dbs, num_shards })
    }

    fn store_vectors<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<'py, f32>, identifiers: &PyAny) -> PyResult<()> {
        let vectors = data.as_array();
        let ids_iter = identifiers.iter()?;
        
        // Ensure lengths match (approximate check, exact check happens in iteration)
        if let Ok(len) = identifiers.len() {
            if vectors.shape()[0] != len {
                 return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch between data and identifiers"));
            }
        }

        let dim = vectors.shape()[1];

        // 1. Process Keys (GIL required)
        // We group data into buckets: bucket[shard_index] -> Vec<(key_bytes, vector_bytes)>
        let mut buckets: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = RsLmdbStorage::process_key(id_obj, self.num_shards, i)?;
            
            // Extract vector data bytes
            let row = vectors.row(i);
            let byte_len = dim * 4;
            let mut val_bytes = Vec::with_capacity(byte_len);
            unsafe {
                let ptr = row.as_ptr() as *const u8;
                let slice = std::slice::from_raw_parts(ptr, byte_len);
                val_bytes.extend_from_slice(slice);
            }

            buckets[key_info.shard_idx].push((key_info.key_bytes, val_bytes));
        }

        // 2. Write to LMDB (Parallel, No GIL)
        py.allow_threads(|| -> PyResult<()> {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, batch)| {
                if batch.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let mut txn = env.begin_rw_txn().unwrap();
                
                for (key_bytes, val_bytes) in batch {
                    let _ = txn.put(db, &key_bytes, &val_bytes, WriteFlags::empty());
                }
                txn.commit().unwrap();
            });
            Ok(())
        })
    }

    fn get_vectors<'py>(&self, py: Python<'py>, identifiers: &PyAny) -> PyResult<&'py PyList> {
        let ids_iter = identifiers.iter()?;
        let mut num_items = 0;

        // 1. Process Keys (GIL required)
        let mut buckets: Vec<Vec<(usize, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = RsLmdbStorage::process_key(id_obj, self.num_shards, i)?;
            buckets[key_info.shard_idx].push((i, key_info.key_bytes));
            num_items += 1;
        }

        let results = Arc::new(Mutex::new(vec![None; num_items]));

        // 2. Parallel Read
        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();

                for (orig_idx, key_bytes) in reqs {
                    if let Ok(bytes) = txn.get(db, &key_bytes) {
                        let float_count = bytes.len() / 4;
                        let mut vec_f32 = Vec::with_capacity(float_count);
                        unsafe {
                            let ptr = bytes.as_ptr() as *const f32;
                            let slice = std::slice::from_raw_parts(ptr, float_count);
                            vec_f32.extend_from_slice(slice);
                        }
                        let mut res_lock = results.lock().unwrap();
                        res_lock[orig_idx] = Some(vec_f32);
                    }
                }
            });
        });

        // 3. Convert to Python
        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        let py_list = PyList::empty(py);
        for opt in final_results {
            match opt {
                Some(vec) => py_list.append(vec.to_pyarray(py))?,
                None => py_list.append(PyNone::get(py))?,
            }
        }
        Ok(py_list)
    }

    fn store_data<'py>(&self, py: Python<'py>, data: Vec<PyObject>, identifiers: &PyAny) -> PyResult<()> {
        let ids_iter = identifiers.iter()?;
        let num_ids = identifiers.len()?;
        
        if data.len() != num_ids {
            return Err(pyo3::exceptions::PyValueError::new_err("Length mismatch"));
        }
        
        let pickle = PyModule::import(py, "pickle")?;
        let dumps = pickle.getattr("dumps")?;

        // 1. Serialize and Key Processing (GIL required)
        let mut buckets: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let obj = &data[i];
            
            let key_info = RsLmdbStorage::process_key(id_obj, self.num_shards, i)?;
            
            // Pickle dump
            let bytes_obj = dumps.call1((obj,))?;
            let val_bytes: &[u8] = bytes_obj.extract::<&PyBytes>()?.as_bytes();
            
            buckets[key_info.shard_idx].push((key_info.key_bytes, val_bytes.to_vec()));
        }

        // 2. Parallel Write
        py.allow_threads(|| -> PyResult<()> {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, batch)| {
                if batch.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let mut txn = env.begin_rw_txn().unwrap();
                
                for (key_bytes, val_bytes) in batch {
                    let _ = txn.put(db, &key_bytes, &val_bytes, WriteFlags::empty());
                }
                txn.commit().unwrap();
            });
            Ok(())
        })?;

        Ok(())
    }

    fn get_data<'py>(&self, py: Python<'py>, identifiers: &PyAny) -> PyResult<&'py PyList> {
        let ids_iter = identifiers.iter()?;
        let mut num_items = 0;

        let mut buckets: Vec<Vec<(usize, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = RsLmdbStorage::process_key(id_obj, self.num_shards, i)?;
            buckets[key_info.shard_idx].push((i, key_info.key_bytes));
            num_items += 1;
        }

        let results = Arc::new(Mutex::new(vec![None; num_items]));

        // 1. Parallel Read
        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();

                for (orig_idx, key_bytes) in reqs {
                    if let Ok(bytes) = txn.get(db, &key_bytes) {
                        let mut res_lock = results.lock().unwrap();
                        res_lock[orig_idx] = Some(bytes.to_vec());
                    }
                }
            });
        });

        // 2. Deserialize (Pickle loads)
        let pickle = PyModule::import(py, "pickle")?;
        let loads = pickle.getattr("loads")?;
        
        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        let py_list = PyList::empty(py);

        for opt in final_results {
            match opt {
                Some(bytes_vec) => {
                    let py_bytes = PyBytes::new(py, &bytes_vec);
                    let obj = loads.call1((py_bytes,))?;
                    py_list.append(obj)?;
                },
                None => {
                    py_list.append(PyNone::get(py))?;
                }
            }
        }
        Ok(py_list)
    }

    fn delete_data<'py>(&self, py: Python<'py>, identifiers: &PyAny) -> PyResult<()> {
        let ids_iter = identifiers.iter()?;
        
        let mut buckets: Vec<Vec<Vec<u8>>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = RsLmdbStorage::process_key(id_obj, self.num_shards, i)?;
            buckets[key_info.shard_idx].push(key_info.key_bytes);
        }

        py.allow_threads(|| -> PyResult<()> {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, keys)| {
                if keys.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let mut txn = env.begin_rw_txn().unwrap();
                
                for key_bytes in keys {
                    let _ = txn.del(db, &key_bytes, None);
                }
                txn.commit().unwrap();
            });
            Ok(())
        })
    }

    fn get_data_count(&self) -> PyResult<usize> {
        let total: usize = self.shards.par_iter().map(|env| {
            let stat = env.stat().unwrap();
            stat.entries()
        }).sum();
        Ok(total)
    }

    fn sync(&self) -> PyResult<()> {
        for env in &self.shards {
            let _ = env.sync(true);
        }
        Ok(())
    }
}

#[pymodule]
fn lightning_disk_kv(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RsLmdbStorage>()?;
    Ok(())
}