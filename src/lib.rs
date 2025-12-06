use pyo3::prelude::*;
use pyo3::types::{PyList, PyBytes, PyNone};
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

#[pyclass]
struct RsLmdbStorage {
    shards: Vec<Arc<Environment>>,
    dbs: Vec<Database>,
    num_shards: usize,
}

#[pymethods]
impl RsLmdbStorage {
    #[new]
    #[pyo3(signature = (base_path, num_shards=5, map_size=1099511627776))] // Default ~1TB
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
                // Flags for speed: WRITE_MAP (direct mem writing), NO_SYNC (OS buffers io)
                .set_flags(lmdb::EnvironmentFlags::WRITE_MAP | lmdb::EnvironmentFlags::NO_SYNC) 
                .open(path)
                .map_err(StorageError::from)?;

            let db = env.open_db(None).map_err(StorageError::from)?;

            shards.push(Arc::new(env));
            dbs.push(db);
        }

        Ok(RsLmdbStorage { shards, dbs, num_shards })
    }

    /// Store Numpy Vectors (Zero-copy, Fast)
    fn store_vectors<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<'py, f32>, identifiers: Vec<i64>) -> PyResult<()> {
        let vectors = data.as_array();
        if vectors.shape()[0] != identifiers.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch"));
        }

        let dim = vectors.shape()[1];

        py.allow_threads(|| -> PyResult<()> {
            // Group data by shard to write in parallel
            let mut buckets: Vec<Vec<(i64, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();

            // CPU Bound: Copying data from Array View to Byte Vectors
            for (i, &id) in identifiers.iter().enumerate() {
                let shard_idx = (id.abs() as usize) % self.num_shards;
                
                // Unsafe: Reinterpreting f32 bytes as u8 bytes for storage
                let row = vectors.row(i);
                let byte_len = dim * 4;
                let mut byte_data = Vec::with_capacity(byte_len);
                unsafe {
                    let ptr = row.as_ptr() as *const u8;
                    let slice = std::slice::from_raw_parts(ptr, byte_len);
                    byte_data.extend_from_slice(slice);
                }
                buckets[shard_idx].push((id, byte_data));
            }

            // IO Bound: Parallel Write
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, batch)| {
                if batch.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let mut txn = env.begin_rw_txn().unwrap();
                
                for (id, val_bytes) in batch {
                    let mut key_bytes = [0u8; 8];
                    LittleEndian::write_i64(&mut key_bytes, id);
                    let _ = txn.put(db, &key_bytes, &val_bytes, WriteFlags::empty());
                }
                txn.commit().unwrap();
            });

            Ok(())
        })
    }

    /// Retrieve Numpy Vectors
    fn get_vectors<'py>(&self, py: Python<'py>, identifiers: Vec<i64>) -> PyResult<&'py PyList> {
        let num_items = identifiers.len();
        let results = Arc::new(Mutex::new(vec![None; num_items]));

        // Group requests
        let mut buckets: Vec<Vec<(usize, i64)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        for (i, &id) in identifiers.iter().enumerate() {
            let shard_idx = (id.abs() as usize) % self.num_shards;
            buckets[shard_idx].push((i, id));
        }

        // Parallel Read
        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();

                for (orig_idx, id) in reqs {
                    let mut key_bytes = [0u8; 8];
                    LittleEndian::write_i64(&mut key_bytes, id);
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

        // Convert to Python List
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

    /// Store arbitrary Python objects (Strings, Lists, etc.)
    /// Uses pickle internally.
    fn store_data<'py>(&self, py: Python<'py>, data: Vec<PyObject>, identifiers: Vec<i64>) -> PyResult<()> {
        if data.len() != identifiers.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Length mismatch"));
        }
        
        // Import pickle module once
        let pickle = PyModule::import(py, "pickle")?;
        let dumps = pickle.getattr("dumps")?;

        // 1. Serialize all objects in Python thread (GIL required)
        // Note: This is the bottleneck for generic objects, but required for pickle.
        let mut buckets: Vec<Vec<(i64, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (obj, &id) in data.iter().zip(identifiers.iter()) {
            let shard_idx = (id.abs() as usize) % self.num_shards;
            
            // Call pickle.dumps(obj)
            let bytes_obj = dumps.call1((obj,))?;
            let bytes: &[u8] = bytes_obj.extract::<&PyBytes>()?.as_bytes();
            
            buckets[shard_idx].push((id, bytes.to_vec()));
        }

        // 2. Parallel Write (No GIL)
        py.allow_threads(|| -> PyResult<()> {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, batch)| {
                if batch.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let mut txn = env.begin_rw_txn().unwrap();
                
                for (id, val_bytes) in batch {
                    let mut key_bytes = [0u8; 8];
                    LittleEndian::write_i64(&mut key_bytes, id);
                    let _ = txn.put(db, &key_bytes, &val_bytes, WriteFlags::empty());
                }
                txn.commit().unwrap();
            });
            Ok(())
        })?;

        Ok(())
    }

    /// Get arbitrary Python objects
    fn get_data<'py>(&self, py: Python<'py>, identifiers: Vec<i64>) -> PyResult<&'py PyList> {
        let num_items = identifiers.len();
        let results = Arc::new(Mutex::new(vec![None; num_items]));

        let mut buckets: Vec<Vec<(usize, i64)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        for (i, &id) in identifiers.iter().enumerate() {
            let shard_idx = (id.abs() as usize) % self.num_shards;
            buckets[shard_idx].push((i, id));
        }

        // 1. Parallel Read (Returns raw bytes)
        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();

                for (orig_idx, id) in reqs {
                    let mut key_bytes = [0u8; 8];
                    LittleEndian::write_i64(&mut key_bytes, id);
                    if let Ok(bytes) = txn.get(db, &key_bytes) {
                        let mut res_lock = results.lock().unwrap();
                        res_lock[orig_idx] = Some(bytes.to_vec());
                    }
                }
            });
        });

        // 2. Deserialize (Pickle loads) with GIL
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

    /// Delete data by identifiers
    fn delete_data<'py>(&self, py: Python<'py>, identifiers: Vec<i64>) -> PyResult<()> {
        let mut buckets: Vec<Vec<i64>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        for id in identifiers {
            let shard_idx = (id.abs() as usize) % self.num_shards;
            buckets[shard_idx].push(id);
        }

        py.allow_threads(|| -> PyResult<()> {
            buckets.into_par_iter().enumerate().for_each(|(shard_idx, ids)| {
                if ids.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let mut txn = env.begin_rw_txn().unwrap();
                
                for id in ids {
                    let mut key_bytes = [0u8; 8];
                    LittleEndian::write_i64(&mut key_bytes, id);
                    let _ = txn.del(db, &key_bytes, None);
                }
                txn.commit().unwrap();
            });
            Ok(())
        })
    }

    /// Return total number of entries across all shards
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