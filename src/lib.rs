use pyo3::prelude::*;
use pyo3::types::{PyList, PyBytes, PyNone, PyString};
use numpy::{PyReadonlyArray2, PyArray1};
use lmdb::{Environment, Transaction, WriteFlags, Database};
use rayon::prelude::*;
use std::sync::{Arc};
use std::path::Path;
use std::fs;
use byteorder::{ByteOrder, LittleEndian};

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
struct LDKV {
    shards: Vec<Arc<Environment>>,
    dbs: Vec<Database>,
    num_shards: usize,
}

impl LDKV {
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
            let digest = md5::compute(bytes);
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
impl LDKV {
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

        Ok(LDKV { shards, dbs, num_shards })
    }

    fn store_vectors<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<'py, f32>, identifiers: &PyAny) -> PyResult<()> {
        let vectors = data.as_array();
        let ids_iter = identifiers.iter()?;
        
        if let Ok(len) = identifiers.len() {
            if vectors.shape()[0] != len {
                 return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch between data and identifiers"));
            }
        }

        let dim = vectors.shape()[1];

        // 1. Process Keys & Copy Data (GIL required)
        let mut buckets: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
            
            let row = vectors.row(i);
            let byte_len = dim * 4;
            
            // Optimization: Create vector with exact capacity and copy directly
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
                
                let mut txn = env.begin_rw_txn().expect("Failed to begin transaction");
                
                for (key_bytes, val_bytes) in batch {
                    let _ = txn.put(db, &key_bytes, &val_bytes, WriteFlags::empty());
                }
                txn.commit().expect("Failed to commit");
            });
            Ok(())
        })
    }

    fn get_vectors<'py>(&self, py: Python<'py>, identifiers: &PyAny) -> PyResult<&'py PyList> {
        let ids_iter = identifiers.iter()?;
        
        // 1. Bucket keys
        let mut buckets: Vec<Vec<(usize, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        let mut num_items = 0;
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
            buckets[key_info.shard_idx].push((i, key_info.key_bytes));
            num_items += 1;
        }

        // Pre-allocate results with None.
        let mut results: Vec<Option<Vec<f32>>> = vec![None; num_items];
        
        // POINTER SMUGGLING:
        // Cast the raw pointer to usize (integer). Integers are Send+Sync.
        let result_ptr_addr = results.as_mut_ptr() as usize;

        // 2. Parallel Read (No Mutex, No GIL)
        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(move |(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();

                // Reconstruct pointer inside thread
                let result_ptr = result_ptr_addr as *mut Option<Vec<f32>>;

                for (orig_idx, key_bytes) in reqs {
                    if let Ok(bytes) = txn.get(db, &key_bytes) {
                        let byte_len = bytes.len();
                        let float_count = byte_len / 4;
                        
                        // Optimized copy that handles unaligned memory safely
                        let mut vec_f32 = Vec::with_capacity(float_count);
                        unsafe {
                            let dst_ptr = vec_f32.as_mut_ptr() as *mut u8;
                            let src_ptr = bytes.as_ptr();
                            // memcpy is robust against unaligned source pointers from LMDB
                            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, byte_len);
                            vec_f32.set_len(float_count);
                            
                            // SAFETY: orig_idx is guaranteed unique by the single-threaded bucketing step
                            std::ptr::write(result_ptr.add(orig_idx), Some(vec_f32));
                        }
                    }
                }
            });
        });

        // 3. Convert to Python Objects
        let py_list = PyList::empty(py);
        for opt in results {
            match opt {
                Some(vec) => {
                    let arr = PyArray1::from_vec(py, vec);
                    py_list.append(arr)?;
                },
                None => py_list.append(PyNone::get(py))?,
            }
        }
        Ok(py_list)
    }

    fn store_data<'py>(&self, py: Python<'py>, data: Vec<PyObject>, identifiers: &PyAny) -> PyResult<()> {
        let ids_iter = identifiers.iter()?;
        
        if let Ok(len) = identifiers.len() {
             if data.len() != len {
                return Err(pyo3::exceptions::PyValueError::new_err("Length mismatch"));
            }
        }
        
        let pickle = PyModule::import(py, "pickle")?;
        let dumps = pickle.getattr("dumps")?;

        // 1. Serialize and Key Processing (GIL required)
        let mut buckets: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let obj = &data[i];
            
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
            
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
        
        let mut buckets: Vec<Vec<(usize, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        let mut num_items = 0;

        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
            buckets[key_info.shard_idx].push((i, key_info.key_bytes));
            num_items += 1;
        }

        // Pre-allocate results
        let mut results: Vec<Option<Vec<u8>>> = vec![None; num_items];
        
        // POINTER SMUGGLING:
        let result_ptr_addr = results.as_mut_ptr() as usize;

        // 1. Parallel Read
        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(move |(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();
                
                let result_ptr = result_ptr_addr as *mut Option<Vec<u8>>;

                for (orig_idx, key_bytes) in reqs {
                    if let Ok(bytes) = txn.get(db, &key_bytes) {
                        unsafe {
                            std::ptr::write(result_ptr.add(orig_idx), Some(bytes.to_vec()));
                        }
                    }
                }
            });
        });

        // 2. Deserialize (Pickle loads)
        let pickle = PyModule::import(py, "pickle")?;
        let loads = pickle.getattr("loads")?;
        
        let py_list = PyList::empty(py);

        for opt in results {
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
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
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
    m.add_class::<LDKV>()?;
    Ok(())
}