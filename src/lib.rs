use pyo3::prelude::*;
use pyo3::types::{PyList, PyBytes, PyNone, PyString, PyDict, PyTuple};
use numpy::{PyReadonlyArray2, PyArray1};
use lmdb::{Environment, Transaction, WriteFlags, Database, Cursor};
use rayon::prelude::*;
use std::sync::{Arc};
use std::path::Path;
use std::fs;
use byteorder::{ByteOrder, LittleEndian};

// LMDB Constants
const MDB_NEXT: u32 = 8;
const MDB_SET_RANGE: u32 = 17;

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

    fn close(&self) -> PyResult<()> {
        Ok(())
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

        // 1. Process Keys
        let mut buckets: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
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

        // 2. Parallel Write
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
        
        let mut buckets: Vec<Vec<(usize, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        let mut num_items = 0;
        
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
            buckets[key_info.shard_idx].push((i, key_info.key_bytes));
            num_items += 1;
        }

        let mut results: Vec<Option<Vec<f32>>> = vec![None; num_items];
        let result_ptr_addr = results.as_mut_ptr() as usize;

        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(move |(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();
                let result_ptr = result_ptr_addr as *mut Option<Vec<f32>>;

                for (orig_idx, key_bytes) in reqs {
                    if let Ok(bytes) = txn.get(db, &key_bytes) {
                        let byte_len = bytes.len();
                        let float_count = byte_len / 4;
                        let mut vec_f32 = Vec::with_capacity(float_count);
                        unsafe {
                            let dst_ptr = vec_f32.as_mut_ptr() as *mut u8;
                            let src_ptr = bytes.as_ptr();
                            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, byte_len);
                            vec_f32.set_len(float_count);
                            std::ptr::write(result_ptr.add(orig_idx), Some(vec_f32));
                        }
                    }
                }
            });
        });

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
             if data.len() != len { return Err(pyo3::exceptions::PyValueError::new_err("Length mismatch")); }
        }
        
        let pickle = PyModule::import(py, "pickle")?;
        let dumps = pickle.getattr("dumps")?;

        let mut buckets: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        for (i, id_obj) in ids_iter.enumerate() {
            let id_obj = id_obj?;
            let obj = &data[i];
            let key_info = LDKV::process_key(id_obj, self.num_shards, i)?;
            let bytes_obj = dumps.call1((obj,))?;
            let val_bytes: &[u8] = bytes_obj.extract::<&PyBytes>()?.as_bytes();
            buckets[key_info.shard_idx].push((key_info.key_bytes, val_bytes.to_vec()));
        }

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

        let mut results: Vec<Option<Vec<u8>>> = vec![None; num_items];
        let result_ptr_addr = results.as_mut_ptr() as usize;

        py.allow_threads(|| {
            buckets.into_par_iter().enumerate().for_each(move |(shard_idx, reqs)| {
                if reqs.is_empty() { return; }
                let env = &self.shards[shard_idx];
                let db = self.dbs[shard_idx];
                let txn = env.begin_ro_txn().unwrap();
                let result_ptr = result_ptr_addr as *mut Option<Vec<u8>>;

                for (orig_idx, key_bytes) in reqs {
                    if let Ok(bytes) = txn.get(db, &key_bytes) {
                        unsafe { std::ptr::write(result_ptr.add(orig_idx), Some(bytes.to_vec())); }
                    }
                }
            });
        });

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
                None => { py_list.append(PyNone::get(py))?; }
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

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        // 1. Collect raw key bytes from all shards in parallel
        // We return Vec<Vec<u8>> instead of Vec<String> to defer type interpretation
        let mut all_keys_raw = py.allow_threads(move || -> Result<Vec<Vec<u8>>, StorageError> {
            let keys: Vec<Vec<u8>> = self.shards.par_iter().zip(&self.dbs).map(|(env, db)| {
                let txn = env.begin_ro_txn().map_err(StorageError::Lmdb)?;
                let mut cursor = txn.open_ro_cursor(*db).map_err(StorageError::Lmdb)?;
                let mut shard_keys = Vec::new();
                for (key_bytes, _) in cursor.iter() {
                    shard_keys.push(key_bytes.to_vec());
                }
                Ok(shard_keys)
            }).collect::<Result<Vec<Vec<Vec<u8>>>, StorageError>>()?
            .into_iter()
            .flatten()
            .collect();
            Ok(keys)
        })?;

        // 2. Sort by byte representation to ensure deterministic output order
        all_keys_raw.sort();

        // 3. Convert to Python objects (Int or String) based on byte length
        let py_list = PyList::empty(py);
        
        for key_bytes in all_keys_raw {
            // Heuristic: process_key writes i64 as exactly 8 bytes.
            // If the key is 8 bytes, we decode it as an integer.
            if key_bytes.len() == 8 {
                let val = LittleEndian::read_i64(&key_bytes);
                py_list.append(val)?;
            } else {
                // Otherwise, try to decode as UTF-8 string
                if let Ok(key_str) = std::str::from_utf8(&key_bytes) {
                    py_list.append(key_str)?;
                }
                // Keys that are not 8 bytes and not valid UTF-8 are skipped
            }
        }
        
        Ok(py_list)
    }

    fn keys_with_prefix<'py>(&self, py: Python<'py>, prefix: String) -> PyResult<Vec<String>> {
        let prefix_clone = prefix.clone();
        
        let mut found_keys = py.allow_threads(move || -> Result<Vec<String>, StorageError> {
            let prefix_bytes = prefix_clone.as_bytes();
            
            let keys: Vec<String> = self.shards.par_iter().zip(&self.dbs).map(|(env, db)| {
                let txn = env.begin_ro_txn().map_err(StorageError::Lmdb)?;
                let mut cursor = txn.open_ro_cursor(*db).map_err(StorageError::Lmdb)?;
                let mut shard_results = Vec::new();
                
                // Use pattern matching to unwrap Option<Key>
                let start_res = cursor.get(Some(prefix_bytes), None, MDB_SET_RANGE);
                
                if let Ok((Some(k_bytes), _v)) = start_res {
                    let mut k = k_bytes;
                    loop {
                        if let Ok(key_str) = std::str::from_utf8(k) {
                            if key_str.starts_with(&prefix_clone) {
                                shard_results.push(key_str.to_string());
                            } else {
                                break;
                            }
                        }

                        match cursor.get(None, None, MDB_NEXT) {
                            Ok((Some(next_k), _)) => k = next_k,
                            _ => break,
                        }
                    }
                }
                
                Ok(shard_results)
            }).collect::<Result<Vec<Vec<String>>, StorageError>>()?
            .into_iter()
            .flatten()
            .collect();
            
            Ok(keys)
        })?;

        found_keys.sort();
        Ok(found_keys)
    }

    fn items_in_range<'py>(
        &self, 
        py: Python<'py>, 
        start_key: String, 
        end_key: String, 
        prefix: String
    ) -> PyResult<&'py PyList> {
        let start_key_c = start_key.clone();
        let end_key_c = end_key.clone();
        let prefix_c = prefix.clone();

        let mut raw_items = py.allow_threads(move || -> Result<Vec<(String, Vec<u8>)>, StorageError> {
            let start_bytes = start_key_c.as_bytes();
            let end_bytes = end_key_c.as_bytes();
            
            let items: Vec<(String, Vec<u8>)> = self.shards.par_iter().zip(&self.dbs).map(|(env, db)| {
                let txn = env.begin_ro_txn().map_err(StorageError::Lmdb)?;
                let mut cursor = txn.open_ro_cursor(*db).map_err(StorageError::Lmdb)?;
                let mut shard_results = Vec::new();
                
                // Use pattern matching to unwrap Option<Key>
                let start_res = cursor.get(Some(start_bytes), None, MDB_SET_RANGE);

                if let Ok((Some(k_bytes), v_bytes)) = start_res {
                    let mut k = k_bytes;
                    let mut v = v_bytes;
                    loop {
                        // Compare slices directly
                        if k >= end_bytes {
                            break;
                        }

                        if let Ok(key_str) = std::str::from_utf8(k) {
                            if key_str.starts_with(&prefix_c) {
                                shard_results.push((key_str.to_string(), v.to_vec()));
                            }
                        }

                        match cursor.get(None, None, MDB_NEXT) {
                            Ok((Some(next_k), next_v)) => {
                                k = next_k;
                                v = next_v;
                            },
                            _ => break,
                        }
                    }
                }

                Ok(shard_results)
            }).collect::<Result<Vec<Vec<(String, Vec<u8>)>>, StorageError>>()?
            .into_iter()
            .flatten()
            .collect();
            
            Ok(items)
        })?;

        raw_items.sort_by(|a, b| a.0.cmp(&b.0));

        let pickle = PyModule::import(py, "pickle")?;
        let loads = pickle.getattr("loads")?;
        let py_list = PyList::empty(py);

        for (k, v_bytes) in raw_items {
            let py_bytes = PyBytes::new(py, &v_bytes);
            let val_obj = loads.call1((py_bytes,))?;
            let tuple = PyTuple::new(py, &[k.into_py(py), val_obj.into()]);
            py_list.append(tuple)?;
        }

        Ok(py_list)
    }

    fn batch_atomic_update<'py>(
        &self, 
        py: Python<'py>, 
        updates: &PyDict, 
        merge_func: &PyAny
    ) -> PyResult<()> {
        let pickle = PyModule::import(py, "pickle")?;
        let dumps = pickle.getattr("dumps")?;
        let loads = pickle.getattr("loads")?;

        let mut buckets: Vec<Vec<(PyObject, PyObject, Vec<u8>)>> = (0..self.num_shards)
            .map(|_| Vec::new())
            .collect();

        for (key_obj, new_data) in updates.iter() {
            let key_info = LDKV::process_key(key_obj, self.num_shards, 0)?;
            buckets[key_info.shard_idx].push((
                key_obj.to_object(py), 
                new_data.to_object(py), 
                key_info.key_bytes
            ));
        }

        for (shard_idx, batch) in buckets.into_iter().enumerate() {
            if batch.is_empty() { continue; }

            let env = &self.shards[shard_idx];
            let db = self.dbs[shard_idx];
            let mut txn = env.begin_rw_txn().map_err(StorageError::Lmdb)?;

            for (_key_obj, new_data_obj, key_bytes) in batch {
                let existing_val = match txn.get(db, &key_bytes) {
                    Ok(bytes) => {
                        let py_bytes = PyBytes::new(py, bytes);
                        let obj = loads.call1((py_bytes,))?;
                        Some(obj)
                    },
                    Err(lmdb::Error::NotFound) => None,
                    Err(e) => return Err(StorageError::Lmdb(e).into()),
                };

                let old_arg = match existing_val {
                    Some(obj) => obj.to_object(py),
                    None => PyNone::get(py).to_object(py),
                };

                let merged_obj = merge_func.call1((old_arg, new_data_obj))?;
                let dumped = dumps.call1((merged_obj,))?;
                let final_bytes = dumped.extract::<&PyBytes>()?.as_bytes();

                txn.put(db, &key_bytes, &final_bytes, WriteFlags::empty())
                    .map_err(StorageError::Lmdb)?;
            }
            txn.commit().map_err(StorageError::Lmdb)?;
        }

        Ok(())
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