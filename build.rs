fn main() {
    // If we are on Windows, link the Advapi32 library required by LMDB
    #[cfg(windows)]
    println!("cargo:rustc-link-lib=dylib=Advapi32");
}