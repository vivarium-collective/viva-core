//! Build script: link GLPK only when the `glpk` cargo feature is enabled, so the
//! default wheel build has no GLPK dependency. GLPK lives at /opt/homebrew by
//! default (Homebrew on Apple Silicon); override the lib dir with GLPK_LIB_DIR.

fn main() {
    // Cargo sets CARGO_FEATURE_<NAME> for each enabled feature.
    if std::env::var_os("CARGO_FEATURE_GLPK").is_some() {
        let lib_dir =
            std::env::var("GLPK_LIB_DIR").unwrap_or_else(|_| "/opt/homebrew/lib".to_string());
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=dylib=glpk");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GLPK");
    println!("cargo:rerun-if-env-changed=GLPK_LIB_DIR");
}
