//! Compile the GLSL shaders in `shaders/` to SPIR-V via the system `glslc`
//! and drop them in `OUT_DIR` so the source can `include_bytes!` them.
//!
//! glslc ships with the Vulkan SDK / `shaderc` package — Arch's `shaderc`
//! provides it. If it's missing the build fails loudly with the install
//! hint baked into the panic message; we don't want a silent fallback to
//! stale SPIR-V because then a shader-source edit could go un-applied.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shaders_dir = manifest.join("shaders");
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    let entries = [
        ("quad.vert", "vertex"),
        ("quad.frag", "fragment"),
        ("solid.frag", "fragment"),
        ("blur.frag", "fragment"),
        ("backdrop.frag", "fragment"),
    ];

    for (name, stage) in entries {
        let src = shaders_dir.join(name);
        let dst = out_dir.join(format!("{name}.spv"));
        compile(stage, &src, &dst);
    }
}

fn compile(stage: &str, src: &Path, dst: &Path) {
    let stage_flag = format!("-fshader-stage={stage}");
    let status = Command::new("glslc")
        .arg(&stage_flag)
        .arg("--target-env=vulkan1.3")
        .arg("-O")
        .arg("-o")
        .arg(dst)
        .arg(src)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "failed to invoke glslc ({e}). \
                 Install the `shaderc` package (Arch: pacman -S shaderc; \
                 Debian/Ubuntu: apt install glslang-tools shaderc) and rebuild."
            )
        });
    if !status.success() {
        panic!(
            "glslc failed compiling {} (status={}); see preceding stderr",
            src.display(),
            status,
        );
    }
}
