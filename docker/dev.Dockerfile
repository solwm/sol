# Linux build/check environment for sol. The workspace links against
# libinput/libseat/libgbm etc., so it only compiles on Linux; this image
# lets `cargo check/test` run from any host via Docker.
#
#   docker build -t sol-dev -f docker/dev.Dockerfile docker
#   docker run --rm -v "$PWD":/work -v sol-cargo:/usr/local/cargo/registry \
#     -v sol-target:/target -e CARGO_TARGET_DIR=/target \
#     sol-dev cargo check --workspace --features debug-ctl
FROM rust:1-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libdrm-dev \
    libinput-dev \
    libseat-dev \
    libxkbcommon-dev \
    libgbm-dev \
    libvulkan-dev \
    libudev-dev \
    glslc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
