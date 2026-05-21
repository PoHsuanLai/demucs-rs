#!/bin/bash
# Build the Demucs WASM extension component.
#
# Requires: cargo, wasm-tools, wasm32-wasip1 target
#   rustup target add wasm32-wasip1
#   cargo install wasm-tools
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADAPTER_URL="https://github.com/bytecodealliance/wasmtime/releases/download/v39.0.2/wasi_snapshot_preview1.reactor.wasm"
ADAPTER="/tmp/wasi_reactor_v39.wasm"

# 1. Build core module targeting wasip1
echo "Building wasm32-wasip1..."
cargo build --manifest-path "$SCRIPT_DIR/Cargo.toml" --target wasm32-wasip1 --release

# 2. Download WASI adapter if missing
if [ ! -f "$ADAPTER" ]; then
    echo "Downloading WASI adapter..."
    curl -sL "$ADAPTER_URL" -o "$ADAPTER"
fi

# 3. Componentize with adapter.
# `cargo metadata` resolves the target dir wherever the user has it
# pointed (default $WORKSPACE/target or via CARGO_TARGET_DIR /
# `~/.cargo/config.toml`'s `[build] target-dir`).
TARGET_DIR="$(cargo metadata --manifest-path "$SCRIPT_DIR/Cargo.toml" --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
CORE="$TARGET_DIR/wasm32-wasip1/release/demucs_extension.wasm"
OUT="$SCRIPT_DIR/extension.wasm"
echo "Componentizing..."
wasm-tools component new "$CORE" --adapt "wasi_snapshot_preview1=$ADAPTER" -o "$OUT"

echo "Built: $OUT ($(du -h "$OUT" | cut -f1))"
