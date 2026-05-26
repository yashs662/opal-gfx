//! Shared scaffolding for the frostify-gfx examples.
//!
//! Cargo auto-discovers examples at `examples/<name>.rs` (files) or
//! `examples/<name>/main.rs` (subdirs with `main.rs`). A subdir without
//! `main.rs` — like this one — is invisible to the auto-discovery path
//! and is only pulled in by sibling examples via `mod common;`.
//!
//! Each submodule is `#![allow(dead_code)]` because any given example
//! only uses a subset of the helpers; the unused fns still need to
//! compile in the example's binary crate.

#![allow(dead_code)]

pub mod components;
pub mod image;
