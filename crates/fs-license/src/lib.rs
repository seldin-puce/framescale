//! Ed25519 license verification and entitlement resolution.
//!
//! Linked into BOTH the app core and fs-worker. The worker verifies the token
//! independently before it enables any Pro stage or codec, so patching the UI
//! or the host does not unlock features. Only the public key ships.

#![forbid(unsafe_code)]
