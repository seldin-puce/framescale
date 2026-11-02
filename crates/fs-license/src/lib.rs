//! Ed25519 license verification and entitlement resolution.
//!
//! Token: `base64url(payload_json) "." base64url(ed25519_signature)`.
//! Only the 32-byte public key ships. It's compiled in from
//! `FS_LICENSE_PUBKEY` (64 hex chars) at build time; the release workflow sets
//! it. Without it, builds use a published development key, which signs
//! nothing a customer could buy. `is_dev_key()` lets the app warn loudly.
//!
//! This crate is linked into BOTH the app core and fs-worker. The worker
//! verifies independently and refuses Pro stages it isn't entitled to, so
//! patching the UI or the host doesn't unlock anything. Honest limit: a
//! determined attacker can patch any local binary. The goal is to make casual
//! piracy cost more than a license, not to build DRM.
#![forbid(unsafe_code)]

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use ed25519_dalek::{Signature, VerifyingKey};
use fs_types::license::{Entitlements, Feature, Tier};
use fs_types::pipeline::StageConfig;
use fs_types::worker::RenderPlan;
use serde::{Deserialize, Serialize};

pub const TOKEN_VERSION: u32 = 1;

/// Seed of the *development* signing key. Public on purpose: dev builds must
/// be able to mint test licenses, and a dev-key build is flagged at runtime.
pub const DEV_SEED: [u8; 32] = *b"framescale-dev-license-seed-v1!!";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Payload {
    pub v: u32,
    /// License id (for revocation lists and support).
    pub lid: String,
    pub tier: Tier,
    pub features: Vec<Feature>,
    pub licensee: Option<String>,
    /// Issued-at, unix seconds.
    pub iat: u64,
    /// Expiry (subscriptions), unix seconds. `None` = perpetual.
    pub exp: Option<u64>,
    /// Optional device binding: hash of the activation's machine fingerprint.
    pub device: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LicenseError {
    #[error("license token is malformed")]
    Malformed,
    #[error("license signature is invalid")]
    BadSignature,
    #[error("license token version {0} is not supported")]
    Version(u32),
    #[error("license expired")]
    Expired,
    #[error("license is bound to a different device")]
    WrongDevice,
}

#[derive(Debug, Clone)]
pub struct Verifier {
    key: VerifyingKey,
    dev: bool,
}

fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

fn dev_verifying_key() -> VerifyingKey {
    ed25519_dalek::SigningKey::from_bytes(&DEV_SEED).verifying_key()
}

impl Verifier {
    /// The key compiled into this build.
    pub fn embedded() -> Self {
        match option_env!("FS_LICENSE_PUBKEY").and_then(parse_hex32) {
            Some(bytes) => match VerifyingKey::from_bytes(&bytes) {
                Ok(key) => Self { key, dev: false },
                Err(_) => Self {
                    key: dev_verifying_key(),
                    dev: true,
                },
            },
            None => Self {
                key: dev_verifying_key(),
                dev: true,
            },
        }
    }

    pub fn with_key(key: VerifyingKey) -> Self {
        Self { key, dev: false }
    }

    pub fn is_dev_key(&self) -> bool {
        self.dev
    }

    /// Verify `token` at time `now` (unix seconds). `device` is this machine's
    /// fingerprint hash, checked only if the token is device-bound.
    pub fn verify(
        &self,
        token: &str,
        now: u64,
        device: Option<&str>,
    ) -> Result<Entitlements, LicenseError> {
        let (p, s) = token
            .trim()
            .split_once('.')
            .ok_or(LicenseError::Malformed)?;
        let payload_bytes = B64.decode(p).map_err(|_| LicenseError::Malformed)?;
        let sig_bytes = B64.decode(s).map_err(|_| LicenseError::Malformed)?;
        let sig = Signature::from_slice(&sig_bytes).map_err(|_| LicenseError::Malformed)?;
        // Verify the exact bytes we were given, before parsing them.
        self.key
            .verify_strict(&payload_bytes, &sig)
            .map_err(|_| LicenseError::BadSignature)?;
        let payload: Payload =
            serde_json::from_slice(&payload_bytes).map_err(|_| LicenseError::Malformed)?;
        if payload.v != TOKEN_VERSION {
            return Err(LicenseError::Version(payload.v));
        }
        if payload.exp.is_some_and(|exp| now > exp) {
            return Err(LicenseError::Expired);
        }
        if let Some(bound) = &payload.device
            && device != Some(bound.as_str())
        {
            return Err(LicenseError::WrongDevice);
        }
        let features = match payload.tier {
            // Pro always includes everything; `features` lets us sell add-ons later.
            Tier::Pro => Feature::ALL.to_vec(),
            Tier::Free => payload.features,
        };
        Ok(Entitlements {
            tier: payload.tier,
            features,
            licensee: payload.licensee,
            expires_at: payload.exp,
        })
    }

    /// Entitlements for an optional token: missing or invalid → Free.
    pub fn resolve(
        &self,
        token: Option<&str>,
        now: u64,
        device: Option<&str>,
    ) -> (Entitlements, Option<LicenseError>) {
        match token.map(|t| self.verify(t, now, device)) {
            Some(Ok(e)) => (e, None),
            Some(Err(err)) => (Entitlements::free(), Some(err)),
            None => (Entitlements::free(), None),
        }
    }
}

/// Pro features a compiled plan needs. This is the worker's authoritative gate,
/// derived from what it will *actually execute*, not from what the UI claims.
pub fn required_features(plan: &RenderPlan) -> Vec<Feature> {
    let mut need = Vec::new();
    for s in &plan.engine.stages {
        match s {
            StageConfig::FaceFix { .. } => need.push(Feature::FaceFix),
            StageConfig::Interpolate { factor, .. } if *factor > 2 => {
                need.push(Feature::InterpolateHighFactor)
            }
            _ => {}
        }
        if s.model().is_some_and(|m| m.path.is_some()) && !need.contains(&Feature::Byom) {
            need.push(Feature::Byom);
        }
    }
    if plan.encode.encoder.starts_with("prores") {
        need.push(Feature::ProResOutput);
    }
    if u64::from(plan.encode.width) * u64::from(plan.encode.height) > 3840 * 2160 {
        need.push(Feature::OutputAbove4k);
    }
    need
}

/// Privacy-preserving machine fingerprint for device-bound licenses:
/// SHA-256 of the OS machine id, salted with the product name. The raw id
/// never leaves the machine. `None` if the OS doesn't expose one.
///
/// The worker computes this itself rather than trusting the core.
pub fn machine_fingerprint() -> Option<String> {
    use sha2::Digest as _;
    let raw = raw_machine_id()?;
    let mut h = sha2::Sha256::new();
    h.update(b"framescale-device-v1:");
    h.update(raw.trim().as_bytes());
    Some(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn raw_machine_id() -> Option<String> {
    let run = |cmd: &str, args: &[&str]| -> Option<String> {
        let out = std::process::Command::new(cmd).args(args).output().ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    };
    if cfg!(target_os = "linux") {
        ["/etc/machine-id", "/var/lib/dbus/machine-id"]
            .iter()
            .find_map(|p| {
                std::fs::read_to_string(p)
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            })
    } else if cfg!(target_os = "macos") {
        let out = run("ioreg", &["-rd1", "-c", "IOPlatformExpertDevice"])?;
        out.lines()
            .find(|l| l.contains("IOPlatformUUID"))
            .and_then(|l| l.rsplit('"').nth(1))
            .map(str::to_owned)
    } else if cfg!(windows) {
        let out = run(
            "reg",
            &[
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ],
        )?;
        out.split_whitespace().last().map(str::to_owned)
    } else {
        None
    }
}

/// Token issuing (activation server / xtask only).
#[cfg(feature = "issuer")]
pub fn issue(signing_seed: &[u8; 32], payload: &Payload) -> String {
    use ed25519_dalek::Signer as _;
    let key = ed25519_dalek::SigningKey::from_bytes(signing_seed);
    let bytes = serde_json::to_vec(payload).unwrap_or_default();
    let sig = key.sign(&bytes);
    format!("{}.{}", B64.encode(&bytes), B64.encode(sig.to_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer as _;

    fn sign(seed: &[u8; 32], p: &Payload) -> String {
        let key = ed25519_dalek::SigningKey::from_bytes(seed);
        let bytes = serde_json::to_vec(p).expect("json");
        format!(
            "{}.{}",
            B64.encode(&bytes),
            B64.encode(key.sign(&bytes).to_bytes())
        )
    }

    fn payload() -> Payload {
        Payload {
            v: 1,
            lid: "L-1".into(),
            tier: Tier::Pro,
            features: vec![],
            licensee: Some("Ada".into()),
            iat: 1_700_000_000,
            exp: Some(2_000_000_000),
            device: Some("dev-hash".into()),
        }
    }

    #[test]
    fn valid_pro_token() {
        let v = Verifier {
            key: dev_verifying_key(),
            dev: true,
        };
        let e = v
            .verify(
                &sign(&DEV_SEED, &payload()),
                1_800_000_000,
                Some("dev-hash"),
            )
            .expect("ok");
        assert_eq!(e.tier, Tier::Pro);
        assert!(e.has(Feature::FaceFix));
    }

    #[test]
    fn rejects_tampering_expiry_device_and_foreign_keys() {
        let v = Verifier {
            key: dev_verifying_key(),
            dev: true,
        };
        let good = sign(&DEV_SEED, &payload());

        // Flip the tier inside the payload without re-signing.
        let (p, s) = good.split_once('.').expect("dot");
        let mut json: serde_json::Value =
            serde_json::from_slice(&B64.decode(p).expect("b64")).expect("json");
        json["lid"] = "L-2".into();
        let forged = format!(
            "{}.{s}",
            B64.encode(serde_json::to_vec(&json).expect("ser"))
        );
        assert_eq!(
            v.verify(&forged, 1_800_000_000, Some("dev-hash")),
            Err(LicenseError::BadSignature)
        );

        assert_eq!(
            v.verify(&good, 2_100_000_000, Some("dev-hash")),
            Err(LicenseError::Expired)
        );
        assert_eq!(
            v.verify(&good, 1_800_000_000, Some("other")),
            Err(LicenseError::WrongDevice)
        );
        assert_eq!(v.verify("garbage", 0, None), Err(LicenseError::Malformed));

        let other = sign(&[9u8; 32], &payload());
        assert_eq!(
            v.verify(&other, 1_800_000_000, Some("dev-hash")),
            Err(LicenseError::BadSignature)
        );

        let (e, err) = v.resolve(Some(&other), 1_800_000_000, None);
        assert_eq!(e.tier, Tier::Free);
        assert_eq!(err, Some(LicenseError::BadSignature));
    }

    #[test]
    fn hex_key_parsing() {
        assert!(parse_hex32(&"ab".repeat(32)).is_some());
        assert!(parse_hex32("abc").is_none());
    }
}
