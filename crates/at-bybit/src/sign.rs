//! Bybit v5 HMAC-SHA256 signatures.

use hmac::{Hmac, Mac};
use sha2::Sha256;

pub fn hmac_hex(secret: &str, message: &str) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(message.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// REST: sign(timestamp + api_key + recv_window + payload), payload = query string (GET) or JSON body (POST).
pub fn rest_signature(secret: &str, timestamp_ms: i64, api_key: &str, recv_window_ms: u64, payload: &str) -> String {
    hmac_hex(secret, &format!("{timestamp_ms}{api_key}{recv_window_ms}{payload}"))
}

/// WebSocket auth: sign("GET/realtime" + expires).
pub fn ws_signature(secret: &str, expires_ms: i64) -> String {
    hmac_hex(secret, &format!("GET/realtime{expires_ms}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_vectors() {
        // Reference values computed independently with Python's hmac/hashlib.
        assert_eq!(
            rest_signature("secret", 1700000000000, "key", 5000, "category=linear&symbol=BTCUSDT"),
            "3906b813750309cce9879a975510651953382a28592d69104d0b599e3d201f40"
        );
        assert_eq!(ws_signature("secret", 1700000010000), "7ccf9bb4db01ad0e1ee2c3eef8d8b4730fc9bcab069e44972f6723cde7f692f3");
        assert_eq!(hmac_hex("key", "The quick brown fox jumps over the lazy dog"),
            "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8");
    }
}
