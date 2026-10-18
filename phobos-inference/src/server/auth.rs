//! Bearer-token check for every route, and the warning for an open bind.

use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::Response,
};
use std::net::ToSocketAddrs;
use std::sync::Arc;

use super::protocol::error_response;

/// Whether `header` is `Bearer KEY`. Compares every byte, so the time taken
/// does not say how much of a guess was right.
pub(crate) fn authorized(header: Option<&str>, key: &str) -> bool {
    let Some(token) = header.and_then(|h| h.strip_prefix("Bearer ")) else {
        return false;
    };
    let (token, key) = (token.as_bytes(), key.as_bytes());
    let mut diff = token.len() ^ key.len();
    for (i, &k) in key.iter().enumerate() {
        diff |= usize::from(token.get(i).copied().unwrap_or(0) ^ k);
    }
    diff == 0
}

pub(crate) async fn require_key(State(key): State<Arc<str>>, req: Request, next: Next) -> Response {
    let sent = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    if authorized(sent, &key) {
        next.run(req).await
    } else {
        let mut res = error_response(StatusCode::UNAUTHORIZED, "invalid or missing API key");
        res.headers_mut()
            .insert(header::WWW_AUTHENTICATE, "Bearer".parse().unwrap());
        res
    }
}

/// Whether `addr` can be reached from another machine. An address that does
/// not resolve counts as reachable, since it is not known to be safe.
pub(crate) fn is_exposed(addr: &str) -> bool {
    match addr.to_socket_addrs() {
        Ok(mut resolved) => resolved.any(|a| !a.ip().is_loopback()),
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_exact_bearer_token_passes() {
        assert!(authorized(Some("Bearer s3cret"), "s3cret"));
        assert!(!authorized(None, "s3cret"));
        assert!(!authorized(Some("s3cret"), "s3cret"));
        assert!(!authorized(Some("Bearer s3cre"), "s3cret"));
        assert!(!authorized(Some("Bearer s3crets"), "s3cret"));
        assert!(!authorized(Some("Bearer "), "s3cret"));
    }

    #[test]
    fn loopback_is_not_exposed() {
        assert!(!is_exposed("127.0.0.1:8080"));
        assert!(!is_exposed("[::1]:8080"));
        assert!(is_exposed("0.0.0.0:8080"));
        assert!(is_exposed("not an address"));
    }
}
