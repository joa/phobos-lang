//! Readers for `PHOBOS_*` toggles. `ENV.md` lists every variable.
//! Read a toggle through these, not a match of its own, so `PHOBOS_X=0`
//! means the same thing for every `X`.

/// An opt-in toggle: on for `1`, `on`, `yes` or `true`; off otherwise,
/// including unset.
pub fn flag(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref().map(str::trim),
        Ok("1" | "on" | "yes" | "true")
    )
}

/// An opt-out toggle: off for `0`, `off`, `no` or `false`; on otherwise,
/// including unset.
pub fn flag_on(name: &str) -> bool {
    !matches!(
        std::env::var(name).as_deref().map(str::trim),
        Ok("0" | "off" | "no" | "false")
    )
}

/// An opt-out toggle that tells unset apart: `None` when unset, otherwise
/// [`flag_on`]'s reading. Use it for a toggle that falls back to a wider one.
pub fn flag_set(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| !matches!(v.trim(), "0" | "off" | "no" | "false"))
}
