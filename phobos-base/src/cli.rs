use std::fmt::Display;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail, ensure};

/// The program arguments without the executable name.
#[derive(Clone, Debug)]
pub struct Args {
    tokens: Vec<String>,
}

impl Args {
    pub fn from_env() -> Args {
        Args::new(std::env::args().skip(1))
    }

    pub fn new(tokens: impl IntoIterator<Item = String>) -> Args {
        Args {
            tokens: tokens.into_iter().collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn wants_help(&self) -> bool {
        self.has("-h") || self.has("-help") || self.has("--help")
    }

    pub fn has(&self, flag: &str) -> bool {
        self.tokens.iter().any(|t| t == flag)
    }

    pub fn value(&self, flag: &str) -> Result<Option<&str>> {
        let Some(at) = self.tokens.iter().position(|t| t == flag) else {
            return Ok(None);
        };

        self.tokens
            .get(at + 1)
            .map(String::as_str)
            .map(Some)
            .ok_or_else(|| anyhow!("{flag} expects a value"))
    }

    pub fn required(&self, flag: &str) -> Result<&str> {
        self.value(flag)?.ok_or_else(|| anyhow!("missing {flag}"))
    }

    /// [`Args::value`] over a set of spellings, for a flag with a short and a
    /// long form. The first one present wins.
    pub fn value_of(&self, flags: &[&str]) -> Result<Option<&str>> {
        for flag in flags {
            if let Some(value) = self.value(flag)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// The tokens that are neither a flag nor a flag's value.
    ///
    /// `valued` names every flag that takes one, since nothing else says where
    /// a value ends and a positional begins. A long flag outside that set is a
    /// typo rather than a positional, and is reported as one; a single-dash
    /// token is left alone, so a negative number still reads as an argument.
    pub fn positional(&self, valued: &[&str]) -> Result<Vec<&str>> {
        self.positional_with(valued, &[])
    }

    /// [`Args::positional`] where some long flags are switches.
    ///
    /// A switch carries no value, so nothing distinguishes it from a typo
    /// except being named, and `switches` is where it is named. Split from
    /// [`Args::positional`] so the common case still reads as one list.
    pub fn positional_with(&self, valued: &[&str], switches: &[&str]) -> Result<Vec<&str>> {
        let mut out = Vec::new();
        let mut tokens = self.tokens.iter();
        while let Some(token) = tokens.next() {
            if valued.contains(&token.as_str()) {
                tokens.next();
            } else if switches.contains(&token.as_str()) {
                continue;
            } else if token.starts_with("--") {
                return Err(anyhow!("unknown flag {token}"));
            } else {
                out.push(token.as_str());
            }
        }
        Ok(out)
    }

    pub fn parse<T>(&self, flag: &str) -> Result<Option<T>>
    where
        T: FromStr,
        T::Err: Display,
    {
        self.value(flag)?
            .map(|raw| {
                raw.parse::<T>()
                    .map_err(|e| anyhow!("{flag}: cannot parse '{raw}' ({e})"))
            })
            .transpose()
    }

    pub fn parse_required<T>(&self, flag: &str) -> Result<T>
    where
        T: FromStr,
        T::Err: Display,
    {
        self.parse(flag)?.ok_or_else(|| anyhow!("missing {flag}"))
    }

    /// [`Args::parse`] over a set of spellings. See [`Args::value_of`].
    pub fn parse_of<T>(&self, flags: &[&str]) -> Result<Option<T>>
    where
        T: FromStr,
        T::Err: Display,
    {
        for flag in flags {
            if let Some(value) = self.parse::<T>(flag)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    pub fn subcommand(&self) -> Option<(&str, Args)> {
        self.tokens
            .split_first()
            .map(|(head, tail)| (head.as_str(), Args::new(tail.iter().cloned())))
    }
}

/// A size a flag takes: a count of bytes, or one with a `k`, `m`, `g` or
/// `t` suffix in powers of 1024, in either case (`2g`, `1500M`, `512k`).
pub fn parse_size(text: &str) -> Result<u64> {
    let text = text.trim();
    let split = text
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let value: f64 = number
        .parse()
        .with_context(|| format!("'{text}' is not a size"))?;
    let scale: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => bail!("'{text}': unknown size unit '{other}'"),
    };
    ensure!(value >= 0.0, "'{text}' is negative");
    Ok((value * scale) as u64)
}

#[cfg(test)]
mod tests {
    use super::{Args, parse_size};

    fn args(xs: &[&str]) -> Args {
        Args::new(xs.iter().map(|s| s.to_string()))
    }

    #[test]
    fn reads_values_and_absence() {
        let a = args(&["--nodes", "4", "--autotune"]);
        assert_eq!(a.value("--nodes").unwrap(), Some("4"));
        assert_eq!(a.value("--missing").unwrap(), None);
        assert!(a.has("--autotune"));
        assert!(!a.has("--nodes-x"));
    }

    #[test]
    fn required_names_the_missing_flag() {
        let err = args(&[]).required("--job").unwrap_err().to_string();
        assert_eq!(err, "missing --job");
    }

    #[test]
    fn dangling_flag_wants_a_value() {
        let err = args(&["--nodes"]).value("--nodes").unwrap_err().to_string();
        assert_eq!(err, "--nodes expects a value");
    }

    #[test]
    fn parses_into_target_types() {
        let a = args(&["--nodes", "4", "--rate", "1.5"]);
        assert_eq!(a.parse_required::<u16>("--nodes").unwrap(), 4);
        assert_eq!(a.parse::<f64>("--rate").unwrap(), Some(1.5));
        assert_eq!(a.parse::<u64>("--budget").unwrap(), None);
    }

    #[test]
    fn parse_error_quotes_the_offending_text() {
        let err = args(&["--nodes", "big"])
            .parse::<u16>("--nodes")
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("--nodes: cannot parse 'big'"), "got: {err}");
    }

    #[test]
    fn help_and_empty() {
        assert!(args(&["--help"]).wants_help());
        assert!(args(&["-h"]).wants_help());
        assert!(!args(&["--job", "x"]).wants_help());
        assert!(args(&[]).is_empty());
    }

    #[test]
    fn a_flag_can_have_a_short_and_a_long_spelling() {
        let a = args(&["-n", "4"]);
        assert_eq!(a.value_of(&["-n", "--num"]).unwrap(), Some("4"));
        assert_eq!(a.parse_of::<u16>(&["--num", "-n"]).unwrap(), Some(4));
        assert_eq!(a.value_of(&["-k", "--top-k"]).unwrap(), None);
    }

    #[test]
    fn positionals_skip_flag_values() {
        let a = args(&["--model", "m.gguf", "hello", "-n", "8", "world"]);
        assert_eq!(
            a.positional(&["--model", "-n"]).unwrap(),
            vec!["hello", "world"]
        );
    }

    #[test]
    fn an_unknown_long_flag_is_not_a_positional() {
        let err = args(&["--typo", "hello"])
            .positional(&["--model"])
            .unwrap_err()
            .to_string();
        assert_eq!(err, "unknown flag --typo");
    }

    #[test]
    fn a_switch_is_neither_a_positional_nor_a_typo() {
        let a = args(&["--no-tui", "hello", "--listen", "addr", "world"]);
        assert_eq!(
            a.positional_with(&["--listen"], &["--no-tui"]).unwrap(),
            vec!["hello", "world"]
        );
        assert!(a.has("--no-tui"));
        assert!(a.positional(&["--listen"]).is_err());
    }

    #[test]
    fn a_negative_number_stays_a_positional() {
        let a = args(&["-40", "degrees"]);
        assert_eq!(a.positional(&[]).unwrap(), vec!["-40", "degrees"]);
    }

    #[test]
    fn subcommand_splits_head_from_rest() {
        let a = args(&["init", "--uri", "u"]);
        let (cmd, rest) = a.subcommand().unwrap();
        assert_eq!(cmd, "init");
        assert_eq!(rest.value("--uri").unwrap(), Some("u"));
        assert!(args(&[]).subcommand().is_none());
    }

    #[test]
    fn sizes_take_a_unit_in_powers_of_1024() {
        assert_eq!(parse_size("2g").unwrap(), 2 << 30);
        assert_eq!(parse_size("1500M").unwrap(), 1500 << 20);
        assert_eq!(parse_size("512 KiB").unwrap(), 512 << 10);
        assert_eq!(parse_size("1.5g").unwrap(), 3 << 29);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert!(parse_size("2x").is_err());
        assert!(parse_size("lots").is_err());
    }
}
