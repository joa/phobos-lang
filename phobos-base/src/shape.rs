use anyhow::{Context, Result};

pub fn parse(s: &str) -> Result<Vec<u64>> {
    s.split('x')
        .map(|d| {
            d.trim()
                .parse::<u64>()
                .with_context(|| format!("bad shape extent '{d}'"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parses_rank_one_and_two() {
        assert_eq!(parse("16").unwrap(), vec![16]);
        assert_eq!(parse("4x8").unwrap(), vec![4, 8]);
        assert_eq!(parse(" 4 x 8 ").unwrap(), vec![4, 8]);
    }

    #[test]
    fn rejects_non_numeric_extents() {
        let err = parse("4xN").unwrap_err().to_string();
        assert!(err.contains("bad shape extent 'N'"), "got: {err}");
    }
}
