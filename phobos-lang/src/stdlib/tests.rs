use super::*;

#[test]
fn the_shipped_library_parses() {
    assert!(!builtin().is_empty());
}
