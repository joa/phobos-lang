// The display font: the wordmark and the big figures.
//
// Glyphs are drawn on a grid of half cells, two across and two down per
// cell, and rendered with the quadrant blocks, so a bowl can round its
// corners and a stem can be a cell and a half wide. Stems are three half
// cells wide and bars one half row tall, which keeps the contrast of a sans
// like the website's, set in its case: "Phobos", not "PHOBOS".

/// Rows the font is drawn in, ten half rows.
pub const HEIGHT: usize = 5;

/// Half cells between two glyphs.
const GAP: usize = 2;

/// Quadrant blocks indexed by their four quarters, top left as the highest
/// bit, then top right, bottom left, bottom right.
const QUADRANTS: [char; 16] = [
    ' ', '\u{2597}', '\u{2596}', '\u{2584}', '\u{259d}', '\u{2590}', '\u{259e}', '\u{259f}',
    '\u{2598}', '\u{259a}', '\u{258c}', '\u{2599}', '\u{2580}', '\u{259c}', '\u{259b}', '\u{2588}',
];

/// `text` in the font, as [`HEIGHT`] rows of equal length. A space in a row
/// is unlit.
pub fn render(text: &str) -> Vec<String> {
    // Lay the glyphs out on the half-cell grid first.
    let mut grid = vec![Vec::<bool>::new(); HEIGHT * 2];
    for (i, ch) in text.chars().enumerate() {
        let glyph = glyph(ch);
        for (row, line) in grid.iter_mut().zip(glyph) {
            if i > 0 {
                row.extend([false; GAP]);
            }
            row.extend(line.bytes().map(|b| b == b'#'));
        }
    }
    let half_width = grid[0].len();
    let lit = |x: usize, y: usize| x < half_width && grid[y][x];
    (0..HEIGHT)
        .map(|row| {
            (0..half_width.div_ceil(2))
                .map(|col| {
                    let (x, y) = (col * 2, row * 2);
                    let bits = (lit(x, y) as usize) << 3
                        | (lit(x + 1, y) as usize) << 2
                        | (lit(x, y + 1) as usize) << 1
                        | lit(x + 1, y + 1) as usize;
                    QUADRANTS[bits]
                })
                .collect()
        })
        .collect()
}

/// Width [`render`] will produce for `text`, in cells.
pub fn width(text: &str) -> usize {
    let count = text.chars().count();
    if count == 0 {
        return 0;
    }
    let half: usize = text.chars().map(|c| glyph(c)[0].len()).sum::<usize>() + (count - 1) * GAP;
    half.div_ceil(2)
}

/// One glyph, ten rows of half cells, `#` lit. The lowercase letters stand
/// on the baseline with their bowls seven rows high.
fn glyph(ch: char) -> [&'static str; HEIGHT * 2] {
    match ch {
        'P' => [
            "#########.",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
            "#########.",
            "###.......",
            "###.......",
            "###.......",
            "###.......",
        ],
        'h' => [
            "###.......",
            "###.......",
            "###.......",
            "#########.",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
        ],
        'o' => [
            "..........",
            "..........",
            "..........",
            ".########.",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
            ".########.",
        ],
        'b' => [
            "###.......",
            "###.......",
            "###.......",
            "#########.",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
            "###....###",
            "#########.",
        ],
        's' => [
            ".........",
            ".........",
            ".........",
            ".########",
            "###......",
            "###......",
            ".#######.",
            "......###",
            "......###",
            "########.",
        ],
        '0' => [
            ".#######.",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            ".#######.",
        ],
        '1' => [
            "####..", "..###.", "..###.", "..###.", "..###.", "..###.", "..###.", "..###.",
            "..###.", "######",
        ],
        '2' => [
            ".#######.",
            "......###",
            "......###",
            "......###",
            ".#######.",
            "###......",
            "###......",
            "###......",
            "###......",
            "#########",
        ],
        '3' => [
            "########.",
            "......###",
            "......###",
            "......###",
            "..######.",
            "......###",
            "......###",
            "......###",
            "......###",
            "########.",
        ],
        '4' => [
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            ".########",
            "......###",
            "......###",
            "......###",
            "......###",
        ],
        '5' => [
            "#########",
            "###......",
            "###......",
            "###......",
            "########.",
            "......###",
            "......###",
            "......###",
            "......###",
            "########.",
        ],
        '6' => [
            ".########",
            "###......",
            "###......",
            "###......",
            "########.",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            ".#######.",
        ],
        '7' => [
            "#########",
            "......###",
            "......###",
            "......###",
            "......###",
            "......###",
            "......###",
            "......###",
            "......###",
            "......###",
        ],
        '8' => [
            ".#######.",
            "###...###",
            "###...###",
            "###...###",
            ".#######.",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            ".#######.",
        ],
        '9' => [
            ".#######.",
            "###...###",
            "###...###",
            "###...###",
            "###...###",
            ".########",
            "......###",
            "......###",
            "......###",
            "########.",
        ],
        '.' => [
            "...", "...", "...", "...", "...", "...", "...", "...", "###", "###",
        ],
        _ => [
            ".........",
            ".........",
            ".........",
            ".........",
            ".........",
            ".........",
            ".........",
            ".........",
            ".........",
            ".........",
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_is_ten_rows_of_one_width() {
        for ch in "Phobs0123456789. ".chars() {
            let g = glyph(ch);
            assert!(g.iter().all(|row| row.len() == g[0].len()), "{ch:?}");
        }
    }

    #[test]
    fn rows_come_out_as_wide_as_promised() {
        for text in ["Phobos", "1422", "45.8", "0", ""] {
            let rows = render(text);
            assert_eq!(rows.len(), HEIGHT);
            for row in &rows {
                assert_eq!(row.chars().count(), width(text), "{text:?}");
            }
        }
    }

    #[test]
    fn the_quadrants_land_where_the_grid_says() {
        // The P's top row: a full cell, then the stem's right half over a
        // bar, so the first row starts with a full block.
        let rows = render("P");
        assert!(rows[0].starts_with('\u{2588}'), "{rows:?}");
        // The descender ends on a full stem cell and a left half.
        assert_eq!(rows[4].trim_end(), "\u{2588}\u{258c}");
    }
}
