// The logo, the website's mark drawn as ASCII line art at any height.
//
// The geometry is the mark's SVG (`website/src/assets/mark.svg`): an oblong
// body, a crater in its upper right, a pit and a dot below. Each outline is
// walked densely and reduced to one cell per row where it runs steep and one
// per column where it runs flat, so a line stays one character thick, and
// each cell gets the glyph of the outline's direction through it.

/// The body's outline, as the SVG's cubic segments in absolute coordinates.
const BODY: [[(f32, f32); 4]; 7] = [
    [(4.4, 18.6), (3.6, 14.3), (5.8, 10.2), (9.9, 8.3)],
    [(9.9, 8.3), (12.9, 6.9), (16.0, 7.3), (18.9, 6.8)],
    [(18.9, 6.8), (23.0, 6.2), (27.1, 8.2), (28.2, 12.2)],
    [(28.2, 12.2), (29.2, 15.7), (28.0, 19.2), (25.3, 21.8)],
    [(25.3, 21.8), (22.4, 24.6), (18.3, 25.8), (14.3, 25.4)],
    [(14.3, 25.4), (11.4, 25.1), (9.5, 24.2), (7.6, 23.0)],
    [(7.6, 23.0), (5.9, 21.9), (4.7, 20.4), (4.4, 18.6)],
];

/// Centre and radius of the crater, the pit, and the dot.
const CRATER: (f32, f32, f32) = (21.2, 13.8, 4.3);
const PIT: (f32, f32, f32) = (11.0, 18.7, 1.3);
const DOT: (f32, f32) = (15.4, 21.9);

/// The corner the drawing is measured from and its height, in SVG units.
const LEFT: f32 = 3.6;
const TOP: f32 = 6.2;
const SPAN: f32 = 19.4;

/// Points walked per outline. Dense enough that no cell is skipped at any
/// height a terminal has.
const STEPS: usize = 1400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Body,
    Crater,
    Pit,
}

/// One drawn cell.
#[derive(Clone, Copy, Debug)]
pub struct Stroke {
    pub x: u16,
    pub y: u16,
    pub glyph: char,
    pub part: Part,
    /// How far along its outline the cell lies, 0.0 to 1.0, so the mark can
    /// be drawn the way a pen would.
    pub at: f32,
}

/// The mark at one height.
pub struct Mark {
    /// The rows it was asked to fit.
    pub rows: u16,
    pub width: u16,
    pub height: u16,
    pub strokes: Vec<Stroke>,
    /// Row-major, whether a cell lies inside the body.
    inside: Vec<bool>,
}

impl Mark {
    /// The mark drawn `rows` rows high. A row is taken to be twice as tall as
    /// a column is wide.
    pub fn new(rows: u16) -> Mark {
        let scale = rows.saturating_sub(1).max(1) as f32 / SPAN;
        let to_cell = |(x, y): (f32, f32)| ((x - LEFT) * scale * 2.0, (y - TOP) * scale);

        let mut strokes = Vec::new();
        let body: Vec<(f32, f32)> = (0..STEPS)
            .map(|i| {
                let t = i as f32 / STEPS as f32 * BODY.len() as f32;
                let segment = BODY[(t as usize).min(BODY.len() - 1)];
                to_cell(cubic(segment, t.fract()))
            })
            .collect();
        trace(&body, Part::Body, &mut strokes);
        trace(&circle(CRATER, &to_cell), Part::Crater, &mut strokes);
        let (px, py) = to_cell((PIT.0, PIT.1));
        // A pit too small to draw as a ring is a single letter.
        if PIT.2 * scale * 2.0 >= 2.5 {
            trace(&circle(PIT, &to_cell), Part::Pit, &mut strokes);
        } else {
            place(&mut strokes, px.round(), py.round(), 'o', Part::Pit, 0.0);
        }
        let (dx, dy) = to_cell(DOT);
        place(&mut strokes, dx.round(), dy.round(), '.', Part::Pit, 1.0);

        let width = strokes.iter().map(|s| s.x + 1).max().unwrap_or(0);
        let height = strokes.iter().map(|s| s.y + 1).max().unwrap_or(0);
        let inside = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| contains(&body, (x as f32, y as f32)))
            .collect();
        Mark {
            rows,
            width,
            height,
            strokes,
            inside,
        }
    }

    /// Whether `(x, y)` lies inside the body, outline excluded or not.
    pub fn inside(&self, x: u16, y: u16) -> bool {
        x < self.width && y < self.height && self.inside[(y * self.width + x) as usize]
    }
}

fn cubic([a, b, c, d]: [(f32, f32); 4], t: f32) -> (f32, f32) {
    let u = 1.0 - t;
    let blend = |a: f32, b: f32, c: f32, d: f32| {
        u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d
    };
    (blend(a.0, b.0, c.0, d.0), blend(a.1, b.1, c.1, d.1))
}

/// A circle as cell coordinates, starting at its top and running clockwise.
fn circle(
    (cx, cy, r): (f32, f32, f32),
    to_cell: &dyn Fn((f32, f32)) -> (f32, f32),
) -> Vec<(f32, f32)> {
    (0..STEPS)
        .map(|i| {
            let angle =
                i as f32 / STEPS as f32 * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
            to_cell((cx + r * angle.cos(), cy + r * angle.sin()))
        })
        .collect()
}

/// Reduce a closed outline to cells.
///
/// Consecutive points sharing a row (where the outline is steep) or a column
/// (where it is flat) form a run, and each run becomes the one cell whose
/// point lies nearest the cell's centre line.
fn trace(points: &[(f32, f32)], part: Part, out: &mut Vec<Stroke>) {
    let n = points.len();
    // Per point: its slope in degrees from the horizontal, 0 to 90, the sign
    // of the slope, and the row or column that groups it.
    let info: Vec<(f32, bool, bool, i32)> = (0..n)
        .map(|i| {
            let (nx, ny) = points[(i + 1) % n];
            let (px, py) = points[(i + n - 1) % n];
            let (dx, dy) = (nx - px, ny - py);
            let mut angle = dy.atan2(dx).to_degrees().abs();
            if angle > 90.0 {
                angle = 180.0 - angle;
            }
            let steep = angle >= 45.0;
            let (x, y) = points[i];
            let key = if steep { y.round() } else { x.round() } as i32;
            (angle, dx * dy < 0.0, steep, key)
        })
        .collect();

    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && info[end].2 == info[start].2 && info[end].3 == info[start].3 {
            end += 1;
        }
        let steep = info[start].2;
        // The point nearest the centre line of the run's row or column.
        let best = (start..end)
            .min_by(|&a, &b| {
                let off = |i: usize| {
                    let v = if steep { points[i].1 } else { points[i].0 };
                    (v - v.round()).abs()
                };
                off(a).total_cmp(&off(b))
            })
            .unwrap_or(start);
        let (x, y) = points[best];
        let (angle, rising, _, _) = info[best];
        let at = best as f32 / n as f32;
        let diagonal = if rising { '/' } else { '\\' };
        if steep {
            let glyph = if angle > 62.0 { '|' } else { diagonal };
            place(out, x.round(), y.round(), glyph, part, at);
        } else if angle >= 28.0 {
            place(out, x.round(), y.round(), diagonal, part, at);
        } else if (y - y.round()).abs() < 0.25 {
            place(out, x.round(), y.round(), '-', part, at);
        } else {
            // Nearer a cell's floor than its middle: an underscore in the
            // row above sits there.
            place(out, x.round(), y.floor(), '_', part, at);
        }
        start = end;
    }
}

/// Add a cell unless one is already drawn there. Cells off the top or left
/// edge are dropped.
fn place(out: &mut Vec<Stroke>, x: f32, y: f32, glyph: char, part: Part, at: f32) {
    if x < 0.0 || y < 0.0 {
        return;
    }
    let (x, y) = (x as u16, y as u16);
    if out.iter().any(|s| s.x == x && s.y == y) {
        return;
    }
    out.push(Stroke {
        x,
        y,
        glyph,
        part,
        at,
    });
}

/// Even-odd point in polygon.
fn contains(polygon: &[(f32, f32)], (x, y): (f32, f32)) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for (i, &(xi, yi)) in polygon.iter().enumerate() {
        let (xj, yj) = polygon[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(mark: &Mark) -> Vec<String> {
        let mut rows = vec![vec![' '; mark.width as usize]; mark.height as usize];
        for s in &mark.strokes {
            rows[s.y as usize][s.x as usize] = s.glyph;
        }
        rows.into_iter().map(|r| r.into_iter().collect()).collect()
    }

    #[test]
    fn the_mark_keeps_to_the_rows_it_is_given() {
        for rows in [5, 7, 12, 20, 40] {
            let mark = Mark::new(rows);
            assert!(
                mark.height <= rows,
                "{rows} rows drew {}:\n{}",
                mark.height,
                picture(&mark).join("\n")
            );
            // Twice as wide as tall, give or take the oblong body.
            assert!(mark.width >= rows * 2, "{rows} rows: {} wide", mark.width);
        }
    }

    #[test]
    fn the_crater_sits_upper_right_and_the_pit_lower_left() {
        let mark = Mark::new(16);
        let centre = |part: Part| {
            let cells: Vec<_> = mark.strokes.iter().filter(|s| s.part == part).collect();
            let n = cells.len() as f32;
            let x = cells.iter().map(|s| s.x as f32).sum::<f32>() / n;
            let y = cells.iter().map(|s| s.y as f32).sum::<f32>() / n;
            (x, y)
        };
        let (cx, cy) = centre(Part::Crater);
        let (px, py) = centre(Part::Pit);
        assert!(cx > px && cy < py, "crater ({cx}, {cy}), pit ({px}, {py})");
        assert!(mark.inside(cx as u16, cy as u16));
        assert!(!mark.inside(0, 0));
    }

    #[test]
    fn a_line_stays_one_character_thick() {
        // No row holds two neighbouring cells of the same vertical glyph.
        let mark = Mark::new(18);
        for row in picture(&mark) {
            assert!(!row.contains("||"), "{row}");
        }
    }
}
