// Collectives and shuffles: `sum<over ctx.lane>(x)`, `shfl<xor 16>(v)`.
//
// The parser builds each as a call of an internal name the lexer cannot
// produce, with the configuration as trailing arguments, so the passes
// between the parser and the build see an ordinary call.

use super::Expr;

/// The prefix of a collective's internal name; the operation follows it.
pub const COLLECTIVE: &str = "#collective.";
/// The prefix of a shuffle's internal name; the mode follows it.
pub const SHUFFLE: &str = "#shfl.";

/// The names a combine `fn` sees its two operands as.
pub const COMBINE_A: &str = "#a";
pub const COMBINE_B: &str = "#b";

/// What a collective folds its group's values into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Sum,
    Max,
    Min,
    /// The index of the largest value, the smallest index on a tie.
    ArgMax,
    Any,
    All,
    /// A combine `fn` of two operands.
    Reduce,
}

impl Fold {
    pub const ALL: [Fold; 7] = [
        Fold::Sum,
        Fold::Max,
        Fold::Min,
        Fold::ArgMax,
        Fold::Any,
        Fold::All,
        Fold::Reduce,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Fold::Sum => "sum",
            Fold::Max => "max",
            Fold::Min => "min",
            Fold::ArgMax => "argmax",
            Fold::Any => "any",
            Fold::All => "all",
            Fold::Reduce => "reduce",
        }
    }

    pub fn from_name(name: &str) -> Option<Fold> {
        Fold::ALL.into_iter().find(|f| f.name() == name)
    }
}

/// Which threads a collective runs across.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// `over ctx.lane`: the lanes of a warp, or of a `width`-lane piece of one.
    Lanes,
    /// `over ctx.thread`: every thread of the CTA.
    Cta,
}

/// Where a shuffle reads from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `from src`: lane `src`.
    From,
    /// `xor m`: lane `lane ^ m`.
    Xor,
    /// `down d`: lane `lane + d`.
    Down,
    /// `up d`: lane `lane - d`.
    Up,
}

impl Mode {
    pub const ALL: [Mode; 4] = [Mode::From, Mode::Xor, Mode::Down, Mode::Up];

    pub fn name(self) -> &'static str {
        match self {
            Mode::From => "from",
            Mode::Xor => "xor",
            Mode::Down => "down",
            Mode::Up => "up",
        }
    }

    pub fn from_name(name: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| m.name() == name)
    }
}

/// A collective call: `[value, group, width]`, and the combine for
/// `reduce`. `width` 0 means the whole group.
pub fn collective(fold: Fold, value: Expr, group: Group, width: Expr, combine: Option<Expr>) -> Expr {
    let group = Expr::Int(match group {
        Group::Lanes => 0,
        Group::Cta => 1,
    });
    let mut args = vec![value, group, width];
    args.extend(combine);
    Expr::Call {
        callee: format!("{COLLECTIVE}{}", fold.name()),
        args,
    }
}

/// A shuffle call: `[value, operand, width]`.
pub fn shuffle(mode: Mode, value: Expr, operand: Expr, width: Expr) -> Expr {
    Expr::Call {
        callee: format!("{SHUFFLE}{}", mode.name()),
        args: vec![value, operand, width],
    }
}

/// The parts of a collective call.
pub struct Collective<'a> {
    pub fold: Fold,
    pub value: &'a Expr,
    pub group: Group,
    pub width: &'a Expr,
    pub combine: Option<&'a Expr>,
}

pub fn as_collective<'a>(callee: &str, args: &'a [Expr]) -> Option<Collective<'a>> {
    let fold = Fold::from_name(callee.strip_prefix(COLLECTIVE)?)?;
    let [value, Expr::Int(group), width, rest @ ..] = args else {
        return None;
    };
    Some(Collective {
        fold,
        value,
        group: if *group == 0 { Group::Lanes } else { Group::Cta },
        width,
        combine: rest.first(),
    })
}

pub fn as_shuffle<'a>(callee: &str, args: &'a [Expr]) -> Option<(Mode, &'a Expr, &'a Expr, &'a Expr)> {
    let mode = Mode::from_name(callee.strip_prefix(SHUFFLE)?)?;
    let [value, operand, width] = args else {
        return None;
    };
    Some((mode, value, operand, width))
}
