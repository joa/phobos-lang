// `ctx`, the kernel's implicit view of where it runs.
//
// The parser turns a field path such as `ctx.block.x` into a call of an
// internal name the lexer cannot produce, as it does for the jumps, so
// everything past the parser sees an ordinary builtin call.

use super::Expr;

/// `ctx.block.<axis>`: the CTA's index in the grid.
pub const BLOCK: &str = "#ctx.block";
/// `ctx.blocks.<axis>`: the grid's extent.
pub const BLOCKS: &str = "#ctx.blocks";
/// `ctx.threads`: the CTA's thread count, from `@launch`.
pub const THREADS: &str = "#ctx.threads";
/// `ctx.warps`: the CTA's warp count.
pub const WARPS: &str = "#ctx.warps";
/// `ctx.lanes`: the target's warp width.
pub const LANES: &str = "#ctx.lanes";
/// `ctx.thread`: the thread's index in the CTA.
pub const THREAD: &str = "#ctx.thread";
/// `ctx.warp`: the warp's index in the CTA.
pub const WARP: &str = "#ctx.warp";
/// `ctx.lane`: the thread's index in its warp.
pub const LANE: &str = "#ctx.lane";

/// The name `ctx` is reserved for.
pub const NAME: &str = "ctx";

const AXES: [&str; 3] = ["x", "y", "z"];

/// `ctx.block.<axis>` for axis 0, 1 or 2.
pub fn block(axis: u8) -> Expr {
    axis_call(BLOCK, axis)
}

fn axis_call(callee: &str, axis: u8) -> Expr {
    Expr::Call {
        callee: callee.into(),
        args: vec![Expr::Int(axis.into())],
    }
}

/// The call `ctx.<path>` stands for.
pub fn field(path: &[String]) -> Result<Expr, String> {
    let path: Vec<&str> = path.iter().map(String::as_str).collect();
    let bare = |callee: &str| Expr::Call {
        callee: callee.into(),
        args: Vec::new(),
    };
    Ok(match path[..] {
        [group @ ("block" | "blocks"), axis] => {
            let Some(d) = AXES.iter().position(|a| *a == axis) else {
                return Err(format!("`ctx.{group}` has axes x, y and z, not `{axis}`"));
            };
            axis_call(if group == "block" { BLOCK } else { BLOCKS }, d as u8)
        }
        [group @ ("block" | "blocks")] => {
            return Err(format!("`ctx.{group}` needs an axis: `ctx.{group}.x`"));
        }
        ["threads"] => bare(THREADS),
        ["warps"] => bare(WARPS),
        ["lanes"] => bare(LANES),
        ["thread"] => bare(THREAD),
        ["warp"] => bare(WARP),
        ["lane"] => bare(LANE),
        _ => return Err(format!("`ctx` has no field `{}`", path.join("."))),
    })
}

/// How a `ctx` call is spelled in source, for messages.
pub fn spelling(callee: &str, args: &[Expr]) -> Option<String> {
    let axis = || match args {
        [Expr::Int(d)] => AXES.get(*d as usize).copied(),
        _ => None,
    };
    Some(match callee {
        BLOCK => format!("ctx.block.{}", axis()?),
        BLOCKS => format!("ctx.blocks.{}", axis()?),
        THREADS => "ctx.threads".into(),
        WARPS => "ctx.warps".into(),
        LANES => "ctx.lanes".into(),
        THREAD => "ctx.thread".into(),
        WARP => "ctx.warp".into(),
        LANE => "ctx.lane".into(),
        _ => return None,
    })
}
