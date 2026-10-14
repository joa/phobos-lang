//! Whether, and how richly, to color what goes to stderr.
//!
//! `--color auto` follows the conventions other tools do: `NO_COLOR` turns
//! color off, `CLICOLOR_FORCE` turns it on, and otherwise it is on for a
//! terminal that is not `TERM=dumb`.

use std::io::IsTerminal;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    #[default]
    Auto,
    Always,
    Never,
}

impl std::str::FromStr for ColorChoice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            other => Err(format!("--color takes auto, always or never, not `{other}`")),
        }
    }
}

/// What a terminal can show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colors {
    Off,
    /// The sixteen named colors.
    Basic,
    /// 24-bit RGB.
    Rgb,
}

/// The colors stderr gets under `choice`. Switches a Windows console into
/// escape-sequence mode on the way, which conhost needs and Windows Terminal
/// already has.
pub fn stderr_colors(choice: ColorChoice) -> Colors {
    let set = |name: &str| std::env::var_os(name).is_some_and(|v| !v.is_empty());
    let on = match choice {
        ColorChoice::Never => false,
        ColorChoice::Always => true,
        ColorChoice::Auto if set("NO_COLOR") => false,
        ColorChoice::Auto if set("CLICOLOR_FORCE") && std::env::var("CLICOLOR_FORCE").as_deref() != Ok("0") => true,
        ColorChoice::Auto => {
            std::io::stderr().is_terminal() && std::env::var("TERM").as_deref() != Ok("dumb")
        }
    };
    if !on {
        return Colors::Off;
    }
    // A console that takes escape sequences takes RGB ones; a terminal that
    // is not a console (mintty) handles them itself.
    if cfg!(windows) {
        enable_escapes();
        return Colors::Rgb;
    }
    let truecolor = matches!(std::env::var("COLORTERM").as_deref(), Ok("truecolor" | "24bit"));
    if truecolor { Colors::Rgb } else { Colors::Basic }
}

#[cfg(windows)]
fn enable_escapes() {
    use std::ffi::c_void;
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> *mut c_void;
        fn GetConsoleMode(console: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(console: *mut c_void, mode: u32) -> i32;
    }
    // SAFETY: the handle comes from the OS and is only passed back to it; a
    // handle that is not a console fails GetConsoleMode and is left alone.
    unsafe {
        let handle = GetStdHandle(STD_ERROR_HANDLE);
        let mut mode = 0;
        if GetConsoleMode(handle, &mut mode) != 0 {
            SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
}

#[cfg(not(windows))]
fn enable_escapes() {}
