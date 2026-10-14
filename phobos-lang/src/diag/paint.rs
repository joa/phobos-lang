// The night sky as terminal escapes: one role per thing a diagnostic draws,
// and one per class of highlighted source. The source keeps the website's
// code colors (keywords in the nebula, builtins and `ctx` in the brand),
// except numbers, which are gold rather than Mars so they never read as an
// error's red.

use phobos_base::palette::{self, Rgb};

use super::Colors;

/// What a highlighted piece of source is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Keyword,
    /// A scalar, `tile` or `tensor`.
    Type,
    Number,
    /// `@name`.
    Attribute,
    /// `ctx` and its fields.
    Ctx,
    /// A name called, a builtin or a `fn`.
    Call,
    Comment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Role {
    Plain,
    Error,
    Warning,
    Note,
    Help,
    /// The headline's message.
    Message,
    /// The gutter, line numbers and the `-->` arrow.
    Chrome,
    /// A secondary label's underline and text.
    Secondary,
    Source(Class),
}

impl Role {
    /// The color, the named color standing in for it, and boldness.
    fn style(self) -> Option<(Option<(Rgb, u8)>, bool)> {
        let brand = (palette::BRAND, 94);
        let brand_hi = (palette::BRAND_HI, 96);
        let nebula = (palette::NEBULA, 95);
        Some(match self {
            Role::Plain => return None,
            Role::Error => (Some((palette::MARS, 91)), true),
            Role::Warning => (Some((palette::GOLD, 93)), true),
            Role::Note | Role::Help => (Some(brand_hi), true),
            Role::Message => (None, true),
            Role::Chrome | Role::Secondary => (Some(brand), true),
            Role::Source(class) => (
                Some(match class {
                    Class::Keyword | Class::Attribute => nebula,
                    Class::Type => brand_hi,
                    Class::Number => (palette::GOLD, 93),
                    Class::Ctx | Class::Call => brand,
                    Class::Comment => (palette::DUST, 90),
                }),
                false,
            ),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Paint {
    pub colors: Colors,
}

impl Paint {
    pub fn push(&self, out: &mut String, role: Role, text: &str) {
        let style = match self.colors {
            Colors::Off => None,
            _ => role.style(),
        };
        let Some((color, bold)) = style.filter(|_| !text.is_empty()) else {
            out.push_str(text);
            return;
        };
        if bold {
            out.push_str("\x1b[1m");
        }
        match (color, self.colors) {
            (Some(((r, g, b), _)), Colors::Rgb) => out.push_str(&format!("\x1b[38;2;{r};{g};{b}m")),
            (Some((_, basic)), _) => out.push_str(&format!("\x1b[{basic}m")),
            (None, _) => {}
        }
        out.push_str(text);
        out.push_str("\x1b[0m");
    }
}
