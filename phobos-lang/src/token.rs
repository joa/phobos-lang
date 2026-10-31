#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),

    // keywords
    Kernel,
    Fn,
    Let,
    Var,
    If,
    Else,
    For,
    In,
    While,
    Break,
    Continue,
    Return,
    True,
    False,

    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    At,
    Dot,

    // operators
    Plus,
    ColonPlus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    PlusEq,
    /// `-=`, `*=` and the like: a compound assignment other than `+=`.
    OpEq(crate::ast::BinOp),
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Bang,
    Amp,
    AmpAmp,
    Pipe,
    PipePipe,
    Caret,
    Tilde,
    Shl,
    Shr,

    Eof,
}

/// Every keyword, as written, and its token: what the lexer reads, the
/// highlighter marks and a message names.
pub(crate) static KEYWORDS: [(&str, Tok); 14] = [
    ("kernel", Tok::Kernel),
    ("fn", Tok::Fn),
    ("let", Tok::Let),
    ("var", Tok::Var),
    ("if", Tok::If),
    ("else", Tok::Else),
    ("for", Tok::For),
    ("in", Tok::In),
    ("while", Tok::While),
    ("break", Tok::Break),
    ("continue", Tok::Continue),
    ("return", Tok::Return),
    ("true", Tok::True),
    ("false", Tok::False),
];

impl Tok {
    /// The keyword `text` spells, if it spells one.
    pub fn keyword(text: &str) -> Option<Tok> {
        KEYWORDS.iter().find(|(k, _)| *k == text).map(|(_, tok)| tok.clone())
    }

    /// The keyword's spelling, if this token is a keyword.
    pub fn keyword_text(&self) -> Option<&'static str> {
        KEYWORDS.iter().find(|(_, tok)| tok == self).map(|&(k, _)| k)
    }
}

impl std::fmt::Display for Tok {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Tok::Ident(s) => return write!(f, "identifier `{s}`"),
            Tok::Int(n) => return write!(f, "integer `{n}`"),
            Tok::Float(x) => return write!(f, "float `{x}`"),
            Tok::LParen => "`(`",
            Tok::RParen => "`)`",
            Tok::LBrace => "`{`",
            Tok::RBrace => "`}`",
            Tok::LBracket => "`[`",
            Tok::RBracket => "`]`",
            Tok::Comma => "`,`",
            Tok::Colon => "`:`",
            Tok::Semicolon => "end of statement",
            Tok::At => "`@`",
            Tok::Dot => "`.`",
            Tok::Plus => "`+`",
            Tok::ColonPlus => "`:+`",
            Tok::Minus => "`-`",
            Tok::Star => "`*`",
            Tok::Slash => "`/`",
            Tok::Percent => "`%`",
            Tok::Eq => "`=`",
            Tok::PlusEq => "`+=`",
            Tok::OpEq(op) => return write!(f, "`{}=`", op.symbol()),
            Tok::EqEq => "`==`",
            Tok::NotEq => "`!=`",
            Tok::Lt => "`<`",
            Tok::Le => "`<=`",
            Tok::Gt => "`>`",
            Tok::Ge => "`>=`",
            Tok::Bang => "`!`",
            Tok::Amp => "`&`",
            Tok::AmpAmp => "`&&`",
            Tok::Pipe => "`|`",
            Tok::PipePipe => "`||`",
            Tok::Caret => "`^`",
            Tok::Tilde => "`~`",
            Tok::Shl => "`<<`",
            Tok::Shr => "`>>`",
            Tok::Eof => "end of file",
            keyword => {
                let text = keyword.keyword_text().expect("every other token is a keyword");
                return write!(f, "`{text}`");
            }
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    /// The token's bytes in the source, `hi` exclusive.
    pub lo: u32,
    pub hi: u32,
}

impl Token {
    pub fn span(&self) -> crate::diag::Span {
        crate::diag::Span::new(self.lo, self.hi)
    }
}
