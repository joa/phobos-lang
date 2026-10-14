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

impl std::fmt::Display for Tok {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Tok::Ident(s) => return write!(f, "identifier `{s}`"),
            Tok::Int(n) => return write!(f, "integer `{n}`"),
            Tok::Float(x) => return write!(f, "float `{x}`"),
            Tok::Kernel => "`kernel`",
            Tok::Let => "`let`",
            Tok::Var => "`var`",
            Tok::If => "`if`",
            Tok::Else => "`else`",
            Tok::For => "`for`",
            Tok::In => "`in`",
            Tok::While => "`while`",
            Tok::Fn => "`fn`",
            Tok::Break => "`break`",
            Tok::Continue => "`continue`",
            Tok::Return => "`return`",
            Tok::True => "`true`",
            Tok::False => "`false`",
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
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
    pub col: u32,
    /// The column just past the token, which never spans lines.
    pub end_col: u32,
    /// The token's bytes in the source, `hi` exclusive.
    pub lo: u32,
    pub hi: u32,
}

impl Token {
    pub fn span(&self) -> crate::diag::Span {
        crate::diag::Span::new(self.lo, self.hi)
    }
}
