#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),

    // keywords
    Kernel,
    Let,
    Var,
    If,
    Else,
    For,
    In,
    While,
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
    At, // @

    // operators
    Plus,
    ColonPlus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    PlusEq,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Bang,

    Eof,
}

impl std::fmt::Display for Tok {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Tok::Ident(s) => return write!(f, "identifier '{s}'"),
            Tok::Int(n) => return write!(f, "integer '{n}'"),
            Tok::Float(x) => return write!(f, "float '{x}'"),
            Tok::Kernel => "'kernel'",
            Tok::Let => "'let'",
            Tok::Var => "'var'",
            Tok::If => "'if'",
            Tok::Else => "'else'",
            Tok::For => "'for'",
            Tok::In => "'in'",
            Tok::While => "'while'",
            Tok::True => "'true'",
            Tok::False => "'false'",
            Tok::LParen => "'('",
            Tok::RParen => "')'",
            Tok::LBrace => "'{'",
            Tok::RBrace => "'}'",
            Tok::LBracket => "'['",
            Tok::RBracket => "']'",
            Tok::Comma => "','",
            Tok::Colon => "':'",
            Tok::Semicolon => "end of statement",
            Tok::At => "'@'",
            Tok::Plus => "'+'",
            Tok::ColonPlus => "':+'",
            Tok::Minus => "'-'",
            Tok::Star => "'*'",
            Tok::Slash => "'/'",
            Tok::Percent => "'%'",
            Tok::Eq => "'='",
            Tok::PlusEq => "'+='",
            Tok::EqEq => "'=='",
            Tok::NotEq => "'!='",
            Tok::Lt => "'<'",
            Tok::Le => "'<='",
            Tok::Gt => "'>'",
            Tok::Ge => "'>='",
            Tok::Bang => "'!'",
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
}
