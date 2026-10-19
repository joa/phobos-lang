"""The front end: a Python module's `@ph.kernel` and `@ph.fn` definitions
written out as Phobos, with a map from every emitted construct back to the
Python it was written from.

Nothing is run: names resolve through the module's imports of
`phobos.lang`, read from its source. Each Python construct is a spelling of
one Phobos construct (README.md lists them); one with none is an error at
the Python line, and so is a spelling whose Phobos meaning would differ
from Python's where the source alone tells them apart. Where only types
tell them apart, the map carries the spelling and the compiler decides:
Python's `/` and `//` both emit Phobos `/`.
"""

import ast
import difflib
import json
import re
from dataclasses import dataclass, field

LANG = "phobos.lang"

# Names `phobos.lang` gives a meaning the front end spells itself; every
# other `ph.<name>` is a built-in, a standard library function or an
# attribute, written as is.
SCALARS = ("f16", "bf16", "f32", "f64", "i8", "i32", "i64", "u8", "u16", "u32", "bool")
COLLECTIVES = ("sum", "max", "min", "argmax", "any", "all", "reduce")
SHUFFLE_MODES = {"from_": "from", "xor": "xor", "down": "down", "up": "up"}
CTX_FIELDS = ("block", "blocks", "threads", "warps", "lanes", "thread", "warp", "lane")

# Python operators and the Phobos operator each spells.
BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "/", ast.Mod: "%",
    ast.LShift: "<<", ast.RShift: ">>", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
}
PY_BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
    ast.LShift: "<<", ast.RShift: ">>", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.Pow: "**", ast.MatMult: "@",
}
COMPARES = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
SPELLINGS = {ast.Div: "true_div", ast.FloorDiv: "floor_div"}

# What a Python construct with no Phobos counterpart is called in an error.
CONSTRUCTS = {
    "IfExp": "conditional expression", "Lambda": "lambda", "ListComp": "list comprehension",
    "SetComp": "set comprehension", "DictComp": "dict comprehension", "GeneratorExp": "generator expression",
    "List": "list", "Tuple": "tuple", "Dict": "dict", "Set": "set", "Starred": "`*` expression",
    "NamedExpr": "`:=`", "JoinedStr": "f-string", "Await": "`await`", "Yield": "`yield`", "YieldFrom": "`yield from`",
    "With": "`with`", "Try": "`try`", "TryStar": "`try`", "Raise": "`raise`", "Assert": "`assert`",
    "Delete": "`del`", "Global": "`global`", "Nonlocal": "`nonlocal`", "Match": "`match`",
    "FunctionDef": "nested function", "AsyncFunctionDef": "nested function", "ClassDef": "class",
    "Import": "`import`", "ImportFrom": "`import`", "AsyncFor": "`async for`", "AsyncWith": "`async with`",
}


def construct(node):
    return CONSTRUCTS.get(type(node).__name__, f"`{type(node).__name__}`")


# Phobos precedence, loosest first: what decides the parentheses written.
PREC = {"||": 1, "&&": 2, "==": 3, "!=": 3, "<": 4, "<=": 4, ">": 4, ">=": 4, "|": 5, "^": 6, "&": 7,
        "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10}
UNARY = 11
ATOM = 12


class FrontendError(Exception):
    """Errors at Python lines, each a diagnostic for the compiler to render."""

    def __init__(self, path, text, diagnostics):
        self.path, self.text, self.diagnostics = path, text, diagnostics
        super().__init__("\n".join(d["message"] for d in diagnostics))


@dataclass
class Program:
    """The Phobos a module spells, and where each piece came from."""

    path: str
    python: str
    text: str
    map: list
    spellings: list
    kernels: list = field(default_factory=list)

    def origin_json(self):
        return json.dumps({
            "file": self.path,
            "text": self.python,
            "map": self.map,
            "spellings": self.spellings,
        })


class Bail(Exception):
    pass


class Source:
    """The module's text, as bytes, so AST columns turn into byte offsets."""

    def __init__(self, text):
        self.text = text
        self.data = text.encode()
        self.line_starts = [0]
        for i, b in enumerate(self.data):
            if b == 10:
                self.line_starts.append(i + 1)

    def offset(self, line, col):
        return self.line_starts[line - 1] + col

    def span(self, node):
        return (self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset))

    def find(self, needle, lo, hi):
        """The span of `needle` in `[lo, hi)`, outside comments."""
        text = self.data[lo:hi].decode()
        at = 0
        while True:
            i = text.find(needle, at)
            hash_ = text.find("#", at)
            if hash_ != -1 and (i == -1 or hash_ < i):
                nl = text.find("\n", hash_)
                if nl == -1:
                    return None
                at = nl
                continue
            if i == -1:
                return None
            lo_b = lo + len(text[:i].encode())
            return (lo_b, lo_b + len(needle.encode()))


class Names:
    """What the module binds `phobos.lang` and its names to."""

    def __init__(self, module):
        self.modules = set()
        self.direct = {}
        self.bound = set()
        for node in module.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == LANG and a.asname:
                        self.modules.add(a.asname)
                    self.bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    local = a.asname or a.name
                    if node.module == "phobos" and a.name == "lang":
                        self.modules.add(local)
                    elif node.module == LANG:
                        self.direct[local] = a.name
                    self.bound.add(local)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.bound.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            self.bound.add(n.id)

    def lang(self, node, local=()):
        """The `phobos.lang` name `node` spells, as a dotted path, or None."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id in local:
            return None
        if node.id in self.modules:
            return ".".join(reversed(parts)) or None
        if node.id in self.direct:
            return ".".join([self.direct[node.id], *reversed(parts)])
        return None


class Emitter:
    """Phobos text written piece by piece, each construct mapped back."""

    def __init__(self, src):
        self.src = src
        self.parts = []
        self.at = 0
        self.map = []
        self.spellings = []

    def write(self, text):
        self.parts.append(text)
        self.at += len(text)

    def mapped(self, py_span, text):
        lo = self.at
        self.write(text)
        self.map.append([lo, self.at, py_span[0], py_span[1]])

    def open(self):
        return self.at

    def close(self, lo, py_span):
        self.map.append([lo, self.at, py_span[0], py_span[1]])

    def text(self):
        return "".join(self.parts)


class Translator:
    def __init__(self, path, text):
        self.path, self.src = path, Source(text)
        self.errors = []
        self.module = ast.parse(text, filename=path)
        self.names = Names(self.module)
        self.out = Emitter(self.src)
        self.fns = {}
        for node in self.module.body:
            if isinstance(node, ast.FunctionDef) and self.decorated(node, "fn"):
                self.fns[node.name] = node

    # Diagnostics.

    def error(self, node_or_span, message, label="", help=None, notes=()):
        span = node_or_span if isinstance(node_or_span, tuple) else self.src.span(node_or_span)
        d = {"message": message, "labels": [[span[0], span[1], label, True]], "notes": list(notes), "help": [help] if help else []}
        self.errors.append(d)
        raise Bail()

    def guard(self, f, *args):
        try:
            f(*args)
        except Bail:
            pass

    # Module structure.

    def decorated(self, node, what):
        return any(self.names.lang(d.func if isinstance(d, ast.Call) else d) == what for d in node.decorator_list)

    def program(self, only=None):
        """The module's functions, then its kernels in order; with `only`,
        that kernel alone."""
        kernels = []
        for node in self.module.body:
            if isinstance(node, ast.FunctionDef) and node.name in self.fns:
                self.guard(self.function, node)
        for node in self.module.body:
            if isinstance(node, ast.FunctionDef) and self.decorated(node, "kernel") and only in (None, node.name):
                kernels.append(node.name)
                self.guard(self.kernel, node)
        if self.errors:
            raise FrontendError(self.path, self.src.text, self.errors)
        return Program(self.path, self.src.text, self.out.text(), self.out.map, self.out.spellings, kernels)

    def name_span(self, node):
        lo, hi = self.src.span(node)
        found = self.src.find(f"def {node.name}", lo, hi)
        if found is None:
            return (lo, hi)
        return (found[1] - len(node.name.encode()), found[1])

    def check_signature(self, node, typed):
        a = node.args
        if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults or a.kw_defaults:
            self.error(node.args, "a Phobos parameter list is plain names", "not a plain parameter",
                       help="drop defaults, `*args`, `**kwargs` and keyword-only parameters")
        if node.returns is not None:
            self.error(node.returns, "a Phobos kernel or `fn` returns nothing by its signature", "a return annotation")
        for arg in a.args:
            if typed and arg.annotation is None:
                self.error(arg, f"kernel parameter `{arg.arg}` needs a type", "no annotation",
                           help="annotate it, as `x: ph.tensor[ph.f32][N]` or `alpha: ph.f32`")
            if not typed and arg.annotation is not None:
                self.error(arg.annotation, "a `fn` parameter takes no type", "the argument's type is the caller's")

    def body_of(self, node):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        return body

    def kernel(self, node):
        self.check_signature(node, typed=True)
        self.local = {a.arg for a in node.args.args}
        self.uses = None
        for d in node.decorator_list:
            call = d if isinstance(d, ast.Call) else None
            name = self.names.lang(call.func if call else d)
            if name in ("kernel", None):
                if name is None:
                    self.error(d, "a kernel's decorators are `phobos.lang` attributes", "not one",
                               help="spell Phobos attributes as `@ph.name(...)`")
                continue
            self.attribute(d, name, call)
        out = self.out
        out.write("kernel ")
        out.mapped(self.name_span(node), node.name)
        out.write("(")
        for i, arg in enumerate(node.args.args):
            if i:
                out.write(", ")
            lo = out.open()
            at = self.src.span(arg)[0]
            out.mapped((at, at + len(arg.arg.encode())), arg.arg)
            out.write(": ")
            self.type(arg.annotation)
            out.close(lo, self.src.span(arg))
        out.write(") {\n")
        self.block(self.body_of(node), 1)
        out.write("}\n\n")

    def attribute(self, d, name, call):
        out = self.out
        lo = out.open()
        out.write("@")
        func = call.func if call else d
        span = self.src.span(func)
        out.mapped((span[1] - len(name.split(".")[-1].encode()), span[1]), name)
        if call is not None:
            out.write("(")
            first = True
            for a in call.args:
                if not first:
                    out.write(", ")
                first = False
                self.attr_literal(a)
            for kw in call.keywords:
                if not first:
                    out.write(", ")
                first = False
                if kw.arg is None:
                    self.error(kw, "an attribute takes no `**` arguments", "")
                klo = out.open()
                value = kw.value
                if isinstance(value, (ast.List, ast.Tuple)):
                    out.write(f"{kw.arg} in [")
                    for i, e in enumerate(value.elts):
                        if i:
                            out.write(", ")
                        if not (isinstance(e, ast.Constant) and type(e.value) is int):
                            self.error(e, "an `@autotune` choice is an integer", "not one")
                        out.mapped(self.src.span(e), str(e.value))
                    out.write("]")
                else:
                    out.write(f"{kw.arg} = ")
                    self.attr_literal(value)
                out.close(klo, self.src.span(kw))
            out.write(")")
        out.close(lo, self.src.span(d))
        out.write("\n")

    def attr_literal(self, e):
        if isinstance(e, ast.Constant) and isinstance(e.value, bool):
            self.out.mapped(self.src.span(e), "true" if e.value else "false")
        elif isinstance(e, ast.Constant) and isinstance(e.value, str) and re.fullmatch(r"[A-Za-z_]\w*", e.value):
            self.out.mapped(self.src.span(e), e.value)
        elif isinstance(e, ast.Constant) and type(e.value) in (int, float):
            self.out.mapped(self.src.span(e), self.number(e))
        elif isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant):
            self.out.mapped(self.src.span(e), "-" + self.number(e.operand))
        else:
            self.error(e, "an attribute argument is a literal", "not one",
                       help="write a number, `True`/`False`, or a name as a string, `\"off\"`")

    def function(self, node):
        self.check_signature(node, typed=False)
        uses = ()
        for d in node.decorator_list:
            call = d if isinstance(d, ast.Call) else None
            name = self.names.lang(call.func if call else d)
            if name != "fn":
                self.error(d, "a `fn` takes no attributes", "")
            for kw in call.keywords if call else ():
                if kw.arg != "uses" or not isinstance(kw.value, (ast.Tuple, ast.List)) or not all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in kw.value.elts
                ):
                    self.error(kw, "`@ph.fn` takes `uses`, the names its body reads from its caller, as strings", "")
                uses = tuple(e.value for e in kw.value.elts)
        params = [a.arg for a in node.args.args]
        for name in uses:
            if name in self.names.bound:
                self.error(d, f"`{name}` is both a name of this module and one `{node.name}` reads from its caller",
                           f"`{name}` is read where `{node.name}` is called",
                           notes=[f"Python would read the module's `{name}`; Phobos reads the caller's"],
                           help="rename one of them")
        self.local = set(params)
        self.uses = set(uses)
        body = self.body_of(node)
        out = self.out
        out.write("fn ")
        out.mapped(self.name_span(node), node.name)
        out.write("(" + ", ".join(params) + ")")
        if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
            out.write(" = ")
            self.expr(body[0].value, 0)
            out.write("\n\n")
            return
        for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(stmt, ast.Return):
                self.error(stmt, "a statement `fn` does not return", "a `return` in a statement body",
                           help="a body of one `return e` is an expression `fn`; a statement `fn` ending in `e` gives `e`")
        out.write(" {\n")
        self.block(body, 1)
        out.write("}\n\n")

    # Types.

    def type(self, node):
        out = self.out
        lo = out.open()
        name = self.names.lang(node, self.local)
        if name in SCALARS:
            out.write(name)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript):
            kind = self.names.lang(node.value.value, self.local)
            elem = self.names.lang(node.value.slice, self.local)
            if kind not in ("tensor", "tile"):
                self.unknown_type(node.value.value, kind, ("tensor", "tile"))
            if elem not in SCALARS:
                self.unknown_type(node.value.slice, elem, SCALARS)
            out.write(f"{kind}<{elem}>[")
            dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for i, d in enumerate(dims):
                if i:
                    out.write(", ")
                if isinstance(d, ast.Name):
                    self.free(d)
                    out.mapped(self.src.span(d), d.id)
                elif isinstance(d, ast.Constant) and type(d.value) is int:
                    out.mapped(self.src.span(d), str(d.value))
                else:
                    self.error(d, "a dimension is a name or an integer", "not one")
            out.write("]")
        else:
            self.error(node, "a Phobos type is `ph.tensor[E][dims]`, `ph.tile[E][dims]` or a scalar `ph.E`", "not one",
                       notes=["a module of kernels starts `from __future__ import annotations`, so Python never evaluates the dimensions"])
        out.close(lo, self.src.span(node))

    def unknown_type(self, node, name, known):
        near = difflib.get_close_matches(name or "", known, n=1)
        self.error(node, f"unknown type `ph.{name}`" if name else "a Phobos type is `ph.tensor[E][dims]`, `ph.tile[E][dims]` or a scalar `ph.E`",
                   "not a type", help=f"a similar type exists: `ph.{near[0]}`" if near else None)

    # Statements.

    def block(self, body, depth):
        for stmt in body:
            self.guard(self.stmt, stmt, depth)

    def indent(self, depth):
        self.out.write("    " * depth)

    def stmt(self, s, depth):
        out = self.out
        if isinstance(s, ast.Pass):
            return
        self.indent(depth)
        lo = out.open()
        if isinstance(s, ast.AnnAssign):
            self.declaration(s)
        elif isinstance(s, ast.Assign):
            if len(s.targets) != 1:
                self.error(s, "an assignment has one target", "chained", help="assign each in its own statement")
            self.target(s.targets[0])
            out.write(" = ")
            self.expr(s.value, 0)
        elif isinstance(s, ast.AugAssign):
            op = type(s.op)
            if op not in BINOPS:
                self.error(s, f"Phobos has no `{PY_BINOPS.get(op, '?')}=`", "",
                           help="write `ph.pow(a, b)`" if op is ast.Pow else "write `ph.dot(a, b)`")
            self.target(s.target)
            out.write(" ")
            olo = out.at
            out.write(BINOPS[op] + "=")
            py_op = self.src.find(PY_BINOPS[op] + "=", self.src.span(s.target)[1], self.src.span(s.value)[0])
            if py_op:
                out.map.append([olo, out.at, py_op[0], py_op[1]])
            if op in SPELLINGS:
                out.spellings.append([olo, out.at, SPELLINGS[op]])
            out.write(" ")
            self.expr(s.value, 0)
        elif isinstance(s, ast.For):
            self.loop(s, depth, lo)
            return
        elif isinstance(s, ast.While):
            if s.orelse:
                self.error(s, "a `while` has no `else`", "")
            out.write("while ")
            self.expr(s.test, 0)
            out.write(" {\n")
            out.close(lo, self.header(s))
            self.block(s.body, depth + 1)
            self.indent(depth)
            out.write("}\n")
            return
        elif isinstance(s, ast.If):
            self.branch(s, depth, lo)
            return
        elif isinstance(s, (ast.Break, ast.Continue)):
            out.write("break" if isinstance(s, ast.Break) else "continue")
        elif isinstance(s, ast.Return):
            if s.value is not None:
                self.error(s, "a kernel returns nothing", "a value", help="write the result to a tensor parameter")
            out.write("return")
        elif isinstance(s, ast.Expr):
            self.expr(s.value, 0)
        else:
            self.error(s, f"Phobos has no {construct(s)}", "")
        out.close(lo, self.src.span(s))
        out.write("\n")

    def header(self, s):
        """A compound statement's span, up to the colon of its first line."""
        lo, hi = self.src.span(s)
        first = s.body[0]
        return (lo, self.src.span(first)[0]) if first else (lo, hi)

    def declaration(self, s):
        out = self.out
        if not s.simple or not isinstance(s.target, ast.Name):
            self.error(s.target, "a declaration names one variable", "not a plain name")
        ann = s.annotation
        kind = self.names.lang(ann, self.local)
        ty = None
        if kind is None and isinstance(ann, ast.Subscript):
            kind = self.names.lang(ann.value, self.local)
            ty = ann.slice
        if kind not in ("let", "var"):
            self.error(ann, "a declaration says `ph.let` or `ph.var`", "neither",
                       notes=["`let` binds a value or a view of a tensor; `var` a mutable value, a tile, or a copy of a slice into one"],
                       help="write `x: ph.let = ...` or `x: ph.var[T] = ...`")
        if kind == "let" and s.value is None:
            self.error(s, "a `ph.let` needs a value", "no `=`")
        if kind == "var" and s.value is None and ty is None:
            self.error(s, "a `ph.var` with no value needs a type", "", help="write `t: ph.var[ph.tile[ph.f32][R, C]]`")
        out.write(f"{kind} ")
        name = s.target.id
        out.mapped(self.src.span(s.target), name)
        if ty is not None:
            out.write(": ")
            self.type(ty)
        if s.value is not None:
            out.write(" = ")
            self.expr(s.value, 0)
        self.local.add(name)

    def target(self, t):
        if isinstance(t, ast.Name):
            self.out.mapped(self.src.span(t), t.id)
        elif isinstance(t, ast.Subscript):
            self.expr(t, 0)
        else:
            self.error(t, "an assignment's target is a name or a subscript of one", "not one")

    def loop(self, s, depth, lo):
        out = self.out
        if s.orelse:
            self.error(s, "a `for` has no `else`", "")
        if not isinstance(s.target, ast.Name):
            self.error(s.target, "a loop variable is one name", "not one")
        it, unroll = s.iter, None
        if isinstance(it, ast.Call) and self.names.lang(it.func, self.local) == "unroll":
            if len(it.args) != 2 or it.keywords:
                self.error(it, "`ph.unroll(n, range(..))` takes the count and the range", "")
            unroll, it = it.args
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and 1 <= len(it.args) <= 3 and not it.keywords):
            self.error(it, "a Phobos loop runs over `range(lo, hi[, step])`", "not a range",
                       help="unroll it with `ph.unroll(n, range(lo, hi))`")
        out.write("for")
        if unroll is not None:
            out.write("<unroll ")
            self.expr(unroll, 0)
            out.write(">")
        out.write(" ")
        out.mapped(self.src.span(s.target), s.target.id)
        out.write(" in range(")
        args = it.args
        if len(args) == 1:
            out.write("0, ")
        for i, a in enumerate(args):
            if i:
                out.write(", ")
            self.expr(a, 0)
        out.write(") {\n")
        out.close(lo, self.header(s))
        self.local.add(s.target.id)
        self.block(s.body, depth + 1)
        self.indent(depth)
        out.write("}\n")

    def branch(self, s, depth, lo):
        out = self.out
        out.write("if ")
        self.expr(s.test, 0)
        out.write(" {\n")
        out.close(lo, self.header(s))
        self.block(s.body, depth + 1)
        self.indent(depth)
        out.write("}")
        if s.orelse:
            if len(s.orelse) == 1 and isinstance(s.orelse[0], ast.If):
                out.write(" else ")
                inner = out.open()
                self.branch(s.orelse[0], depth, inner)
                return
            out.write(" else {\n")
            self.block(s.orelse, depth + 1)
            self.indent(depth)
            out.write("}")
        out.write("\n")

    # Expressions.

    def free(self, name):
        """A name a `fn` reads that its caller binds must be in `uses`."""
        if self.uses is None or name.id in self.local or name.id in self.uses:
            return
        if name.id in self.fns:
            return
        self.error(name, f"`{name.id}` is read from where the `fn` is called", "not a parameter or a name of its own",
                   notes=["a name a Phobos `fn` does not bind resolves at its call site"],
                   help=f"list it: `@ph.fn(uses=(\"{name.id}\",))`")

    def number(self, c):
        v = c.value
        if type(v) is int:
            return str(v)
        if v != v or v in (float("inf"), float("-inf")):
            self.error(c, "a float literal is finite", "")
        # As written, where Phobos reads it the same: digits, a point and
        # an exponent, no `_` and no leading point.
        lo, hi = self.src.span(c)
        written = self.src.data[lo:hi].decode()
        if re.fullmatch(r"\d+(\.\d*)?([eE][+-]?\d+)?", written) and ("." in written or "e" in written.lower()):
            return written
        text = repr(v)
        if "e" not in text and "." not in text:
            text += ".0"
        return text.replace("e+", "e")

    def expr(self, e, prec):
        """Writes `e`, parenthesized if it binds looser than `prec`."""
        out = self.out
        mine = self.precedence(e)
        paren = mine < prec
        if paren:
            out.write("(")
        lo = out.open()
        self.expr_inner(e)
        out.close(lo, self.src.span(e))
        if paren:
            out.write(")")

    def precedence(self, e):
        if isinstance(e, ast.BoolOp):
            return 1 if isinstance(e.op, ast.Or) else 2
        if isinstance(e, ast.Compare):
            return 2 if len(e.ops) > 1 else PREC[COMPARES.get(type(e.ops[0]), "==")]
        if isinstance(e, ast.BinOp):
            return PREC.get(BINOPS.get(type(e.op), "+"), 9)
        if isinstance(e, ast.UnaryOp):
            if isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant):
                return UNARY
            return UNARY
        return ATOM

    def expr_inner(self, e):
        out = self.out
        if isinstance(e, ast.Constant):
            if isinstance(e.value, bool):
                out.write("true" if e.value else "false")
            elif type(e.value) in (int, float):
                out.write(self.number(e))
            else:
                self.error(e, f"Phobos has no {type(e.value).__name__} literal", "")
        elif isinstance(e, ast.Name):
            if e.id in self.names.direct and e.id not in self.local:
                self.error(e, f"`{e.id}` is a `phobos.lang` name, not a value", "")
            self.free(e)
            out.write(e.id)
        elif isinstance(e, ast.Attribute):
            name = self.names.lang(e, self.local)
            if name and name.startswith("ctx."):
                out.write(name)
            else:
                self.error(e, "an attribute reads `ph.ctx` and nothing else", "",
                           notes=["`ctx` has the fields " + ", ".join(f"`{f}`" for f in CTX_FIELDS)])
        elif isinstance(e, ast.BinOp):
            op = type(e.op)
            if op not in BINOPS:
                self.error(e, f"Phobos has no `{PY_BINOPS.get(op, '?')}`", "",
                           help="write `ph.pow(a, b)`" if op is ast.Pow else "write `ph.dot(a, b)`")
            sym = BINOPS[op]
            self.expr(e.left, PREC[sym])
            out.write(" ")
            olo = out.at
            out.write(sym)
            py_op = self.src.find(PY_BINOPS[op], self.src.span(e.left)[1], self.src.span(e.right)[0])
            if py_op:
                out.map.append([olo, out.at, py_op[0], py_op[1]])
            if op in SPELLINGS:
                out.spellings.append([olo, out.at, SPELLINGS[op]])
            out.write(" ")
            self.expr(e.right, PREC[sym] + 1)
        elif isinstance(e, ast.UnaryOp):
            op = type(e.op)
            sym = {ast.USub: "-", ast.Not: "!", ast.Invert: "~"}.get(op)
            if sym is None:
                self.error(e, "Phobos has no unary `+`", "", notes=["in a subscript, `s:+n` is the span of `n` from `s`"])
            out.write(sym)
            self.expr(e.operand, UNARY)
        elif isinstance(e, ast.BoolOp):
            sym = "||" if isinstance(e.op, ast.Or) else "&&"
            for i, v in enumerate(e.values):
                if i:
                    out.write(f" {sym} ")
                self.expr(v, PREC[sym] + (1 if i else 0))
        elif isinstance(e, ast.Compare):
            self.compare(e)
        elif isinstance(e, ast.Call):
            self.call(e)
        elif isinstance(e, ast.Subscript):
            self.expr(e.value, ATOM)
            out.write("[")
            subs = e.slice.elts if isinstance(e.slice, ast.Tuple) else [e.slice]
            for i, s in enumerate(subs):
                if i:
                    out.write(", ")
                self.subscript(s)
            out.write("]")
        else:
            self.error(e, f"Phobos has no {construct(e)}", "",
                       help="a select is `ph.where(c, a, b)`" if isinstance(e, ast.IfExp) else None)

    def compare(self, e):
        out = self.out
        for op in e.ops:
            if type(op) not in COMPARES:
                self.error(e, f"Phobos has no `{type(op).__name__}` comparison", "")
        # `a < b < c` is `a < b && b < c`; `b` is written twice, which is
        # safe since a Phobos expression has no side effects.
        operands = [e.left, *e.comparators]
        for i, op in enumerate(e.ops):
            if i:
                out.write(" && ")
            sym = COMPARES[type(op)]
            single = len(e.ops) == 1
            self.expr(operands[i], PREC[sym] if single else PREC["&&"] + 1)
            out.write(f" {sym} ")
            self.expr(operands[i + 1], PREC[sym] + 1 if single else PREC["&&"] + 1)

    def subscript(self, s):
        out = self.out
        lo = out.open()
        if isinstance(s, ast.Slice):
            if s.step is not None:
                self.error(s.step, "a Phobos slice has no step", "")
            span = isinstance(s.upper, ast.UnaryOp) and isinstance(s.upper.op, ast.UAdd)
            if span and s.lower is None:
                self.error(s, "a span `s:+n` needs its start", "")
            if s.lower is not None:
                self.expr(s.lower, 0)
                out.write(" ")
            if span:
                out.write(":+ ")
                self.expr(s.upper.operand, 0)
            else:
                out.write(":")
                if s.upper is not None:
                    out.write(" ")
                    self.expr(s.upper, 0)
            out.close(lo, self.src.span(s))
        else:
            self.expr(s, 0)

    def call(self, e):
        out = self.out
        if isinstance(e.func, ast.Name) and e.func.id not in self.local:
            name = e.func.id
            if name in self.fns:
                callee = name
            elif name in ("min", "max", "abs", "sum", "round", "pow", "any", "all", "len", "int", "float", "bool"):
                self.error(e.func, f"`{name}` is Python's", "", help=f"Phobos's is `ph.{name}`" if name not in ("len", "int", "float", "bool") else None)
            else:
                callee = name
        else:
            callee = self.names.lang(e.func, self.local)
            if callee is None:
                self.error(e.func, "a call names a `phobos.lang` built-in or a `@ph.fn` of this module", "neither")
        span = self.src.span(e.func)
        tail = callee.split(".")[-1]
        name_span = (span[1] - len(tail.encode()), span[1])
        kws = {k.arg: k for k in e.keywords}
        if None in kws:
            self.error(e, "a call takes no `**` arguments", "")
        if callee in COLLECTIVES and "over" in kws:
            settings = [("over", kws.pop("over").value)]
            if "width" in kws:
                settings.append(("width", kws.pop("width").value))
            self.config_call(callee, name_span, settings, e.args, kws)
        elif callee == "shfl":
            modes = [k for k in kws if k in SHUFFLE_MODES]
            if len(modes) != 1:
                self.error(e, "`ph.shfl` takes one of `from_`, `xor`, `down` and `up`", "")
            settings = [(SHUFFLE_MODES[modes[0]], kws.pop(modes[0]).value)]
            if "width" in kws:
                settings.append(("width", kws.pop("width").value))
            self.config_call("shfl", name_span, settings, e.args, kws)
        else:
            if kws:
                k = next(iter(kws.values()))
                self.error(k, f"`{callee}` takes no keyword arguments", "",
                           notes=["only a collective's `over` and `width` and a shuffle's mode are keywords"])
            out.mapped(name_span, callee)
            self.args(e.args)

    def config_call(self, name, name_span, settings, args, rest):
        out = self.out
        if rest:
            k = next(iter(rest.values()))
            self.error(k, f"`ph.{name}` has no setting `{k.arg}`", "")
        out.mapped(name_span, name)
        out.write("<")
        for i, (word, value) in enumerate(settings):
            if i:
                out.write(", ")
            out.write(word + " ")
            self.expr(value, ATOM if word == "over" else PREC["+"])
        out.write(">")
        self.args(args)

    def args(self, args):
        out = self.out
        out.write("(")
        for i, a in enumerate(args):
            if i:
                out.write(", ")
            if isinstance(a, ast.Starred):
                self.error(a, "a call takes no `*` arguments", "")
            self.expr(a, 0)
        out.write(")")


def translate(path, text, only=None):
    """The Phobos program `text`, the module at `path`, spells: its `fn`s and
    its kernels, or with `only` that kernel alone."""
    try:
        return Translator(path, text).program(only)
    except SyntaxError as e:
        src, lines = Source(text), text.split("\n")
        line = min(max(e.lineno or 1, 1), len(lines))
        col = max((e.offset or 1) - 1, 0)
        lo = src.line_starts[line - 1] + len(lines[line - 1][:col].encode())
        d = {"message": e.msg, "labels": [[lo, lo + 1, "", True]], "notes": [], "help": []}
        raise FrontendError(path, text, [d]) from None
