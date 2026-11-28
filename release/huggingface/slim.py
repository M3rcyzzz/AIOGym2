"""The code a release ships: only what running the network needs, without docstrings or comments.

slim(root, files, roots) -> {relative path: source}. Each module is parsed, its docstrings and
`if __name__ == '__main__':` blocks are dropped, and of its top-level functions and classes only those
reachable from `roots` are kept: a name is reachable when a kept statement refers to it, by its bare name
in its own module, through `from module import name`, or as `alias.name` of `import module as alias`.
Every other top-level statement (imports, constants) is kept. The source is written back with
ast.unparse, so comments go too. export.py checks the result bit for bit against the checkpoint.
"""
import ast
from pathlib import Path


def _mod(rel):
    return '.'.join(Path(rel).with_suffix('').parts).removesuffix('.__init__')


def _strip_docs(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                b.pop(0)
                if not b:
                    b.append(ast.Pass())


def _is_main(n):
    t = isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
    return t and n.test.left.id == '__name__'


def slim(root, files, roots):
    """root: the repository; files: module paths relative to it; roots: {module: {names}} that are used from outside."""
    root, mods = Path(root), {_mod(f): f for f in files}
    trees = {m: ast.parse((root/f).read_text(encoding='utf-8')) for m, f in mods.items()}
    defs, alias, imported = {}, {}, {}
    for m, t in trees.items():
        t.body = [n for n in t.body if not _is_main(n)]
        defs[m] = {n.name: n for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        alias[m], imported[m] = {}, {}
        pkg = m.rsplit('.', 1)[0] if '.' in m else m
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                for a in n.names:
                    alias[m][a.asname or a.name] = a.name
            elif isinstance(n, ast.ImportFrom) and n.module:
                src = n.module if not n.level else pkg+'.'+n.module
                for a in n.names:
                    full = src+'.'+a.name
                    if full in mods:                               # from aio import formula as FM
                        alias[m][a.asname or a.name] = full
                    else:                                          # from aio.sim.model import KNOTS
                        imported[m][a.asname or a.name] = (src, a.name)

    def refs(m, node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and alias[m].get(n.value.id) in mods:
                out.add((alias[m][n.value.id], n.attr))
            elif isinstance(n, ast.ImportFrom) and n.module:                 # an import keeps what it names
                src = n.module if not n.level else m.rsplit('.', 1)[0]+'.'+n.module
                out |= {(src, a.name) for a in n.names}
            elif isinstance(n, ast.Name):
                if n.id in defs[m]:
                    out.add((m, n.id))
                elif n.id in imported[m]:
                    out.add(imported[m][n.id])
        return out

    todo = {(m, x) for m, xs in roots.items() for x in xs}
    for m, t in trees.items():                                     # the module-level statements are always kept
        for n in t.body:
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                todo |= refs(m, n)
    keep = set()
    while todo:
        m, x = todo.pop()
        if (m, x) in keep or m not in defs or x not in defs[m]:
            continue
        keep.add((m, x))
        todo |= refs(m, defs[m][x])
    out = {}
    for m, t in trees.items():
        t.body = [n for n in t.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or (m, n.name) in keep]
        _strip_docs(t)
        out[mods[m]] = ast.unparse(t)+'\n'
    return out
