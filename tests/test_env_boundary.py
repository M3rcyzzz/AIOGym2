"""Hathaway boundary: the environment layer (aio/sim) knows no network and no trainer. It may import aio.sim,
aio.interface (the packet spec) and aio.scale (the score), which move into it; anything else of aio is
training side. Zero inverted imports are allowed (LEFT == set()), and dynamic/smuggled imports of training
modules are also forbidden inside aio/sim."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_OK = {'aio.sim', 'aio.interface', 'aio.scale'}
LEFT = set()
DYNAMIC_OK = {'plugin.py', 'plants.py'}
_AIO_MOD = re.compile(r'\baio(?:\.[A-Za-z_]\w*)+')


def _is_training_mod(name):
    return (name == 'aio' or name.startswith('aio.')) and not any(name == ok or name.startswith(ok+'.') for ok in ENV_OK)


def _imports(path, rel):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    pkg_parts = ('aio', 'sim', *rel.parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield from ([f'{node.module}.{a.name}' for a in node.names] if node.module == 'aio' else [node.module])
            elif node.level > 0:
                base = pkg_parts[:len(pkg_parts)-(node.level-1)]
                mod = '.'.join((*base, node.module)) if node.module else '.'.join(base)
                yield from ([f'{mod}.{a.name}' for a in node.names] if mod == 'aio' else [mod])


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _call_name(fn):
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ''


def test_the_environment_imports_nothing_of_the_training_side():
    sim = ROOT/'aio'/'sim'
    found = set()
    for f in sim.rglob('*.py'):
        rel = f.relative_to(sim)
        for m in _imports(f, rel):
            if _is_training_mod(m):
                found.add((rel.as_posix(), '.'.join(m.split('.')[:2])))
    assert LEFT == set()
    assert found <= LEFT, f'imports of the training side in aio/sim: {sorted(found-LEFT)}'


def test_no_dynamic_or_smuggled_training_imports_in_sim():
    sim = ROOT/'aio'/'sim'
    bad_dyn, bad_str = [], []
    for f in sim.rglob('*.py'):
        rel = f.relative_to(sim).as_posix()
        tree = ast.parse(f.read_text(encoding='utf-8'))
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name == '__import__':
                    bad_dyn.append((rel, '__import__'))
                elif name == 'import_module' and rel not in DYNAMIC_OK:
                    bad_dyn.append((rel, 'import_module'))
                elif name == 'import_module' and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    if _is_training_mod(node.args[0].value):
                        bad_dyn.append((rel, f'import_module({node.args[0].value!r})'))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                for m in _AIO_MOD.findall(node.value):
                    if _is_training_mod(m):
                        bad_str.append((rel, m, node.value))
    assert not bad_dyn, f'disallowed dynamic imports in aio/sim: {bad_dyn}'
    assert not bad_str, f'training-side module string smuggled in aio/sim code: {bad_str}'
