"""Hathaway boundary: the environment layer (aio/sim) knows no network and no trainer. It may import aio.sim, aio.interface
(the packet spec) and aio.scale (the score), which move into it; anything else of aio is training side. The few imports
that still point the wrong way are listed and go away one by one (docs/hathaway.md, step 2); a new one fails here."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_OK = {'aio.sim', 'aio.interface', 'aio.scale'}
LEFT = {('control.py', 'aio.actor'), ('control.py', 'aio.frame'), ('replay.py', 'aio.net'),
        ('validate/boptest.py', 'aio.net'), ('rates.py', 'aio.tiers'), ('famexam.py', 'aio.net'), ('faults.py', 'aio.net'), ('famexam.py', 'aio.actor'), ('famexam.py', 'aio.frame')}


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield from ([f'{node.module}.{a.name}' for a in node.names] if node.module == 'aio' else [node.module])


def test_the_environment_imports_nothing_of_the_training_side():
    sim = ROOT/'aio'/'sim'
    found = set()
    for f in sim.rglob('*.py'):
        for m in _imports(f):
            if (m == 'aio' or m.startswith('aio.')) and not any(m == ok or m.startswith(ok+'.') for ok in ENV_OK):
                found.add((f.relative_to(sim).as_posix(), '.'.join(m.split('.')[:2])))
    assert found <= LEFT, f'new imports of the training side in aio/sim: {sorted(found-LEFT)}'
