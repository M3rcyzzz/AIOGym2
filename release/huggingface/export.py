"""Write the Hugging Face release of one checkpoint of aio.train: the folder that is uploaded as it is.

  python release/huggingface/export.py --ckpt RUN/last.pt --out STAGING --name v2-stay-rl5 [--eval FILE ...]

STAGING is emptied first (but for .git* and .cache), so a release overwrites the one before: one
checkpoint, no folders of versions. It gets
  model.safetensors   the network that runs (actor.* and pairs.*)
  config.json         what loading needs (name, row, sizes, interface, probe, unpaired): no training settings
  code/               the modules the network is made of, only what running it reaches, without docstrings or
                      comments (slim.py): nothing of the trainer, the teacher or the simulator is published
  modeling_aio.py     load(), Controller and encode on those files
  eval/               only the files given with --eval (none by default)
  README.md, README_zh.md, LICENSE
and is then checked from outside the repository: a fresh interpreter that is given STAGING alone loads the
network and must give, bit for bit, the moves the checkpoint gives here on the same packets. Copy a
checkpoint before converting it if a run may still write it. ModelScope's copy: release/modelscope/build.py.
Upload either with the hub's own deletion of what is no longer in the folder (hf upload --delete "*").
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from aio import net as N  # noqa: E402
from slim import slim  # noqa: E402

KEEP = ('.git', '.cache')                                         # of a staging folder: the hub's own
OF = ('law', 'knots', 'tied', 'given', 'slow', 'arx', 'dv', 'diff', 'unseen', 'told', 'reader', 'apart')   # what the pair pathway has (aio.net: *_of)
CHECK = '''
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import modeling_aio as M
assert M._N.__file__.startswith(sys.argv[1]), M._N.__file__     # the copy in code/, not a repository on the path
net, d = M.load(), np.load(sys.argv[2])
out = {}
for k in d.files:
    ctl = M.Controller(net)
    out[k] = np.stack([ctl.step(p) for p in d[k]])
np.savez(sys.argv[3], **out)
'''


def packets(seed=0):
    """The public packets of two small plants over a few decisions (a 3x2 and a 5x4, a reverse-acting MV and
    one of unknown sign, a setpoint change half way): {name: (T, SIZE)}."""
    enc = sys.modules[N.shape_of.__module__].encode
    g, out = np.random.default_rng(seed), {}
    for n, m, sign in ((3, 2, (1, -1)), (5, 4, (1, 1, -1, 0))):
        y, u = g.normal(size=n), g.uniform(.3, .7, m)
        sp = y+.5*g.normal(size=n)
        py, pu, psp, seq = y.copy(), u.copy(), sp.copy(), []
        for t in range(12):
            seq.append(enc(values=y, previous=py, target=sp, last_target=psp, scale=np.ones(n), lower=sp-1., upper=sp+1., tracking=np.ones(n),
                           command=u, last_command=pu, u_low=np.zeros(m), u_high=np.ones(m), slew=np.full(m, .05), pairing=list(range(m)),
                           dt=60., sign=sign))
            py, pu, psp = y.copy(), u.copy(), sp.copy()
            y, u = y+.1*g.normal(size=n), np.clip(u+.02*g.normal(size=m), 0., 1.)
            if t == 5:
                sp = sp+.3*g.normal(size=n)
        out[f'plant{n}x{m}'] = np.stack(seq)
    return out


def moves(net, plants):
    out = {}
    for k, seq in plants.items():
        ctl = N.Controller(net)
        out[k] = np.stack([ctl.step(p) for p in seq])
    return out


def modules():
    """The repository's files among the modules loaded now, but for this folder's."""
    out = []
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        f = Path(f).resolve() if f and Path(f).is_absolute() else None   # (torch has modules with a bare file name)
        if f and f.is_file() and f.is_relative_to(ROOT) and not f.is_relative_to(ROOT/'release'):
            out.append(f.relative_to(ROOT))
    return sorted(set(out))


def commit():
    def git(*q):
        return subprocess.run(('git', '-C', str(ROOT))+q, capture_output=True, text=True).stdout.strip()
    return git('rev-parse', 'HEAD')+('+changes' if git('status', '--porcelain', '--untracked-files=no') else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='the staging folder; emptied first')
    ap.add_argument('--name', required=True, help='what the release is called in config.json and the cards (the run)')
    ap.add_argument('--eval', type=Path, nargs='*', default=(), help='more files of results, copied to eval/ with this folder\'s')
    a = ap.parse_args()
    net, ck = N.load(a.ckpt)
    plants = packets()
    ref = moves(net, plants)                                      # (and what running it imports is loaded now)
    a.out.mkdir(parents=True, exist_ok=True)
    for p in a.out.iterdir():
        if not p.name.startswith(KEEP):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    model, pairs, cfg = ck['model'], ck['model'].get('pairs'), ck.get('config') or {}
    state = {'actor.'+k: v for k, v in model['actor'].items()}
    state.update({'pairs.'+k: v for k, v in (pairs or {}).items()})
    state = {k: v.detach().cpu().clone().contiguous() for k, v in state.items()}
    config = dict(name=a.name, row=ck.get('iter'),                # only what loading needs, and what it is: no training settings
                  params={q: sum(v.numel() for v in model[q].values()) for q in ('actor', 'pairs') if model.get(q)},
                  interface=N.shape_of.__module__, probe=float(cfg.get('probe') or 0.), unpaired=float(cfg.get('unpaired') or 0.))
    save_file(state, a.out/'model.safetensors',                   # (one entry: the file is the same bytes at every export of a commit)
              metadata=dict(release=json.dumps(dict(name=a.name, row=config['row']))))
    (a.out/'config.json').write_text(json.dumps(config, indent=1, default=str, ensure_ascii=False), encoding='utf-8')
    files = modules()
    roots = {'aio.net': {'released', 'Controller'}, N.shape_of.__module__: {'encode', 'NCV', 'NMV', 'SIZE'}}
    for f, src in slim(ROOT, files, roots).items():               # what running it needs, without docstrings or comments
        (a.out/'code'/f).parent.mkdir(parents=True, exist_ok=True)
        (a.out/'code'/f).write_text(src, encoding='utf-8')
    for f in ('modeling_aio.py', 'README.md', 'README_zh.md', 'LICENSE'):
        shutil.copyfile(HERE/f, a.out/f)
    for f in list(a.eval):                                         # results only when given; none by default
        (a.out/'eval').mkdir(exist_ok=True)
        shutil.copyfile(f, a.out/'eval'/f.name)
    with tempfile.TemporaryDirectory() as tmp:                    # from outside: the folder alone, a fresh interpreter
        np.savez(Path(tmp)/'packets.npz', **plants)
        subprocess.run((sys.executable, '-I', '-c', CHECK, str(a.out.resolve()), str(Path(tmp)/'packets.npz'), str(Path(tmp)/'moves.npz')),
                       check=True, cwd=tmp)
        got = np.load(Path(tmp)/'moves.npz')
        for k, x in ref.items():
            assert np.array_equal(x, got[k]) and np.abs(x).max() > 0, f'{k}: the released network does not give the moves of the checkpoint'
    for p in list(a.out.rglob('__pycache__')):
        shutil.rmtree(p, ignore_errors=True)
    print(json.dumps(dict(name=a.name, row=config['row'], commit=commit(), params=config['params'], checked=sorted(ref),
                          files={str(p.relative_to(a.out)): p.stat().st_size for p in sorted(a.out.rglob('*')) if p.is_file() and not p.name.startswith('.')})))


if __name__ == '__main__':
    main()
