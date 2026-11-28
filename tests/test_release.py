"""release/: a checkpoint written as the one folder that is uploaded (export.py runs its own check: the folder
alone, in a fresh interpreter, gives the checkpoint's moves bit for bit), the folder before it overwritten,
and ModelScope's copy of it."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def run(*q, cwd):
    r = subprocess.run((sys.executable,)+tuple(map(str, q)), capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_export_is_one_checkpoint_that_overwrites_the_one_before(tmp_path):
    pytest.importorskip('safetensors')
    from aio import net as N
    torch.manual_seed(0)
    net = N.AIO(law=True, slow=1e6).eval()
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), config=dict(unpaired=0.), iter=3, hours=.1),
               tmp_path/'last.pt')
    out, ms = tmp_path/'hf', tmp_path/'ms'
    (out/'older-weights').mkdir(parents=True)                     # a release from before, and the hub's own files
    (out/'older-weights'/'actor.safetensors').write_bytes(b'x')
    (out/'export.py').write_text('old')
    (out/'.gitattributes').write_text('kept')
    (tmp_path/'score.json').write_text('{}')
    said = json.loads(run(ROOT/'release/huggingface/export.py', '--ckpt', tmp_path/'last.pt', '--out', out, '--name', 'test',
                          '--eval', tmp_path/'score.json', cwd=tmp_path).splitlines()[-1])
    assert {p.name for p in out.iterdir()} == {'.gitattributes', 'model.safetensors', 'config.json', 'code', 'modeling_aio.py', 'eval',
                                               'README.md', 'README_zh.md', 'LICENSE'}
    assert not list(out.rglob('__pycache__'))
    assert {p.name for p in (out/'eval').iterdir()} == {'score.json'}
    c = json.loads((out/'config.json').read_text(encoding='utf-8'))      # what loading needs, no training settings
    assert set(c) == {'name', 'row', 'params', 'interface', 'probe', 'unpaired'} and (c['name'], c['row']) == ('test', 3)
    assert said['params'] == c['params']
    for f in (out/'code').rglob('*.py'):                          # no docstrings or comments, nothing of the trainer
        src = f.read_text(encoding='utf-8')
        assert '"""' not in src and '#' not in src and 'import law' not in src, f
    assert (out/'code'/Path(*c['interface'].split('.'))).with_suffix('.py').is_file()
    first = (out/'model.safetensors').read_bytes()               # the same bytes at the next export
    run(ROOT/'release/huggingface/export.py', '--ckpt', tmp_path/'last.pt', '--out', out, '--name', 'test', cwd=tmp_path)
    assert (out/'model.safetensors').read_bytes() == first and not (out/'eval').exists()
    back = N.released(out)                                        # the same network, read here from the folder
    for k, v in net.state_dict().items():
        assert torch.equal(v, back.state_dict()[k]), k
    packet = np.zeros(sys.modules[N.shape_of.__module__].SIZE, np.float32)
    assert np.array_equal(N.Controller(net).step(packet), N.Controller(back).step(packet))
    run(ROOT/'release/modelscope/build.py', '--hf', out, '--out', ms, '--ms-repo', 'OWNER/AIO1', cwd=tmp_path)
    assert {p.name for p in ms.iterdir()} == {'model.safetensors', 'config.json', 'code', 'modeling_aio.py', 'README.md', 'README_en.md', 'LICENSE'}
    assert (ms/'model.safetensors').read_bytes() == (out/'model.safetensors').read_bytes()


def test_export_seqaio_v2_round_trips_bit_for_bit(tmp_path):
    pytest.importorskip('safetensors')
    from aio import seqnet as SQ
    torch.manual_seed(0)
    net = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1,e8', w=32).eval()
    with torch.no_grad():
        net.actor.act[-1].weight.normal_(0, .1)
        net.actor.mem_fb.weight.normal_(0, .05)
        for lay in net.actor.layers:
            lay.edge_bias.weight.normal_(0, .1)
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), config=dict(unpaired=0.25), iter=40, hours=1.75),
               tmp_path/'seq_best.pt')
    out, ms = tmp_path/'hf', tmp_path/'ms'
    run(ROOT/'release/huggingface/export.py', '--ckpt', tmp_path/'seq_best.pt', '--out', out, '--name', 'aio1-seq-v2', cwd=tmp_path)
    assert (out/'code/aio/seqnet.py').is_file()
    run(ROOT/'release/modelscope/build.py', '--hf', out, '--out', ms, '--ms-repo', 'M3rcyzzz/AIO1', cwd=tmp_path)
    assert 'Mirror of Hugging Face' in (ms/'README_en.md').read_text(encoding='utf-8')
    assert 'ModelScope 下载' in (ms/'README.md').read_text(encoding='utf-8')


def test_export_seqaio_v3_rel_round_trips_bit_for_bit(tmp_path):
    pytest.importorskip('safetensors')
    from aio import seqnet as SQ
    torch.manual_seed(1)
    net = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1,r1', w=32).eval()
    with torch.no_grad():
        net.actor.act[-1].weight.normal_(0, .1)
        net.actor.mem_fb.weight.normal_(0, .05)
        net.actor.rel_mv_out.weight.normal_(0, .05)
        net.actor.rel_cv_out.weight.normal_(0, .05)
        for lay in net.actor.layers:
            lay.rel_bias.weight.normal_(0, .1)
            lay.rel_lag_bias.weight.normal_(0, .05)
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), config=dict(unpaired=0.25), iter=1, hours=0.1),
               tmp_path/'seq_v3.pt')
    out = tmp_path/'hf_v3'
    run(ROOT/'release/huggingface/export.py', '--ckpt', tmp_path/'seq_v3.pt', '--out', out, '--name', 'aio1-seq-v3', cwd=tmp_path)
    assert (out/'code/aio/seqnet.py').is_file()


def test_export_seqaio_v4_r2_round_trips_bit_for_bit(tmp_path):
    pytest.importorskip('safetensors')
    from aio import seqnet as SQ
    torch.manual_seed(2)
    net = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1,r2', w=32).eval()
    with torch.no_grad():
        net.actor.act[-1].weight.normal_(0, .1)
        net.actor.act_gate.weight.normal_(0, .05)
        net.actor.rel_pipe_in.weight.normal_(0, .05)
        net.actor.mem_fb.weight.normal_(0, .05)
        net.actor.rel_mv_out.weight.normal_(0, .05)
        net.actor.rel_cv_out.weight.normal_(0, .05)
        for lay in net.actor.layers:
            lay.rel_bias.weight.normal_(0, .1)
            lay.rel_lag_bias.weight.normal_(0, .05)
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), config=dict(unpaired=0.25), iter=1, hours=0.1),
               tmp_path/'seq_v4.pt')
    out = tmp_path/'hf_v4'
    run(ROOT/'release/huggingface/export.py', '--ckpt', tmp_path/'seq_v4.pt', '--out', out, '--name', 'aio1-seq-v4', cwd=tmp_path)
    assert (out/'code/aio/seqnet.py').is_file()
