"""AIO1: load and run the released network. Needs torch, numpy and safetensors, nothing else.

This folder holds model.safetensors (the network), config.json (what loading needs) and code/ (the
modules the network is made of).

    import modeling_aio as M
    net = M.load()                     # this folder's model.safetensors
    ctl = M.Controller(net)            # one per plant, it keeps the memory; ctl.reset() clears it for a new plant
    packet = M.encode(values=y, previous=y_prev, target=sp, last_target=sp_prev, scale=cv_scale,
                      lower=cv_low, upper=cv_high, tracking=is_tracked, command=u, last_command=u_prev,
                      u_low=u_min, u_high=u_max, slew=du_max, pairing=pairing, dt=dt, sign=paired_gain_sign)
    move = ctl.step(packet)            # every MV slot's move in [-1, 1] of its slew limit, 0 on slots not in use
    u = np.clip(u + move[:len(u)] * du_max, u_min, u_max)
"""
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code'))
from aio import net as _N  # noqa: E402

CONFIG = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
_I = importlib.import_module(CONFIG['interface'])                 # the public packet
encode, NCV, NMV, SIZE = _I.encode, _I.NCV, _I.NMV, _I.SIZE
Controller = _N.Controller


def load(path=None, device='cpu'):
    """The network in eval mode from a folder that holds model.safetensors and config.json (this one)."""
    return _N.released(path or HERE, device)
