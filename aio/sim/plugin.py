"""Your controller in the environment's command-line tools: --ctl MODULE:FACTORY names a function that builds it.

A controller of the core (aio.sim.control) has reset(prog) and __call__(prog, st, k, obs) -> moves (n, NMV) in
[-1, 1] of each MV's slew limit. A packet controller (aio.sim.validate.boptest, one plant at a time) has
step(packet) -> moves (NMV,). FACTORY receives the tool's --arg string (a weights file, a config) and the device."""
import importlib


def factory(spec):
    """The function named by 'package.module:function'."""
    mod, _, fn = spec.partition(':')
    if not mod or not fn:
        raise SystemExit(f'--ctl wants MODULE:FACTORY, got {spec!r}')
    return getattr(importlib.import_module(mod), fn)


def build(spec, arg=None, **kw):
    """factory(spec)(arg, **kw), returning (controller, meta): a factory may return the controller alone."""
    out = factory(spec)(arg, **kw)
    return out if isinstance(out, tuple) else (out, None)
