"""The direct-acting frame of the public packet (aio.interface).

A plant publishes, per MV, the sign of its paired steady gain (interface.SIGN): commissioning knowledge, as a PID
loop's direct/reverse action. `canonical` mirrors every MV whose published sign is -1, so a known paired gain always
reads +1 and the network only learns direct-acting control; its action on that MV is mirrored back before the plant
gets it. Hidden or unknown signs (0) stay as published: those are the MVs whose direction has to be found online.
Canonicalisation gives the symmetry exactly instead of learning it from augmented data (Puny et al., ICLR 2022;
Kaba et al., ICML 2023)."""
import torch

from aio.interface import CF, MF, SIGN, shape as shape_of


def _mvs(obs):
    """Where the MV rows are in packets of obs' length and how many they are (MV0, MV1, NMV): the size is the packets'
    (aio.interface.SHAPES)."""
    ncv, nmv = shape_of(obs.shape[-1])
    return ncv*CF, ncv*CF+nmv*MF, nmv


def canonical(obs, unpaired=None):
    """Packets in the direct-acting frame and the mirror s (..., NMV) of +-1 that maps them back: MVs
    published with sign -1 are mirrored (features 0, 1 negated, sign read +1); hidden or unknown signs (0)
    stay as published. The plant gets s * z for the agent's action z.
    unpaired (...,) bool: those packets go on with nothing published about the pairing. Which row an MV is
    published with (the block behind the MVs) is all zero, the sign slot reads +1 on every MV in service and
    nothing is mirrored (s = 1: the plant's own frame, in which the agent has to find out which way an MV
    moves a row). Both come from the plant's true gains (aio.sim.prior), as a loop sheet would have them."""
    s = 1.-2.*(signs(obs) < 0).to(obs.dtype)
    if unpaired is None:
        return flip_mvs(obs, s), s
    b = unpaired.to(obs.dtype)[..., None]
    s = 1.+(s-1.)*(1.-b)
    o = flip_mvs(obs, s)                                           # the others mirrored; these as published
    MV0, MV1, NMV = _mvs(obs)
    mv = o[..., MV0:MV1].unflatten(-1, (NMV, MF))
    mv = torch.cat((mv[..., :SIGN], mv[..., SIGN:SIGN+1].abs(), mv[..., SIGN+1:]), -1).flatten(-2)
    return torch.cat((o[..., :MV0], mv, o[..., MV1:]*(1.-b)), -1), s


def _mv_scaled(obs, g):
    """Packets with the MV rows multiplied by g (..., NMV, MF), broadcast against obs' leading dims."""
    MV0, MV1, NMV = _mvs(obs)
    mv = (obs[..., MV0:MV1].unflatten(-1, (NMV, MF))*g.to(obs.dtype)).flatten(-2)
    return torch.cat((obs[..., :MV0].expand(*mv.shape[:-1], -1), mv, obs[..., MV1:].expand(*mv.shape[:-1], -1)), -1)


def flip_mvs(obs, f, keep=None):
    """Packets with MV features 0, 1 (position, last move) and SIGN multiplied by f (..., NMV) of +-1,
    which broadcasts against obs' leading dims; keep (same shape, 0/1) also zeroes the sign where 0.
    Without keep its own inverse; out of place (CUDA-graph safe)."""
    one = torch.ones_like(f)[..., None]
    sg = f*keep if keep is not None else f
    g = torch.cat((f[..., None], f[..., None], one.expand(*f.shape, SIGN-2), sg[..., None], one.expand(*f.shape, MF-SIGN-1)), -1)
    return _mv_scaled(obs, g)


def signs(obs):
    """Published paired-gain signs (..., NMV) of packets (0 for inactive MVs)."""
    MV0, MV1, NMV = _mvs(obs)
    return obs[..., MV0:MV1].unflatten(-1, (NMV, MF))[..., SIGN]
