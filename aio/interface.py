"""Public, dimensionless CV/MV tokens; no plant or task identity input.

MV row: 0 position 2(u-u_low)/span-1, 1 last move, 2 slew, 3 lower bound (-1), 4 sign of the paired
steady gain (+1, -1, or 0 when it is not known or changes sign over the operating range; before 11-07
this slot was always 1, which every family published until then had: all their paired gains are
positive), 5 active. The sign is commissioning knowledge, as a PID loop's direct/reverse action.

CV row: 0 error/scale, 1 last change, 2-3 limits about the target, 4 MV-driven flag (0 on a measured
disturbance row: a DV the MVs do not move, published for feedforward; 1 otherwise), 5 priority weight
of the tracked CV (the reward's tracking term carries the same weight; 1 when not set), 6 tracked,
7 known, 8 log age, 9 log1p(dt)/10, 10 target change, 11 active. Slots 4 and 5 were always 1 until
11-07 (DV flag and priority: the APC configuration items a DMC engineer sets), so a packet without
DVs or priorities is the same as before.

A row without a measurement (known 0) publishes 0 in slots 0 and 1 whatever value the caller holds for it
(training does the same); when the measurement is back, `previous` is the last good value."""
import numpy as np

NCV, NMV, CF, MF = 12, 5, 12, 6
SIZE=NCV*CF+NMV*MF+NCV*NMV
SIGN = 4   # MV feature: published sign of the paired steady gain
DV, PRIORITY = 4, 5   # CV features: MV-driven flag (0 = measured disturbance), priority weight
SHAPES = ((NCV, NMV), (24, 10), (48, 20))   # (rows, MVs) of the packets in use; the first is the default, the larger ones are aio.sim's (prior.SIZES)


def size(ncv=NCV, nmv=NMV):
    """Length of the packet of ncv rows and nmv MVs."""
    return ncv*CF+nmv*MF+ncv*nmv


def shape(length, nmv=None):
    """(rows, MVs) of a packet of this length: by the width of the action when given, else the one of SHAPES
    with this length (a length alone does not tell: 234 is 12 rows by 5 MVs and 11 by 6)."""
    if nmv is None:
        for s in SHAPES:
            if size(*s) == length:
                return s
        raise ValueError(f'no packet in use has length {length}')
    ncv, rest = divmod(length-nmv*MF, CF+nmv)
    if rest or ncv < 1:
        raise ValueError(f'a packet of length {length} does not go with {nmv} MVs')
    return ncv, nmv


def rows(nmv):
    """Rows of the packets in use that have nmv MVs (SHAPES)."""
    for s in SHAPES:
        if s[1] == nmv:
            return s[0]
    raise ValueError(f'no packet in use has {nmv} MVs')


def split(obs, nmv=None):
    """A packet (..., length), numpy or torch, as its CV rows (..., rows, CF), MV rows (..., MVs, MF) and
    pairing (..., rows, MVs)."""
    ncv, nmv = shape(obs.shape[-1], nmv)
    a, b, lead = ncv*CF, ncv*CF+nmv*MF, tuple(obs.shape[:-1])
    return obs[..., :a].reshape(lead+(ncv, CF)), obs[..., a:b].reshape(lead+(nmv, MF)), obs[..., b:].reshape(lead+(ncv, nmv))


def encode(values,previous,target,last_target,scale,lower,upper,tracking,
           command,last_command,u_low,u_high,slew,pairing,dt,*,known=None,age=None,sign=None,dv=None,weight=None):
    y=np.asarray(values,float);n=len(y);m=len(command)
    if not 1<=n<=NCV or not 1<=m<=NMV:raise ValueError('Unsupported public channel count')
    if np.any(np.asarray(scale)<=0) or np.any(np.asarray(u_high)<=u_low):raise ValueError('Invalid public scales')
    cv=np.zeros((NCV,CF),np.float32);mv=np.zeros((NMV,MF),np.float32);links=np.zeros((NCV,NMV),np.float32)
    cv[:n,0]=(y-target)/scale;cv[:n,1]=(y-previous)/scale
    cv[:n,2]=(lower-target)/scale;cv[:n,3]=(upper-target)/scale
    cv[:n,4:6]=1;cv[:n,6]=tracking;cv[:n,7]=1 if known is None else known
    if known is not None:cv[:n,:2]*=(np.asarray(known,float)>0)[:,None]   # a row without a measurement publishes zeros for its value: whatever stands in `values` there is not evidence (as aio.sim.core.observe)
    if dv is not None:cv[:n,DV]=1-np.asarray(dv,float)          # dv: 1 on measured-disturbance rows
    if weight is not None:cv[:n,PRIORITY]=weight                # tracked-CV priority (geometric mean 1 by convention)
    cv[:n,8]=0 if age is None else np.log1p(np.asarray(age)/dt)
    cv[:n,9]=np.log1p(dt)/10;cv[:n,10]=(target-last_target)/scale;cv[:n,11]=1
    span=np.asarray(u_high)-u_low
    mv[:m,0]=2*(command-u_low)/span-1;mv[:m,1]=2*(command-last_command)/span
    mv[:m,2]=2*np.asarray(slew)/span;mv[:m,3]=-1;mv[:m,4:6]=1
    if sign is not None:mv[:m,4]=np.sign(np.asarray(sign,float))
    for j,i in enumerate(pairing):
        if not 0<=i<n:raise ValueError('Pairing is a public CV index')
        links[i,j]=1
    result=np.r_[cv.ravel(),mv.ravel(),links.ravel()]
    if result.shape!=(SIZE,) or not np.isfinite(result).all():raise ValueError('Invalid public packet')
    return result


def permute(obs,cv_order,mv_order):
    a=obs[:NCV*CF].reshape(NCV,CF)[cv_order]
    b=obs[NCV*CF:NCV*CF+NMV*MF].reshape(NMV,MF)[mv_order]
    c=obs[-NCV*NMV:].reshape(NCV,NMV)[cv_order][:,mv_order]
    return np.r_[a.ravel(),b.ravel(),c.ravel()].astype(np.float32)
