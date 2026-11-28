"""Gymnasium vector environment over one batch of the environment.

    from aio.sim.gym import VecEnv
    env = VecEnv(256, seed=0, device='cuda', dims='24x10')     # 256 plants, 24 rows x 10 MVs packets
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)   # action (256, NMV) in [-1, 1] of each slew limit

All episodes of a batch are at the same decision, so they all end together after HORIZON decisions; the next reset
(or the step after the last, autoreset) draws a fresh batch with the next seed. Observations are the packet of
aio.interface. reward: 'outside' (minus the rows outside their band or limit, from the first event on: the measure the
0-100 scale sums; a count, so no gradient), 'cost' (minus the tracking, zone and move cost) or 'soft' (the same
with Huber tails: the one to differentiate). terminated is a physical failure of the
plant (the episode runs on and goes on counting). info['mask'] marks the MVs a plant has.

tensors=True returns tensors on the batch's device and takes them back, so a policy on the GPU never copies; with
grad=True the step keeps the graph: reward is differentiable with respect to every earlier action (the core's
straight-through gradients at MV limits and valve friction; the stiff solver's Jacobian is held constant in the
backward pass)."""
import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium.vector import VectorEnv
except ImportError as e:                                           # pragma: no cover
    raise ImportError('aio.sim.gym needs gymnasium: pip install gymnasium') from e

from aio.sim import core as C, prior as PR
from aio.interface import size as packet_size

REWARDS = ('outside', 'cost', 'soft')


class VecEnv(VectorEnv):
    metadata = {'autoreset_mode': gym.vector.AutoresetMode.NEXT_STEP} if hasattr(gym.vector, 'AutoresetMode') else {}

    def __init__(self, num_envs, seed=0, device='cpu', reward='outside', tensors=False, grad=False, **program):
        """program: what aio.sim.core.build takes (mech: the share of mechanistic plants, plants: which ones, dims:
        a size of prior.SIZES: '12x5', '24x10', '48x20', menu: what changes, ...)."""
        assert reward in REWARDS, reward
        self.num_envs, self.device, self.reward, self.as_torch, self.grad = num_envs, device, reward, tensors, grad
        self.program, self.seed = program, seed
        dims = program.get('dims', PR.DIMS)
        ncv, nmv = (PR.sizes_of(*(PR.SIZES[dims] if isinstance(dims, str) else dims)))[:2]
        self.nmv = nmv
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (packet_size(ncv, nmv),), np.float32)
        self.single_action_space = gym.spaces.Box(-1., 1., (nmv,), np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)
        self.prog = self.st = None
        self.k = 0

    def _out(self, x):
        return x if self.as_torch else x.detach().cpu().numpy()

    def _info(self):
        return dict(mask=self._out(self.prog.c['mvm'] > 0), k=self.k)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.prog = C.build(self.num_envs, self.seed, self.device, **self.program)
        self.seed += 1
        self.st, self.k = C.reset(self.prog), 0
        return self._out(C.observe(self.prog, self.st, 0).float()), self._info()

    def step(self, action):
        if self.prog is None or self.k >= PR.HORIZON:              # autoreset after the last decision of the batch
            obs, info = self.reset()
            z = torch.zeros(self.num_envs, device=self.prog.device)
            return obs, self._out(z), self._out(z > 0), self._out(z > 0), info
        a = action if isinstance(action, torch.Tensor) else torch.as_tensor(np.asarray(action))
        a = a.to(self.prog.device, self.prog.dtype)
        if not self.grad:
            a = a.detach()
        with torch.set_grad_enabled(self.grad):
            st, out = C.step(self.prog, self.st, a, self.k)
        on = (self.k >= self.prog.k_first).to(out['cost'].dtype)
        if self.reward == 'outside':
            r = -out['outside'].sum(-1).to(out['cost'].dtype)*on
        else:
            r = -out[self.reward]*on
        self.st = st if self.grad else C.detach(st)
        self.k += 1
        fail = out['fail'] > 0 if out['fail'].dtype != torch.bool else out['fail']
        fail = fail.reshape(self.num_envs, -1).any(-1)
        trunc = torch.full((self.num_envs,), self.k >= PR.HORIZON, device=r.device)
        obs = C.observe(self.prog, self.st, min(self.k, PR.HORIZON-1)).float()
        info = self._info()
        info['cost'] = self._out(out['cost'].detach())
        return self._out(obs), self._out(r if self.grad else r.detach()), self._out(fail), self._out(trunc), info
