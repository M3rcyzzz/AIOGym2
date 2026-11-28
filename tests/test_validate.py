"""The validation tools (aio.sim.validate) on the plants of the engine: the fast parts only."""
import numpy as np
import pytest

from aio.sim.validate import checks as CK, papers as PP
from aio.sim.validate.tools import Unit

KNOWN = {}                                                       # differences from the papers that are reported (docs/validation.md), not yet changed in the plant


@pytest.mark.parametrize('name', list(PP.PAPERS))
def test_equations_match_the_second_transcription(name):
    assert max(CK.equations(name, n=40).values()) < 1e-12


@pytest.mark.parametrize('name', list(CK.PUBLISHED))
def test_published_numbers(name):
    bad = {r['what'] for r in CK.published(name) if not r['ok']}
    assert bad == KNOWN.get(name, set())


@pytest.mark.parametrize('name', ['qtank', 'ph', 'evap', 'furnace', 'gasnet', 'hex', 'column'])
def test_ledgers_close(name):
    assert max(CK.conservation(name, n=40).values()) < 1e-12


def test_reference_integrator_agrees_with_the_engine_refined():
    u = Unit('vdv')
    xs, ok = u.steady()
    w = u.w0.copy()
    w[u.mv[0]] *= 1.1
    W, Pm = [w]*5, [u.p0]*5
    ref, fine = u.run(xs, W, Pm), u.engine(xs, W, Pm, substeps=64, refresh=True)
    assert ok and np.abs(u.outputs(ref, np.array(W), np.array(Pm))-u.outputs(fine, np.array(W), np.array(Pm))).max() < 1e-4


def test_valve_fit_recovers_a_power_law():
    from aio.sim.validate import realdata as RD
    x = np.array([0., 4., 6., 8., 10., 12., 14., 16., 18., 20., 24.])
    u = np.clip((x-4.)/16., 0., 1.)
    r = RD.valve_fit(x, 3e-4*u**1.3)
    assert abs(r['gamma']-1.3) < .01 and r['gap'] < 1e-3 and r['turns'] == 0


def test_step_model_recovers_gain_and_dead_time():
    from aio.sim.validate import realdata as RD
    rng = np.random.default_rng(0)
    N, th, tau, K = 4000, 3, 8., np.array([[2., -1.], [.5, 1.5]])
    U = np.repeat(rng.standard_normal((N//20, 2)), 20, 0)                   # inputs held 20 samples
    a = np.exp(-1/tau)
    X = np.zeros((N, 2, 2))
    for t in range(1, N):
        X[t] = a*X[t-1]+(1-a)*K*(U[t-1-th][None, :] if t-1-th >= 0 else 0.)
    Y = X.sum(2)+.05*rng.standard_normal((N, 2))
    S, fit = RD.step_model(U, Y, 60)
    G = S[-1]*Y.std(0)[:, None]/U.std(0)[None, :]                            # back to the units of the record
    assert fit.min() > .9 and np.abs(G-K).max() < .15
    assert abs(int(np.argmax(np.abs(S[:, 0, 0]) >= .05*abs(S[-1, 0, 0])))-(th+1)) <= 1


def test_boptest_formula_pi_and_shape():
    """The step-test reading used on BOPTEST: a single lag has time-to-63 % over time-to-20 % of 4.5, and
    the formula PI of a lag with dead time is the lambda rule's."""
    import numpy as np
    from aio.sim.validate import boptest as B
    k = np.arange(241.)
    lag = 1-np.exp(-k/20)
    assert abs(B.shape(lag)['spread']-np.log(1-.632)/np.log(1-.2)) < .05 and B.shape(lag)['settled'] < 1e-3
    two = .3*(1-np.exp(-k/2))+.7*(1-np.exp(-k/200))               # a fast part and a slow tail
    assert B.shape(two)['spread'] > 20
    S = (2.*np.where(k >= 5, 1-np.exp(-(k-5)/20), 0.))[:, None, None]
    t = B.lambda_pi(S)
    theta, tau = t['theta'][0], t['tau'][0]
    assert abs(theta-5.5) <= 1 and abs(tau-20) <= 2 and abs(t['kc'][0]-tau/(t['gain'][0]*(max(tau, 3*theta)+theta))) < 1e-9 and abs(t['gain'][0]-2.) < 1e-3


def test_a_real_loop_is_placed_among_the_paired_responses_of_the_prior():
    """One loop is compared with the prior's paired responses one by one, not with the plants' largest ratio. The
    numbers come from code written apart from the engine's (aio.sim.stats.shapes, alike) and agree with it."""
    import numpy as np
    import torch
    from aio.sim import stats as ST
    from aio.sim.validate import boptest as B, realdata as RD
    k = np.arange(241.)
    lag, two = np.where(k >= 3, 1-np.exp(-(k-3)/20), 0.), .3*(1-np.exp(-k/2))+.7*(1-np.exp(-k/200))
    for d in (lag, two):
        v, w = RD.loop(d), B.shape(d)
        assert abs(v['spread']-w['spread']) < 1e-9 and abs(v['settled']-w['settled']) < 1e-9
    v = RD.loop(lag)
    assert v['theta'] == 5 and 15 <= v['tau'] <= 20 and abs(v['ratio']-v['theta']/v['tau']) < 1e-12
    n, seed = 96, 3
    P, b = RD.prior_pairs(n, seed), RD.prior_batch(n, seed)
    ratio, spread = ST.shapes(b['S'], b['pair'], b['act'] & ~torch.gather(b['integ'], 1, b['pair']))
    assert len(P['ratio']) == len(ratio) > n and np.abs(P['ratio']-ratio.numpy()).max() < 1e-9 and np.abs(P['spread']-spread.numpy()).max() < 1e-9
    for q, (a, sp) in ST.LOOPS.items():
        assert abs(RD.alike(P, dict(ratio=a, spread=sp))-ST.alike(ratio, spread, q)) < 1e-12
    fe, pairs = RD.place(dict(theta=v['theta'], tau=v['tau'], ratio=v['ratio'], speed=1.), [v], n, seed)
    assert set(fe) == {'theta', 'tau', 'ratio'} and fe['ratio']['percentile'] == pairs[0]['ratio']['percentile']      # one loop: against the pairs
    assert 0 < pairs[0]['ratio']['percentile'] < 100 and 0 <= pairs[0]['alike'] <= 1
    rec = RD.replace(dict(a=dict(features={q: dict(value=x['value']) for q, x in fe.items()}, loops=[v])), n, seed)
    assert rec['a']['pairs'] == pairs and rec['prior']['batch']['pairs'] == len(ratio)
    assert RD.pct(np.array([1., 2., 2., 3.]), 2.) == 50.


def test_boptest_overwrite_drives_every_point_of_an_mv():
    from aio.sim.validate import boptest as B
    q = B.overwrite(B.LOOPS['bestest_hydronic_heat_pump'], [.4])
    assert q == {'oveHeaPumY_u': .4, 'oveHeaPumY_activate': 1, 'oveFan_u': 1., 'oveFan_activate': 1, 'ovePum_u': 1., 'ovePum_activate': 1}
    assert B.overwrite(B.LOOPS['bestest_hydronic_heat_pump'], [0.])['ovePum_u'] == 0.
    q = B.overwrite(B.LOOPS['twozone_apartment_hydronic'], [1., 1.])
    assert abs(q['hydronicSystem_oveMpumCon_u']-.344) < 1e-9


def test_stress_measures_the_guarded_step_against_a_converged_reference():
    from aio.sim.validate import numerics as NU
    r = NU.stress('qtank', n=2, draws=64, jump=.2, steps=40)
    assert r['decisions'] == 64 and r['jump'] == .2
    assert r['reference_vs_finer'] < 1e-2                         # the reference is converged
    assert r['guarded_max'] <= r['plain_max']+1e-9                # the guard never makes a decision worse
    assert 0 < r['moved_median'] and r['guarded_max'] < 1.        # a fifth of the span on the four tanks stays inside a band


def test_stress_at_the_slew_limit_and_the_rows_of_the_task():
    from aio.sim.validate import numerics as NU
    r = NU.stress('qtank', n=2, draws=64, jump='slew', steps=40)
    a = NU.stress('qtank', n=2, draws=64, jump=.2, steps=40)
    assert r['jump'] == 'slew' and 0 < r['moved_median'] < a['moved_median']       # the pumps' own slew (5 % of the span) moves the plant less than a fifth of the span
    assert r['guarded_max_task'] <= r['guarded_max'] == max(r['guarded_by_output'].values())
    n = NU.numerics('btcol', n=2, controller='pi', exact=0, steps=6, refine=4)       # a designed column: only the tracked and limited outputs are rows of the task
    assert n['task_outputs'] == ['yD', 'yB', 'P', 'dP', 'TB', 'LD', 'LB'] and n['local_max_task'] <= n['local_max']
    assert set(n['local_by_output']) > set(n['task_outputs'])


class _Ideal:
    """Constant relative volatility, no sensible heat, one heat of vaporisation: constant molar overflow holds exactly."""

    def __init__(self, alpha, P=1e5):
        self.a, self.P = alpha, P

    def kh(self, T, P, x, y):
        pb = self.P*np.exp(-4000.*(1./T-1./350.))
        return np.array([self.a*pb, pb])/P, 0., 30000.

    def hl(self, T, P, x):
        return 0.


def test_stage_solver_reproduces_mccabe_thiele():
    from scipy.optimize import brentq
    from aio.sim.validate.columns import Rigorous
    al, N, nf, F, z, L, D = 2.5, 10, 5, 100., .5, 150., 50.
    B, V = F-D, L+D
    eq = lambda y: y/(al-(al-1)*y)                               # noqa: E731

    def bottom(xd):                                              # stage to stage down the operating lines
        x = eq(xd)
        for j in range(N-1, -1, -1):
            x = eq((L*x+D*xd)/V if j >= nf else ((L+F)*x+D*xd-F*z)/V)
        return x-(F*z-D*xd)/B
    xd = brentq(bottom, .9, .985, xtol=1e-14)                    # above that the stepping leaves the diagram
    xa = np.linspace(.05, .95, N+2)
    x = np.stack((xa, 1-xa), 1)
    y = np.stack((al*xa, 1-xa), 1)/(1+(al-1)*xa)[:, None]
    T = 1./(1./350.+np.log(1+(al-1)*xa)/4000.)
    e = dict(species=['a', 'b'], nstage=N, nfeed=nf, light=[True, False], x=x, y=y, T=T, P=np.full(N+2, 100.), F=F, z=np.array([z, 1-z]), TF=350.,
             V=np.full(N+1, V), L=np.r_[np.full(nf, L+F), np.full(N+1-nf, L)], B=B, reflux=L)
    r = Rigorous(e, _Ideal(al))
    X, res = r.solve(L, B)
    g = r.read(X, L)
    assert res < 1e-9
    assert abs(g['xd'][0]-xd) < 1e-7 and abs(g['xb'][0]-(F*z-D*xd)/B) < 1e-7
    assert np.abs(np.array(g['V'])-V).max() < 1e-6               # constant molar overflow


def test_column_state_closes_the_balances():
    from aio.sim.validate.columns import engine_state
    e = engine_state('btcol')
    j = np.arange(e['nstage']+1)
    assert np.abs(e['V']-e['L']-np.where(j >= e['nfeed'], e['D'], -e['B'])).max() < 1e-6      # between any two stages: up minus down is what leaves above
    assert np.abs(e['F']*e['z']-e['D']*e['x'][-1]-e['B']*e['x'][0]).max() < 1e-6
    assert abs(e['yD']+np.log10(e['x'][-1][1])) < 1e-9 and abs(e['yB']+np.log10(e['x'][0][0])) < 1e-9


def test_column_agrees_with_the_stage_solution_on_the_fitted_package():
    pytest.importorskip('thermo')
    from aio.sim.validate.columns import against, engine_state, models
    e = engine_state('btcol')
    s = against(e, models(e['species'], e['model'])['fitted'](), specs=False)['same_flows']
    assert abs(s['yD_bands']) < .1 and abs(s['yB_bands']) < .1 and s['T_max'] < .1 and abs(s['Qr']) < .005
