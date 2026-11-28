"""The scale a controller is read on: where it stands between the MVs left where they are and the best causal
controller, on the same episodes (docs/sim.md, 'Scale').

    points = 100 * (hold - controller) / (hold - best),   decisions outside, summed over the batch

0 is the plant with its MVs left where they are (no control at all); below 0 is worse than doing nothing. 100 is,
per episode, the best score of every causal controller that was run on it: the Oracle (aio.sim.oracle: a
predictive controller that knows the plant and its state, and nothing of what comes), the re-identified
exact-model DMC, the training teacher, the step-test DMC. The envelope, not the Oracle alone, because the
Oracle's plan is made for a calm, frozen future and loses to a more cautious controller on some plants; a
controller that beats the envelope reads above 100, and the envelope is then lowered to it when the file is
next written.

The formula PI is a reference on the scale and neither of its ends. Until 11-10 the scale was the multiple of
the formula PI: a student at 0.91 of it read as "better than PI" while it was a small part of the way to a
controller that knows the plant. Until 11-11 the formula PI was still the scale's 0 ("recovered"), and a long
run's criterion was written as "above the formula PI". The aim is a controller of APC level: on the validation
batch of 11-11 the formula PI reads 53, the step-test DMC (what an APC project has after step tests on the
plant) 88, the teacher 94. `Scale.from_recovered` reads a number of the old scale on this one.

The sums are over the same episodes: the ratio is paired, and it is an affine function of the controller's
mean score, so choosing a checkpoint by it is choosing by the score. A file of per-episode reference scores
(aio.sim.oracle --out) is read with `Scale.load`; it is used only on the batch it was computed on, which is
checked on the formula PI's per-episode scores (`same_batch`): the same seed draws other episodes on another
version of the core, and has drawn other plants on another machine (11-10, a tie in the pairing).
"""
import json
from pathlib import Path

ZERO = 'hold'                                 # what the scale's 0 is; written with every number that is kept (a checkpoint's best)
NOT_CAUSAL = ('bound', 'bound_causal')        # aio.sim.bound: they know the noise to come; never part of the envelope


def envelope(per):
    """Per episode, the lowest score of the causal controllers in `per` ({name: per-episode scores})."""
    rows = [v for q, v in per.items() if q not in NOT_CAUSAL and q != 'hold']
    return [min(x) for x in zip(*rows)]


def points(score, hold, best):
    """Where `score` stands from the MVs left where they are (0) to the best causal controller (100), over the batch."""
    s, h, b = (float(sum(x)) for x in (score, hold, best))
    return 100.*(h-s)/max(h-b, 1e-9)


def tail(score, hold, best, share=.1):
    """How a controller does where it does worst, which the batch's points do not show: `worse`, the share of the
    episodes it leaves worse than the MVs left where they are; `low`, its points over the tenth of the episodes it
    does worst on (by each episode's own points; `share`); `worst`, the lowest of those. A controller for a plant
    has to be safe on every unit, not on the average one."""
    floor = .1*max(sum(hold)-sum(best), 1e-9)/max(len(hold), 1)           # (an episode with next to nothing to gain is read against a tenth of the average gain)
    each = sorted((100.*(h-s)/max(h-b, floor), s, h, b) for s, h, b in zip(score, hold, best))
    k = max(1, round(share*len(each)))
    return dict(worse=sum(s > h for s, h in zip(score, hold))/max(len(each), 1), low=points(*zip(*[e[1:] for e in each[:k]])), worst=each[0][0])


def same_batch(a, b, tol=.03):
    """Whether two lists of the formula PI's per-episode scores are of the same episodes: equal up to what
    rounding does through the loop and the band's threshold (CPU against CUDA: under 1 % of the mean)."""
    if len(a) != len(b):
        return False
    return sum(abs(x-y) for x, y in zip(a, b)) <= tol*max(sum(a), 1.)


class Scale:
    """The two ends on one batch. hold, best: per-episode scores; basis: the controllers the envelope is of;
    pi: the formula PI's scores (the batch is told by them, and numbers of the old scale are read through them)."""

    def __init__(self, hold, best, basis, pi):
        self.hold, self.best, self.pi, self.basis = [float(x) for x in hold], [float(x) for x in best], [float(x) for x in pi], tuple(basis)

    @classmethod
    def of(cls, per):
        """From per-episode scores of the controllers run ({name: scores}, 'hold' and 'pi' among them)."""
        per = {q: [float(x) for x in v] for q, v in per.items()}
        return cls(per['hold'], envelope(per), [q for q in per if q not in NOT_CAUSAL and q not in ('hold', 'best')], per['pi'])

    @classmethod
    def load(cls, path, pi=None):
        """From a file of aio.sim.oracle. pi: the formula PI's scores on the batch at hand; None is returned
        if they are not the file's (another batch: its envelope says nothing here)."""
        d = json.loads(Path(path).read_text())['per_episode']
        if pi is not None and not same_batch(d['pi'], [float(x) for x in pi]):
            return None
        return cls.of({q: v for q, v in d.items() if q != 'best'})

    def join(self, per):
        """The envelope lowered by further controllers run on the same batch (the references of this run); the
        0 is this run's own hold where it ran one."""
        per = dict({q: [float(x) for x in v] for q, v in per.items()}, _=self.best)
        basis = self.basis+tuple(q for q in per if q not in self.basis and q not in ('_', 'hold', 'best')+NOT_CAUSAL)
        return Scale(per.get('hold', self.hold), envelope(per), basis, self.pi)

    def __call__(self, score):
        return points([float(x) for x in score], self.hold, self.best)

    def from_recovered(self, rec):
        """A number of the scale as it was until 11-11 (0 the formula PI, 1 the envelope), in points."""
        h, p, b = sum(self.hold), sum(self.pi), sum(self.best)
        return 100.*(h-p+rec*(p-b))/max(h-b, 1e-9)

    def meta(self):
        n = max(len(self.pi), 1)
        return dict(zero=ZERO, basis=list(self.basis), oracle=any(q.startswith('oracle') for q in self.basis), hold=round(sum(self.hold)/n, 1),
                    pi=round(sum(self.pi)/n, 1), best=round(sum(self.best)/n, 1))
