"""Plant equations as data (docs/sim.md, 'Engine').

A plant is written once as symbolic expressions (`Graph`, `E`): states x, inputs w (MVs and loads),
parameters p, the right-hand sides dx/dt = f(x, w, p) and the outputs y = g(x, w, p). From them come, by
symbolic differentiation with shared subexpressions, the Jacobian df/dx the integrator needs and the full
linearisation (df/dw, dy/dx, dy/dw) the build tools need. `compile_table` turns any set of expressions into
a table: per level, lists of operations on registers (linear combinations, products, quotients and a few
unary functions). The engine (aio.sim.engine) runs tables; it has no code of any plant.

Operations (every plant is made of these):
  lin    c0 + sum c_i r_i                      mul   r_a r_b                div   r_a / r_b
  exp, log, sqrt, abs, sign, relu, step (1 where r > 0), tanh
min, max, clamp, where and real powers are composed from them (`minimum`, `maximum`, `clamp`, `where`, `**`).
"""
import math
import numpy as np

UN = ('exp', 'log', 'sqrt', 'abs', 'sign', 'relu', 'step', 'tanh')
CLS = ('lin', 'mul', 'div')+UN
INLINE = 4                       # a sum of up to this many terms is merged into the sum that uses it
TINY = 1e-30


def _fold(op, v):
    if op == 'exp':
        return math.exp(min(max(v, -80.), 80.))
    if op == 'log':
        return math.log(max(v, TINY))
    if op == 'sqrt':
        return math.sqrt(max(v, 0.))
    if op == 'abs':
        return abs(v)
    if op == 'sign':
        return float((v > 0)-(v < 0))
    if op == 'relu':
        return max(v, 0.)
    if op == 'step':
        return float(v > 0)
    return math.tanh(v)


class Graph:
    """Expression nodes, each stored once: ('in', kind, i), ('lin', c0, ((c, node), ...)), ('mul', a, b),
    ('div', a, b), (unary, a), ('hold', a). Arguments are node numbers; a node's arguments come before it."""

    def __init__(self):
        self.nodes, self.key = [], {}

    def _node(self, nd):
        i = self.key.get(nd)
        if i is None:
            i = len(self.nodes); self.nodes.append(nd); self.key[nd] = i
        return i

    def leaf(self, kind, i):
        return self._node(('in', kind, i))

    def const(self, v):
        return self._node(('lin', float(v)+0., ()))

    def value(self, i):
        nd = self.nodes[i]
        return nd[1] if nd[0] == 'lin' and not nd[2] else None

    def _terms(self, i):
        """The node as c0 + sum of terms: a short sum is taken apart, anything else is one term."""
        nd = self.nodes[i]
        if nd[0] == 'lin' and len(nd[2]) <= INLINE:
            return nd[1], nd[2]
        return 0., ((1., i),)

    def lincomb(self, c0, terms):
        """c0 + sum c node over (c, node) pairs, merged and stored in canonical form."""
        acc = {}
        for c, i in terms:
            if c == 0.:
                continue
            b0, bt = self._terms(i)
            c0 += c*b0
            for d, j in bt:
                acc[j] = acc.get(j, 0.)+c*d
        t = tuple(sorted((float(c)+0., j) for j, c in acc.items() if c != 0.))
        if len(t) == 1 and t[0][0] == 1. and c0 == 0.:
            return t[0][1]
        return self._node(('lin', float(c0)+0., tuple((c, j) for c, j in t)))

    def add(self, i, j):
        return self.lincomb(0., ((1., i), (1., j)))

    def scale(self, i, s):
        return self.lincomb(0., ((s, i),))

    def _factor(self, i):
        """The node as s times a node (a scaled single term gives up its factor)."""
        nd = self.nodes[i]
        if nd[0] == 'lin' and nd[1] == 0. and len(nd[2]) == 1:
            return nd[2][0]
        return 1., i

    def mul(self, i, j):
        vi, vj = self.value(i), self.value(j)
        if vi is not None:
            return self.scale(j, vi)
        if vj is not None:
            return self.scale(i, vj)
        (si, bi), (sj, bj) = self._factor(i), self._factor(j)
        return self.scale(self._node(('mul', min(bi, bj), max(bi, bj))), si*sj)

    def div(self, i, j):
        vi, vj = self.value(i), self.value(j)
        if vj is not None:
            return self.scale(i, 1./vj)
        if vi == 0.:
            return i
        (si, bi), (sj, bj) = ((vi, self.const(1.)) if vi is not None else self._factor(i)), self._factor(j)
        return self.scale(self._node(('div', bi, bj)), si/sj)

    def un(self, op, i):
        v = self.value(i)
        if v is not None:
            return self.const(_fold(op, v))
        return self._node((op, i))

    def hold(self, i):
        """The same value as one term: a long sum used in many places is computed once."""
        return i if self.nodes[i][0] != 'lin' else self._node(('hold', i))

    def args(self, i):
        nd = self.nodes[i]
        if nd[0] == 'in':
            return ()
        if nd[0] == 'lin':
            return tuple(j for _, j in nd[2])
        return nd[1:]

    def reach(self, roots):
        seen, stack = set(), list(roots)
        while stack:
            i = stack.pop()
            if i not in seen:
                seen.add(i); stack.extend(self.args(i))
        return sorted(seen)

    def diff(self, roots, wrt):
        """{node: {leaf: node}} for every node the roots depend on: its derivative with respect to each of the
        leaves `wrt` it depends on (absent: zero)."""
        wrt, one, D = set(wrt), self.const(1.), {}
        for i in self.reach(roots):
            nd = self.nodes[i]; op = nd[0]
            if op == 'in':
                D[i] = {i: one} if i in wrt else {}
            elif op == 'lin':
                acc = {}
                for c, k in nd[2]:
                    for v, d in D[k].items():
                        acc.setdefault(v, []).append((c, d))
                D[i] = {v: self.lincomb(0., t) for v, t in acc.items()}
            elif op == 'hold':
                D[i] = D[nd[1]]
            elif op in ('mul', 'div'):
                a, b = nd[1], nd[2]
                out = {}
                for v in set(D[a]) | set(D[b]):
                    if op == 'mul':
                        t = [(1., self.mul(b, D[a][v]))] if v in D[a] else []
                        if v in D[b]:
                            t.append((1., self.mul(a, D[b][v])))
                        out[v] = self.lincomb(0., t)
                    else:
                        t = [(1., D[a][v])] if v in D[a] else []
                        if v in D[b]:
                            t.append((-1., self.mul(i, D[b][v])))
                        out[v] = self.div(self.lincomb(0., t), b)
                D[i] = out
            elif op == 'exp':
                D[i] = {v: self.mul(i, d) for v, d in D[nd[1]].items()}
            elif op == 'log':
                D[i] = {v: self.div(d, nd[1]) for v, d in D[nd[1]].items()}
            elif op == 'sqrt':
                D[i] = {v: self.div(d, self.scale(i, 2.)) for v, d in D[nd[1]].items()}
            elif op == 'abs':
                D[i] = {v: self.mul(self.un('sign', nd[1]), d) for v, d in D[nd[1]].items()}
            elif op == 'relu':
                D[i] = {v: self.mul(self.un('step', nd[1]), d) for v, d in D[nd[1]].items()}
            elif op == 'tanh':
                D[i] = {v: self.mul(self.lincomb(1., ((-1., self.mul(i, i)),)), d) for v, d in D[nd[1]].items()}
            else:                                                 # sign, step: flat
                D[i] = {}
        return D


class E:
    """An expression: a node of a graph with the arithmetic operators."""
    __slots__ = ('g', 'i')

    def __init__(self, g, i):
        self.g, self.i = g, i

    def _id(self, o):
        return o.i if isinstance(o, E) else self.g.const(o)

    def __add__(self, o):
        return E(self.g, self.g.add(self.i, self._id(o)))
    __radd__ = __add__

    def __sub__(self, o):
        return E(self.g, self.g.lincomb(0., ((1., self.i), (-1., self._id(o)))))

    def __rsub__(self, o):
        return E(self.g, self.g.lincomb(0., ((-1., self.i), (1., self._id(o)))))

    def __neg__(self):
        return E(self.g, self.g.scale(self.i, -1.))

    def __mul__(self, o):
        return E(self.g, self.g.mul(self.i, self._id(o)))
    __rmul__ = __mul__

    def __truediv__(self, o):
        return E(self.g, self.g.div(self.i, self._id(o)))

    def __rtruediv__(self, o):
        return E(self.g, self.g.div(self._id(o), self.i))

    def __pow__(self, o):
        if isinstance(o, E):
            v = self.g.value(o.i)
            if v is None:
                return exp(o*log(self))
            o = v
        if o == 1.:
            return self
        if o == 0.:
            return E(self.g, self.g.const(1.))
        if o == .5:
            return sqrt(self)
        if o == -1.:
            return 1./self
        if float(o).is_integer() and 2 <= o <= 4:
            r = self
            for _ in range(int(o)-1):
                r = r*self
            return r
        return exp(o*log(self))

    @property
    def held(self):
        return E(self.g, self.g.hold(self.i))


def _un(op):
    def f(a):
        return E(a.g, a.g.un(op, a.i))
    return f


exp, abs_, sign, relu, step, tanh = (_un(q) for q in ('exp', 'abs', 'sign', 'relu', 'step', 'tanh'))


def log(a, eps=TINY):
    """log(max(a, eps))."""
    return E(a.g, a.g.un('log', (eps+relu(a-eps)).i))


def sqrt(a, eps=1e-12):
    """sqrt(max(a, eps)): the slope stays finite at zero."""
    return E(a.g, a.g.un('sqrt', (eps+relu(a-eps)).i))


def maximum(a, b):
    a, b = (a, b) if isinstance(a, E) else (b, a)
    return a+relu(b-a)


def minimum(a, b):
    a, b = (a, b) if isinstance(a, E) else (b, a)
    return a-relu(a-b)


def clamp(a, lo=None, hi=None):
    if lo is not None:
        a = maximum(a, lo)
    if hi is not None:
        a = minimum(a, hi)
    return a


def where(c, a, b):
    """a where c > 0, else b."""
    return b+step(c)*(a-b)


# ---------------------------------------------------------------- compilation
class Table:
    """A compiled set of expressions. Registers: 1, 0, the states, the inputs, the parameters, then the
    nodes level by level and class by class. `count` (levels, classes) is the table's shape; operands are
    kept as symbolic addresses (space, index) so that tables of different shapes can share one layout:
    space -1 a fixed register (0: one, 1: zero), -2 x, -3 w, -4 p, else level*len(CLS)+class."""

    def __init__(self, count, ops, outs, M):
        self.count, self.ops, self.outs, self.M = count, ops, outs, M

    @property
    def nodes(self):
        return int(self.count.sum())


def compile_table(g, outs, M=4):
    """outs: {name: [node, ...]}. Returns a Table whose outputs are the registers of those nodes."""
    sub = {}

    def res(i):
        """The node to compute for i: holds resolved, sums of more than M terms split into sums of sums."""
        j = sub.get(i)
        if j is not None:
            return j
        nd = g.nodes[i]
        if nd[0] == 'hold':
            j = res(nd[1])
        elif nd[0] == 'lin' and len(nd[2]) > M:
            t = [(c, res(k)) for c, k in nd[2]]
            while len(t) > M:
                t = [(1., g._node(('lin', 0., tuple(t[a:a+M])))) if len(t[a:a+M]) > 1 else t[a] for a in range(0, len(t), M)]
            j = g._node(('lin', nd[1], tuple(t)))
        elif nd[0] == 'lin':
            j = g._node(('lin', nd[1], tuple((c, res(k)) for c, k in nd[2])))
        elif nd[0] == 'in':
            j = i
        else:
            j = g._node((nd[0],)+tuple(res(k) for k in nd[1:]))
        sub[i] = j
        return j
    roots = {q: [res(i) for i in v] for q, v in outs.items()}
    order = g.reach([i for v in roots.values() for i in v])
    level = {}
    for i in order:
        nd = g.nodes[i]
        level[i] = 0 if nd[0] == 'in' else 1+max((level[j] for j in g.args(i)), default=0)
    L = max(level.values(), default=0)
    slots = [[[] for _ in CLS] for _ in range(L+1)]
    place = {}
    for i in order:
        nd = g.nodes[i]
        if nd[0] == 'in':
            place[i] = (-2-'xwp'.index(nd[1]), nd[2])
        elif g.value(i) in (0., 1.):                              # the two fixed registers
            place[i] = (-1, int(g.value(i) == 0.))
        else:
            c = CLS.index(nd[0])
            place[i] = (level[i]*len(CLS)+c, len(slots[level[i]][c])); slots[level[i]][c].append(i)
    addr = place.__getitem__
    ops = {}
    for l in range(1, L+1):
        for c, name in enumerate(CLS):
            ids = slots[l][c]
            if not ids:
                continue
            if name == 'lin':
                ref = np.full((len(ids), M, 2), (-1, 1), dtype=np.int64); coef = np.zeros((len(ids), M)); c0 = np.zeros(len(ids))
                for a, i in enumerate(ids):
                    c0[a] = g.nodes[i][1]
                    for b, (cf, j) in enumerate(g.nodes[i][2]):
                        ref[a, b], coef[a, b] = addr(j), cf
                ops[(l, c)] = (ref, coef, c0)
            else:
                ops[(l, c)] = (np.array([[addr(j) for j in g.nodes[i][1:]] for i in ids], dtype=np.int64),)
    count = np.array([[len(s) for s in lv] for lv in slots], dtype=np.int64)
    return Table(count, ops, {q: np.array([addr(i) for i in v], dtype=np.int64).reshape(-1, 2) for q, v in roots.items()}, M)
