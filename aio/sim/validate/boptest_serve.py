"""BOPTEST's test cases behind its REST interface, in one process, for a machine where the full service
(web, Redis, object store) cannot be built.

Runs inside the image of BOPTEST's own worker (service/worker/Dockerfile: Ubuntu 20.04, PyFMI 2.14), where
`testcase.TestCase`, the KPI calculator and the forecaster are the project's unchanged code; this file only
passes requests to the methods the worker's job calls (service/worker/jobs/boptest_run_test/job.py) and
answers in the same shape: {"status", "message", "payload"}. Standard library only.

    python boptest_serve.py TESTCASES_DIR PORT

One process serves one request at a time; start one per core on neighbouring ports to run tests side by side.
"""
import json
import os
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from testcase import TestCase

ROOT, TESTS = None, {}
UNCERTAINTY = '/boptest/lib/forecast/forecast_uncertainty_params.json'


def plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def answer(r):
    return {'status': r[0], 'message': r[1], 'payload': r[2]}


def call(method, name, tc, q):
    """The worker's handlers, one line each."""
    if (method, name) == ('PUT', 'initialize'):
        end = q.get('final_time')
        return tc.initialize(q.get('start_time'), q.get('warmup_period'), float(end) if end else np.inf)
    if (method, name) == ('POST', 'advance'):
        return tc.advance(q)
    if (method, name) == ('PUT', 'step'):
        return tc.set_step(q['step'])
    if (method, name) == ('PUT', 'results'):
        return tc.get_results(q['point_names'], q['start_time'], q['final_time'])
    if (method, name) == ('PUT', 'forecast'):
        return tc.get_forecast(q['point_names'], q['horizon'], q['interval'])
    if (method, name) == ('PUT', 'scenario'):
        return tc.set_scenario({k: q.get(k) for k in ('electricity_price', 'time_period', 'temperature_uncertainty', 'solar_uncertainty', 'seed')})
    get = dict(name=tc.get_name, inputs=tc.get_inputs, measurements=tc.get_measurements, step=tc.get_step, kpi=tc.get_kpis,
               scenario=tc.get_scenario, forecast_points=tc.get_forecast_points, version=tc.get_version)
    if method == 'GET' and name in get:
        return get[name]()
    raise KeyError(name)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body):
        raw = json.dumps(body, default=plain, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def handle_any(self, method):
        parts = [p for p in self.path.split('?')[0].split('/') if p]
        n = int(self.headers.get('Content-Length') or 0)
        q = json.loads(self.rfile.read(n) or b'{}') if n else {}
        try:
            if parts == ['testcases']:
                return self.send(200, [{'testcaseid': d} for d in sorted(os.listdir(ROOT)) if os.path.isfile(os.path.join(ROOT, d, 'models', 'wrapped.fmu'))])
            if parts == ['version']:
                return self.send(200, {'status': 200, 'message': 'Queried the version number successfully.', 'payload': {'version': open('/version.txt').read()}})
            if len(parts) == 3 and parts[0] == 'testcases' and parts[2] == 'select' and method == 'POST':
                fmu = os.path.join(ROOT, parts[1], 'models', 'wrapped.fmu')
                if not os.path.isfile(fmu):
                    return self.send(404, {'message': 'Invalid testcase'})
                tid = str(uuid.uuid4())
                TESTS[tid] = TestCase(fmu, UNCERTAINTY)
                return self.send(200, {'testid': tid})
            name, tid = parts
            if (method, name) == ('PUT', 'stop'):
                TESTS.pop(tid, None)
                return self.send(200, {})
            if (method, name) == ('GET', 'status'):
                return self.send(200, 'Running' if tid in TESTS else 'Unknown')
            if tid not in TESTS:
                return self.send(404, {'message': 'Invalid testid'})
            return self.send(200, answer(call(method, name, TESTS[tid], q)))
        except Exception as e:
            return self.send(400, {'status': 400, 'message': 'Bad Request', 'payload': repr(e)})

    def do_GET(self):
        self.handle_any('GET')

    def do_PUT(self):
        self.handle_any('PUT')

    def do_POST(self):
        self.handle_any('POST')


if __name__ == '__main__':
    ROOT = sys.argv[1]
    os.chdir('/')                                   # TestCase reads version.txt from the working directory
    HTTPServer(('127.0.0.1', int(sys.argv[2])), Handler).serve_forever()
