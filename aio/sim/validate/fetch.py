"""Third-party code and data the validation uses: where each comes from, and a downloader.

None of it is in this repository. `python -m aio.sim.validate.fetch [names]` downloads into
$AIO_VALIDATION_DATA (default: data/validation/, ignored by git), one folder per source, and writes the
record's metadata next to the files. Run it on a machine that reaches the hosts (the cloud sessions of this
project do not reach zenodo.org, skoge.folk.ntnu.no, data.4tu.nl or the KU Leuven servers).

  python -m aio.sim.validate.fetch            # everything but the large ones
  python -m aio.sim.validate.fetch pronto     # one source by name
  python -m aio.sim.validate.fetch --list
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get('AIO_VALIDATION_DATA', Path(__file__).resolve().parents[3]/'data'/'validation'))
SKIP_OVER = 3 << 30      # bytes: a single file larger than this is left out (raw images, audio, spectra)

# name: (kind, where, what it is, licence, large)
SOURCES = dict(
    skogestad=('files', 'https://skoge.folk.ntnu.no/book/matlab_m/cola/', 'Skogestad column A: colamod.m, cola_lv.m and the scripts around them (MATLAB)', 'none stated', False),
    csth=('zenodo', '10093059', 'Continuous stirred tank heater pilot plant (Thornhill, Patwardhan, Shah 2008): model, measured disturbances and noise', 'CC BY 4.0', False),
    tum_column=('zenodo', '17628963', 'Continuous distillation pilot plant, 580 h, 30 s (TUM, Sci. Data 2026)', 'CC BY 4.0', False),
    batch_distillation=('zenodo', '20411905', 'Batch distillation, 119 runs with paired simulations (Arweiler et al.)', 'CC BY 4.0', True),
    pronto=('zenodo', '1341583', 'PRONTO multiphase flow facility, Cranfield (process data, alarms, induced faults)', 'CC BY 4.0', True),
    manipal_column=('zenodo', '22763516', 'Pilot distillation column, open-loop steps 2x2 (MIT Manipal 2026)', 'CC BY 4.0', False),
    daisy=('files', 'https://ftp.esat.kuleuven.be/pub/SISTA/data/', 'DaISy: process_industry/ and thermic/ records (KU Leuven)', 'free for research, cite DaISy', False),
    cascaded_tanks=('files', 'https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks', 'Cascaded tanks with overflow (Schoukens, Mattsson, Wigren, Noel; data at data.4tu.nl)', 'CC BY 4.0', False),
    vam=('files', 'https://terpconnect.umd.edu/~mcavoy/VAC%20Material/', 'Vinyl acetate monomer plant, C and MATLAB (Chen, Dave, McAvoy, Luyben 2003)', 'none stated', False),
)
COLA = ('colamod.m', 'cola_lv.m', 'cola4.m', 'cola_init.m', 'cola_init.mat', 'cola_test.m', 'cola_simf.m', 'cola_lv_lin.m', 'cola_linearize.m', 'cola.html')
DAISY = dict(process_industry=('cstr', 'destill', 'distill2', 'dryer2', 'evaporator', 'exchanger', 'glassfurnace', 'pHdata', 'powerplant', 'steamgen', 'winding'),
             thermic=('heating_system', 'thermic_res_wall'))


def get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size:
        return
    req = urllib.request.Request(url, headers={'User-Agent': 'aio-validation-fetch'})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)


def zenodo(rec, dest):
    meta = dest/'record.json'
    get(f'https://zenodo.org/api/records/{rec}', meta)
    for f in json.loads(meta.read_text())['files']:
        if f['size'] > SKIP_OVER:
            print('  skipped (size)', f['key'], f['size'])
            continue
        get(f['links']['self'], dest/f['key'])


def fetch(name):
    kind, where, what, lic, _ = SOURCES[name]
    dest = ROOT/name
    print(name, '->', dest)
    if kind == 'zenodo':
        zenodo(where, dest)
    elif name == 'skogestad':
        for f in COLA:
            try:
                get(where+f, dest/f)
            except Exception as e:                                # a file the server does not have is not an error of the set
                print('  missing', f, e)
    elif name == 'daisy':
        for sub, names in DAISY.items():
            for n in names:
                for ext in ('.dat.gz', '.txt'):
                    try:
                        get(f'{where}{sub}/{n}{ext}', dest/sub/(n+ext))
                    except Exception as e:
                        print('  missing', n+ext, e)
    else:
        print('  browse and download by hand:', where)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--list' in sys.argv:
        for k, (kind, where, what, lic, large) in SOURCES.items():
            print(f'{k:20s} {what}\n{"":20s} {("zenodo.org/records/"+where) if kind == "zenodo" else where}   [{lic}]{"   large" if large else ""}')
    else:
        for k in args or [k for k, v in SOURCES.items() if not v[4]]:
            fetch(k)
