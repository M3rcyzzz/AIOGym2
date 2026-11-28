"""Publish https://m3rcyzzz.club/aio/ (bilingual English / 简体中文).

The page is the React app in monitoring/research-site, built to its dist/ folder
(`npm ci && npm run build` there; dist/ is committed so publishing needs no Node). `build`
copies dist/, the exam replays the notes list (--replay, else the live ones) and writes a fresh progress.json
(site_progress.py: curated notes and the training runs read from baolong over ssh).
`deploy` switches the site link atomically and removes the superseded release
(docs/site-release-retention.md). If the new page does not load, the link goes back to
the previous release. Re-run both steps at every milestone (user request, 2026-11-02:
keep the site current).

  python -m experiments.shared_rl_view.publish_bilingual_site build  --work W --release R --host H [--baolong baolong|local] [--replay sim-replay.json ...]
  python -m experiments.shared_rl_view.publish_bilingual_site deploy --work W --release R --host H
`build` writes W/R for a local look; `deploy` uploads it. Python >= 3.10, no extra packages.
"""
import argparse
import json
import shutil
import sys
import subprocess
import time
import tarfile
from pathlib import Path

from . import site_i18n, site_progress

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'monitoring/research-site'
DIST = SOURCE / 'dist'
RELEASES = '/home/ubuntu/aio-releases'
LINK = '/home/ubuntu/decks/aio'
URL = 'https://m3rcyzzz.club/aio/'


def ssh(host, *cmd, **kw):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, *cmd], check=True, **kw)


def remote_python(host, code):
    r = ssh(host, 'python3', '-', input=code, text=True, capture_output=True)
    return json.loads(r.stdout)


def note(problems, found):
    for pr in found:
        print(f"{pr['level']}: {pr['text']}", file=sys.stderr)
        problems.append(pr)


def write(path, data, ascii_only=False):
    # allow_nan=False: browsers refuse NaN/Infinity in JSON, so a non-finite number stops the build instead of the page
    path.write_text(json.dumps(data, ensure_ascii=ascii_only, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


def strict_json(out):
    """Every JSON file in the release parses as browsers parse it: no NaN or Infinity (11-09, a -inf training
    return reached progress.json and the live page could not read it). Copied files are checked too."""
    def refuse(c):
        raise ValueError(f'{c} is not valid JSON for a browser')
    for f in sorted(out.rglob('*.json')):
        try:
            json.loads(f.read_text(encoding='utf-8'), parse_constant=refuse)
        except ValueError as e:
            sys.exit(f'{f.relative_to(out)}: {e}')


def build(a):
    work = Path(a.work); out = work / a.release
    if not (DIST / 'index.html').exists():
        sys.exit(f'{DIST} is missing: run `npm ci && npm run build` in {SOURCE}')
    work.mkdir(parents=True, exist_ok=True)
    live = remote_python(a.host, f"import json,pathlib;print(json.dumps(str(pathlib.Path({LINK!r}).resolve())))")
    name = Path(live).name
    assert name != a.release, 'release name is already live'
    with (work / 'live.tgz').open('wb') as f:
        ssh(a.host, 'tar', '-C', RELEASES, '-czf', '-', name, stdout=f)
    with tarfile.open(work / 'live.tgz') as tar:
        tar.extractall(work / 'live', **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
    old = work / 'live' / name
    previous = json.loads((old / 'progress.json').read_text(encoding='utf-8')) if (old / 'progress.json').exists() else None
    progress = site_progress.build(a.baolong, previous)
    problems = progress['freshness']['problems']
    for pr in problems:
        print(f"{pr['level']}: {pr['text']}", file=sys.stderr)
    if any(pr['level'] == 'error' for pr in problems) and not a.allow_stale:
        sys.exit('refusing to build: the page would show out-of-date text (see the errors above; --allow-stale overrides)')
    shutil.copytree(DIST, out)  # fails if the release folder exists
    # the exam replays the page lists (notes `replays`, research_site_sim_replay): a file given with --replay, else the live one
    given = {Path(f).name: Path(f) for f in a.replay or []}
    names = [r['file'] for r in progress.get('replays') or []]
    assert set(given) <= set(names), f'--replay files not listed in the notes replays: {sorted(set(given) - set(names))}'
    for rep in names:
        src = given.get(rep) or old / rep
        if not src.exists():
            print(f'warning: replay {rep} is neither given nor live; the page shows it as unavailable', file=sys.stderr)
            continue
        d = json.loads(src.read_text(encoding='utf-8'))
        assert d.get('version') == 1 and d.get('process') == 'sim', f'{src} is not a replay of the core (research_site_sim_replay)'
        site_i18n.missing_zh(d, rep)
        write(out / rep, d, ascii_only=True)
    site_i18n.missing_zh(progress, 'progress.json')
    write(out / 'progress.json', progress)
    html = (out / 'index.html').read_text(encoding='utf-8')
    assert 'RELEASE_ID' in html and 'UPDATE_TIME_ISO' in html, 'dist/index.html lacks the release placeholders'
    (out / 'index.html').write_text(html.replace('RELEASE_ID', a.release).replace('UPDATE_TIME_ISO', progress['generated_at']), encoding='utf-8')
    site_i18n.missing_tr(SOURCE / 'src')
    strict_json(out)
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
    manifest = dict(release=a.release, base=name, scope='Research site: React page (dist), exam replays, progress.json',
                    training_through=progress['training_through'], files=files)
    (out / 'publication-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(built=str(out), base=name, training_through=progress['training_through'], freshness=progress['freshness']['blocks'],
                          runs={r['id']: [(s['seed'], s['state'], s['samples']) for s in r['seeds']] for r in progress['runs']},
                          files=files)))


def live_check(release, tries=3, wait=5, run=subprocess.run, sleep=time.sleep):
    """The new index.html is served (it names its release). Retried: the site's tunnel sometimes
    stalls one response for 40 s (11-03), and a single stall must not roll back a good release."""
    for i in range(tries):
        try:
            # curl, not urllib: macOS Python may lack CA certificates. The query string skips
            # any CDN copy of the old file.
            r = run(['curl', '-fsS', '--max-time', '30', '-H', 'Cache-Control: no-cache',
                     f'{URL}index.html?release={release}'], capture_output=True, check=True)
            assert f'content="{release}"'.encode() in r.stdout, 'the site still serves the old page'
            return
        except Exception as e:
            if i == tries-1:
                raise
            print(f'live check {i+1}/{tries} failed ({type(e).__name__}); retrying in {wait} s', file=sys.stderr)
            sleep(wait)


def deploy(a):
    work = Path(a.work); out = work / a.release
    archive = work / (a.release + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=a.release)
    upload = f'/home/ubuntu/{archive.name}'
    subprocess.run(['scp', '-o', 'BatchMode=yes', str(archive), f'{a.host}:{upload}'], check=True)
    switched = remote_python(a.host, f'''
import json,os,pathlib,shutil,tarfile
root=pathlib.Path({RELEASES!r});link=pathlib.Path({LINK!r});new=root/{a.release!r};up=pathlib.Path({upload!r})
old=link.resolve();assert not new.exists()
with tarfile.open(up) as t:t.extractall(root,**({{'filter':'data'}} if hasattr(tarfile,'data_filter') else {{}}))
up.unlink()
live_status=old/'train-status.json'
if live_status.is_file() and not (new/'train-status.json').exists():shutil.copy2(live_status,new/'train-status.json')
tmp=link.with_name('aio.next')
if tmp.is_symlink():tmp.unlink()
tmp.symlink_to(new);os.replace(tmp,link)
print(json.dumps(dict(previous=str(old),live=str(link.resolve()))))
''')
    try:
        live_check(a.release)
    except Exception:
        remote_python(a.host, f'''
import json,os,pathlib,shutil
link=pathlib.Path({LINK!r});tmp=link.with_name('aio.back')
if tmp.is_symlink():tmp.unlink()
tmp.symlink_to({switched['previous']!r});os.replace(tmp,link);shutil.rmtree({switched['live']!r})
print(json.dumps(dict(rolled_back_to=str(link.resolve()))))
''')
        raise
    previous = switched['previous']
    retired = remote_python(a.host, f'''
import json,pathlib,shutil
root=pathlib.Path({RELEASES!r});old=pathlib.Path({previous!r});live=pathlib.Path({LINK!r}).resolve()
if old.parent==root and old!=live and old.is_dir() and not old.is_symlink():shutil.rmtree(old)
print(json.dumps(dict(live=str(live),removed=str(old),remaining=sorted(p.name for p in root.iterdir()))))
''')
    print(json.dumps(dict(switched, **retired)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['build', 'deploy'])
    p.add_argument('--work', required=True); p.add_argument('--release', required=True)
    p.add_argument('--host', required=True, help='ssh destination of the site server, e.g. user@host')
    p.add_argument('--baolong', default='baolong', help="ssh destination of the training machine, or 'local' when publishing on it (build only)")
    p.add_argument('--allow-stale', action='store_true', help='build even when the notes are behind the data (build only)')
    p.add_argument('--replay', action='append', help='an exam replay from research_site_sim_replay, named as in the notes replays (repeatable; build only; default: keep the live one)')
    a = p.parse_args()
    {'build': build, 'deploy': deploy}[a.command](a)
