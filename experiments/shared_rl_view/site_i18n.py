"""Chinese checks for the bilingual AIO research site (user request, 2026-11-02).

The page writes every fixed string as tr(en, zh) next to the code that shows it, so a
missing translation is a TypeScript error; `missing_tr` still catches an empty Chinese
argument. Data files carry Chinese in sibling ``<field>_zh`` fields, written by their
exporters (site_progress_notes.json, research_site_replay_v2, research_site_tep_replay_v2,
research_site_ph_replay); `missing_zh` lists displayed text without one. Both warn: the
page then shows English for that string.
"""
import re
import sys
from pathlib import Path

# fields the page shows through pick(obj, key)
SHOWN = ('label', 'short', 'note', 'footnote', 'text', 'title', 'scope', 'pill', 'replay', 'lede',
         'oracle_note', 'oracle_status', 'status_note')
# data the page never shows as text: problem logs and file names
SKIP = ('problems', 'source', 'report')
EMPTY_ZH = re.compile(r'''\btr\(\s*(?:"(?:[^"\\]|\\.)+"|`[^`]+`|'(?:[^'\\]|\\.)+')\s*,\s*(?:""|``|'')\s*\)''')


def missing_zh(data, where):
    """Paths of displayed strings in a data file that have no '<field>_zh' companion."""
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in SKIP:
                    continue
                if k in SHOWN and isinstance(v, str) and v.strip() and not o.get(f'{k}_zh'):
                    out.append(f'{path}.{k}')
                walk(v, f'{path}.{k}')
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f'{path}[{i}]')
    walk(data, '')
    for p in out:
        print(f'warning: no Chinese for {where} {p}; it will show in English', file=sys.stderr)
    return out


def missing_tr(src):
    """tr(en, zh) calls in the page source whose Chinese argument is empty."""
    out = []
    for f in sorted(Path(src).rglob('*.ts*')):
        for m in EMPTY_ZH.finditer(f.read_text(encoding='utf-8')):
            out.append(f'{f.name}: {m.group(0)[:60]}')
    for p in out:
        print(f'warning: empty Chinese in {p}', file=sys.stderr)
    return out
