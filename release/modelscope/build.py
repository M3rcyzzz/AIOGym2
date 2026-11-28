"""Build the ModelScope copy of the AIO1 release from the Hugging Face staging folder.

ModelScope shows README.md, so the Chinese card (README_zh.md) becomes README.md with ModelScope
front matter, and the English HF card becomes README_en.md. Everything else (model.safetensors, config.json,
code/, modeling_aio.py, LICENSE and eval/ if any) is copied unchanged, so the two hubs carry the same files.

  python release/modelscope/build.py --hf HF_STAGING_DIR --out MS_STAGING_DIR --ms-repo OWNER/AIO1
then upload with: modelscope upload OWNER/AIO1 MS_STAGING_DIR --repo-type model
(that upload deletes nothing: what is no longer in the folder goes with HubApi().delete_files).
"""
import argparse, re, shutil
from pathlib import Path

FRONT = """---
frameworks:
- Pytorch
license: MIT
language:
- zh
- en
tags:
- process-control
- advanced-process-control
- safetensors
---

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', type=Path, required=True, help='HF staging dir (README.md, README_zh.md, weights ...)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ms-repo', required=True, help='ModelScope repo id, e.g. OWNER/AIO1')
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    shutil.copytree(a.hf, a.out, ignore=shutil.ignore_patterns('.cache', '.git*', '__pycache__'))
    en = (a.hf/'README.md').read_text(encoding='utf-8')
    en = re.sub(r'\A---\n.*?\n---\n+', '', en, flags=re.S)           # HF front matter
    en = en.replace('[中文](README_zh.md)', '[中文](README.md)')
    en = en.replace('[README_zh.md](README_zh.md)', '[README.md](README.md)')
    zh = (a.hf/'README_zh.md').read_text(encoding='utf-8')
    zh = zh.replace('[English](README.md)', '[English](README_en.md)')
    note = (f'> 本仓库与 Hugging Face [M3rcyzzz/AIO1](https://huggingface.co/M3rcyzzz/AIO1) 同步，文件相同。'
            f'ModelScope 下载：`from modelscope import snapshot_download; snapshot_download("{a.ms_repo}")`。\n\n')
    zh = zh.replace('\n\nAIO1 是', '\n\n'+note+'AIO1 是', 1)
    en = en.replace('\n\n**AIO1** is', f'\n\n> Mirror of Hugging Face [M3rcyzzz/AIO1](https://huggingface.co/M3rcyzzz/AIO1) on ModelScope ({a.ms_repo}); identical files.\n\n**AIO1** is', 1)
    if 'Mirror of Hugging Face' not in en:
        en = en.replace('\n\nAIO1 is', f'\n\n> Mirror of Hugging Face [M3rcyzzz/AIO1](https://huggingface.co/M3rcyzzz/AIO1) on ModelScope ({a.ms_repo}); identical files.\n\nAIO1 is', 1)
    (a.out/'README.md').write_text(FRONT+zh, encoding='utf-8')
    (a.out/'README_en.md').write_text(en, encoding='utf-8')
    (a.out/'README_zh.md').unlink()
    print(sorted(str(p.relative_to(a.out)) for p in a.out.rglob('*') if p.is_file())[:8], '...')


if __name__ == '__main__':
    main()
