"""Build the native Tennessee Eastman library tep_native_record.py reads: a standalone reentrant DLL from the
pinned author archive and the wrapper tep_native_adapter.c (Windows, MSVC).

    python tep_native_build.py --upstream DIR_WITH_THE_UNPACKED_ARCHIVE --out NEW_DIR

Upstream sources stay in the ignored run directory; this adapter does not
redistribute the archive or claim a license for the original model.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

URL = "https://depts.washington.edu/control/LARRY/TE/temexd_mod.zip"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(upstream, out):
    upstream, out = Path(upstream).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    src = (upstream / "temexd_mod.c").read_text(encoding="latin1")
    header = (upstream / "teprob_mod.h").read_text(encoding="latin1")
    # Keep all physical functions and numerical constants byte-for-byte at
    # the text level. Remove only MATLAB entry points and their prototypes.
    header = "\n".join(x for x in header.splitlines()
                       if "SimStruct" not in x)
    (out / "teprob_mod.h").write_text(header, encoding="latin1")
    core = src[src.index("struct ProcessValues {"):src.index(
        "/**************************************************************************\n*                                                                         *\n*                  S - F U N C T I O N - M E T H O D S")]
    physics = src[src.index("static int teinit(void *ModelData,"):src.index(
        "/*=============================*")]
    prefix = '#include <math.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <string.h>\n'
    prefix += '#define min(a,b) ((a)<(b)?(a):(b))\n#define max(a,b) ((a)>(b)?(a):(b))\n#include "teprob_mod.h"\n'
    wrapper = Path(__file__).with_name("tep_native_adapter.c").read_text()
    (out / "standalone.c").write_text(prefix + core + physics + wrapper, encoding="latin1")
    vswhere = "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    vs = subprocess.check_output([vswhere, "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
        "installationPath"], text=True).strip()
    setup = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
    for p in [out, setup]:
        if any(c in str(p) for c in '&|<>^"\n\r%'):
            raise ValueError("Unsupported path")
    command = f'@echo off\ncall "{setup}" >nul\nif errorlevel 1 exit /b 1\ncl /nologo /LD /O2 /fp:strict standalone.c /link /OUT:te.dll\n'
    (out / "compile.bat").write_text(command)
    r = subprocess.run(["cmd.exe", "/d", "/c", "compile.bat"], cwd=out, capture_output=True)
    (out / "compiler.log").write_bytes(r.stdout + r.stderr)
    r.check_returncode()
    receipt = dict(url=URL, upstream={p.name:sha(p) for p in upstream.iterdir() if p.is_file()},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        adapter_sha256=sha(Path(__file__).with_name("tep_native_adapter.c")),
        flags="/O2 /fp:strict", compiler=vs)
    (out / "build.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--upstream",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();build(a.upstream,a.out)
