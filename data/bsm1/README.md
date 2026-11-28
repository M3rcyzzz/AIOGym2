# BSM1 influent data

`dryinfluent.csv` is the IWA Benchmark Simulation Model No. 1 dry-weather influent file (14 days,
15-min samples). Columns: time [d], then SI, SS, XI, XS,
XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK, TSS, Q [m3/d], T [degC] and five dummy states.

Copied from bsm2-python 0.0.16 (https://github.com/FAU-EVT/bsm2-python, `src/bsm2_python/data/`),
which is distributed under the BSD 3-Clause License:

Copyright (c) 2025, FAU-EVT. Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: (1) redistributions
of source code must retain the above copyright notice, this list of conditions and the following
disclaimer; (2) redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution; (3) neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from this software without
specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

`aio/sim/validate/bsm.py` reads it.
