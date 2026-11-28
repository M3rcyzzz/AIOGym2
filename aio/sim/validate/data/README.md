# Reference trajectories

- `bsm1_dry.npz`: BSM1 open loop on the dry-weather influent, computed with bsm2-python 0.0.16
  (https://github.com/FAU-EVT/bsm2-python, BSD 3-Clause, Copyright (c) 2025 FAU-EVT) by
  `python -m aio.sim.validate.bsm record`. Read by `aio/sim/validate/bsm.py` and `tests/test_plants_bsm.py`.
- `tep_native.npz`: state trajectories of the native Tennessee Eastman build on the training machine,
  recorded by `aio/sim/validate/tep_native_record.py` (the library: `tep_native_build.py`). Numbers only; the Tennessee Eastman source is not in
  this repository. Read by `aio/sim/validate/tep.py` and `tests/test_plants_tep.py`.
- `vam_native.npz`: state trajectories of the native vinyl acetate plant (C routines of Chen, Dave, McAvoy, Luyben 2003, UMD site),
  recorded by `aio/sim/validate/vam_native_record.py`. Numbers only; the C source is not in this repository. Read by
  `aio/sim/validate/vam.py` and `tests/test_vam.py`.
- `boiler_plant.json`: the six open-loop experiments of Astrom & Bell (Automatica 36 (2000) 363, Figs. 8-13) read off the figures
  (input, drum pressure, drum level at 10 s). Read by `aio/sim/validate/boiler.py` and `tests/test_boiler2.py`.
- `gaslib_morgen.npz`: morgen 1.2 (github.com/gramian/morgen, BSD-2) runs of GasLib-24 (gaslib.zib.de, CC BY 3.0) under the TRR154
  transient exit flows (github.com/TRR154/InstationaryGasNetworks, Apache-2.0), recorded by `python -m aio.sim.validate.gaslib record`.
  Read by `aio/sim/validate/gaslib.py` and `tests/test_gaslib.py`.
- `refrig_reference.npz`: the two reference runs shipped with the IFAC PID 2018 refrigeration benchmark (www.dia.uned.es/~fmorilla/benchmarkPID2018,
  RSBenchmark_20170524_13_48 and _20170528_22_34): inputs and the outputs compared, a sample a second. Read by `aio/sim/validate/refrig.py`.
- `badgwell_heater.npz`: A, B, C, D of HeaterModelCont.mat and the run heaterData.mat of Badgwell's fired heater case (Georgia Tech ChBE 4412
  fired heater package). Read by `aio/sim/validate/badgwell.py`.
