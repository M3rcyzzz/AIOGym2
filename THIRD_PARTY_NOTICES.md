# Third-Party Notices and Provenance

This repository is licensed under the **MIT License** ([LICENSE](LICENSE), Copyright (c) 2026 Mercy Zhang), except for specific third-party data files, third-party licensed benchmark assets, and industrial code-transcription modules itemized below.

All provenance, version, and license statements below are verified directly against the repository sources (`aio/sim/plants*.py`, `aio/sim/props_fit.py`, `aio/sim/validate/`, `data/`, `docs/validation.md`, and `docs/validation-data.md`).

---

## 1. MIT-Clean Core Mechanistic Implementations (24 Plant Families)

The following **24 mechanistic plant families** are independent PyTorch implementations built either from publicly published chemical/process engineering and control literature equations or as original self-built physical models. They contain no third-party source code and are distributed under the repository's **MIT License**:

### Default Pool (`aio.sim.plants.PLANTS`, 15 families)

| Key | Module | Provenance |
|---|---|---|
| `qtank` | `aio/sim/plants.py` | Independent PyTorch implementation of Johansson (2000) quadruple-tank equations. |
| `ph` | `aio/sim/plants.py` | Independent PyTorch implementation of Henson & Seborg (1994) pH neutralization equations. |
| `vdv` | `aio/sim/plants.py` | Independent PyTorch implementation of Klatt & Engell (1998) / Chen et al. (1995) Van de Vusse reactor equations. |
| `excstr` | `aio/sim/plants.py` | Independent PyTorch implementation of the exothermic CSTR from Seborg et al. (*Process Dynamics and Control*, Sec. 2.4.6 / Table 2.3; Hicks & Ray) with a jacket PI inner loop. |
| `bio` | `aio/sim/plants.py` | Independent PyTorch implementation of the substrate-inhibited continuous fermenter (Bequette 1998, Module 8; Henson & Seborg 1997, Ch. 4, after Jöbses). |
| `evap` | `aio/sim/plants.py` | Independent PyTorch implementation of Newell & Lee (1989, Ch. 2) forced-circulation evaporator equations. |
| `furnace` | `aio/sim/plants.py` | Original self-built two-pass fired heater model (no external paper or code). |
| `gasnet` | `aio/sim/plants.py` | Original self-built gas pressure-reducing station and line model (no external paper or code). |
| `hex` | `aio/sim/plants.py` | Original self-built counter-current heat-exchanger train with bypass (no external paper or code). |
| `column` | `aio/sim/plants.py` | Independent PyTorch implementation of Skogestad & Morari (1988) Distillation Column A equations (constant relative volatility, linearized tray hydraulics, LV level loops). Skogestad's original MATLAB `colamod.m`/`cola_lv.m` scripts are **not** distributed in this repository. |
| `bldg` | `aio/sim/plants_bldg.py` | Independent PyTorch implementation of a two-zone radiator heating plant using the VDI 6007 two-element reduced-order zone structure (IBPSA `ThermalZones.ReducedOrder.RC.TwoElements`) and EN 442-2 radiator equations (Modelica Buildings `RadiatorEN442_2`), with nominal wall parameters fitted to the BOPTEST `bestest_hydronic` step response. |
| `comp` | `aio/sim/plants_comp.py` | Independent PyTorch implementation of Gravdahl, Willems, de Jager & Egeland (2004; CDC 2000) centrifugal compressor surge equations. |
| `rsr` | `aio/sim/plants_rsr.py` | Independent PyTorch implementation of Larsson, Govatsmark, Skogestad & Yu (2003; after Wu & Yu 1996) reactor-separator-recycle equations with standard tray/drum holdup dynamics. |
| `amm` | `aio/sim/plants_amm.py` | Independent PyTorch implementation of Morud & Skogestad (1998) quench-cooled ammonia synthesis converter equations, Temkin-Pyzhev kinetics from Straus & Skogestad (2017), and bed void fraction from Araújo & Skogestad (2008). |
| `heater` | `aio/sim/plants_htr.py` | Independent PyTorch implementation of a two-pass crude heater built from the Homs 21 H-1 data sheet (Ibrahim & Al-Qassimi 2008, 2010), standard engineering correlations (Lobo & Evans, Sieder & Tate, API MPMS 11.1, ASTM D341, IEC 60534), and offline polynomial flue-gas species enthalpy fits generated with `thermo` 0.6.1. |

### Large Pool (`aio.sim.plants.LARGE`, 4 MIT-clean families)

| Key | Module | Provenance |
|---|---|---|
| `bsm1` | `aio/sim/plants_bsm.py` | **Independent PyTorch implementation of the publicly published IWA BSM1 equations** (Alex et al. 2008; Gernaey et al. 2014: five ASM1 reactors, 10-layer Takács settler, BSM2 temperature corrections). The Python model in `plants_bsm.py` contains no `bsm2-python` code and is MIT-clean. (Only the offline validation data files `data/bsm1/dryinfluent.csv` and `aio/sim/validate/data/bsm1_dry.npz` derive from `bsm2-python`; see Section 2.) |
| `btcol` | `aio/sim/plants.py` (`aio/sim/flow.py`, `aio/sim/props.json`) | Original flowsheet 20-stage benzene-toluene distillation column using offline local property fits (`aio/sim/props.json`) generated via `aio/sim/props_fit.py` from `thermo` (Peng-Robinson + ChemSep $k_{ij}$). |
| `meoh` | `aio/sim/plants.py` (`aio/sim/flow.py`, `aio/sim/props.json`) | Original flowsheet 18-stage methanol-water distillation column using offline local property fits (`aio/sim/props.json`) generated via `aio/sim/props_fit.py` from `thermo` (ChemSep NRTL, modified UNIFAC-Dortmund) and `CoolProp` (IAPWS-95 water, de Reuck & Craven methanol). |
| `debut` | `aio/sim/plants.py` (`aio/sim/flow.py`, `aio/sim/props.json`) | Original flowsheet 30-stage C3-C5 debutanizer using offline local property fits (`aio/sim/props.json`) generated via `aio/sim/props_fit.py` from `thermo` (Peng-Robinson + ChemSep $k_{ij}$). |

### Held-Out Pool (`aio.sim.plants.HELDOUT`, 5 MIT-clean families)

| Key | Module | Provenance |
|---|---|---|
| `boiler` | `aio/sim/plants_heldout.py` | Independent PyTorch implementation of Bell & Åström (1987; see also Pellegrinetti & Bentsman 1996) 160 MW drum boiler-turbine 3-state equations. |
| `wo` | `aio/sim/plants_heldout.py` | Independent PyTorch implementation of Williams & Otto (1960) reaction kinetics with simplified separation and recycle drum. |
| `mma` | `aio/sim/plants_heldout.py` | Independent PyTorch implementation of Maner, Doyle, Ogunnaike & Pearson (1996) MMA solution polymerization CSTR equations with energy balance. |
| `cryst` | `aio/sim/plants_heldout.py` | Independent PyTorch implementation of Randolph & Larson continuous MSMPR cooling crystallizer moment equations. |
| `boiler2` | `aio/sim/plants_heldout.py` | Independent PyTorch implementation of Åström & Bell (2000; Lund report TFRT-7577 Appendix B parameters) 4-state drum boiler equations with IAPWS-IF97 Region 4 and Wagner & Pruß (1993) saturation correlations. *(Offline validation file `aio/sim/validate/data/boiler_plant.json` holds open-loop curves digitized from Figs. 8-13 of Åström & Bell 2000 and is excluded from wheel distributions.)* |

*Note:* `aio/sim/lit.py` additionally encodes four published linear transfer-matrix distillation benchmarks (`wb` Wood & Berry 1973, `ory` Ogunnaike & Ray 1979, `ty` Tyreus 1979, `al` Alatiqi & Luyben 1986; cross-checked against Huang & Lin 2006, Vu & Lee 2010, and Edet & Katebi 2018) solely for prior-coverage diagnostics (`aio/sim/stats.py`).

---

## 2. Third-Party Licensed Industrial Benchmarks and Validation Data (Mandatory Attribution)

### 2.1 GasLib-24, morgen, and TRR154 (`aio/sim/plants_gaslib.py` & `aio/sim/validate/data/gaslib_morgen.npz`)

The held-out gas-network benchmark `gaslib` (`aio/sim/plants_gaslib.py`) and its offline validation reference trajectory (`aio/sim/validate/data/gaslib_morgen.npz`, excluded from wheel distributions) incorporate network topology, compressor station maps, discretization structure, and boundary flow data from three third-party open-source/open-data projects:

1. **GasLib-24** — **CC BY 3.0**
   - Source: `https://gaslib.zib.de/` (Pfetsch et al., 2015; compressor station characteristic coefficients `STATIONS` from `GasLib-24-v1-20211130.cs`).
   - License: Creative Commons Attribution 3.0 Unported (`CC BY 3.0`).
   - Modifications made in `aio/sim/plants_gaslib.py`: edge discretization at nominal length `NOM_LEN = 12000 m` for the `gaslib()` APC exam (and `NOM_MORGEN = 2400 m` for comparison), addition of valve loss formulation and APC MV/CV/limit definitions.
2. **morgen** (v1.2) — **BSD-2-Clause**
   - Source: `https://github.com/gramian/morgen` (Himpe, Grundel & Benner, 2021).
   - License: BSD 2-Clause License (`BSD-2-Clause`).
   - Scope: `NET` edge list adapted from `networks/GasLib24.net`, `ode_end` isothermal Euler discretization structure and Schifrinson/AGA88 parameterization in `aio/sim/plants_gaslib.py`, and Octave reference trajectories recorded into `aio/sim/validate/data/gaslib_morgen.npz` via `aio/sim/validate/gaslib.py`.
3. **TRR154 InstationaryGasNetworks** — **Apache-2.0**
   - Source: `https://github.com/TRR154/InstationaryGasNetworks` (`GasLib-24-no-resistor-sinus`).
   - License: Apache License 2.0 (`Apache-2.0`).
   - Scope: initial boundary conditions (`QD0`, `PIN0`) in `aio/sim/plants_gaslib.py` and transient 24-hour exit mass flows used to record `aio/sim/validate/data/gaslib_morgen.npz`.

### 2.2 bsm2-python (`data/bsm1/dryinfluent.csv` & `aio/sim/validate/data/bsm1_dry.npz` ONLY)

- **Scope strictly limited to offline validation data**:
  - `data/bsm1/dryinfluent.csv`: IWA BSM1 14-day dry-weather influent file copied from `bsm2-python` 0.0.16 (`src/bsm2_python/data/`).
  - `aio/sim/validate/data/bsm1_dry.npz`: open-loop reference trajectory computed with `bsm2-python` 0.0.16 via `python -m aio.sim.validate.bsm record`.
  - Both files are **excluded from wheel distributions** (`pyproject.toml`). As stated in Section 1, the simulation equations in `aio/sim/plants_bsm.py` are independently implemented from the published IWA BSM1 specification and contain no `bsm2-python` code.
- **Upstream source & license**: `bsm2-python` 0.0.16 (`https://github.com/FAU-EVT/bsm2-python`), **BSD 3-Clause License**.
- **Required notice (reproduced from `data/bsm1/README.md`)**:
  > Copyright (c) 2025, FAU-EVT. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: (1) redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer; (2) redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution; (3) neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

---

## 3. Industrial Code Transcriptions / Benchmarks Without Explicit OSI License

The following **4 modules (5 plant registry entries)** are derived from published industrial benchmarks whose upstream source archives, course packages, or compiled binaries carry **no explicit OSI-approved open-source license**.

In the upcoming standalone simulator extraction (**`M3rcyzzz/Hflash`** / **`apcgym`**), these modules are scheduled to be physically isolated into a dedicated **`contrib_suite`** separated from the MIT-clean core suite.

For every module below:
1. **No original third-party Fortran, C, MATLAB `.m`/`.mat`, or compiled S-function files are distributed in this repository.**
2. Offline numerical validation trajectories in `aio/sim/validate/data/*.npz` contain numbers only and are **excluded from wheel builds** (`pyproject.toml`).
3. **This project claims MIT copyright only over its own original PyTorch expression-graph wrapper/adapter code and does NOT claim original MIT copyright over upstream numerical constants, parameter tables, initial state vectors, or third-party benchmark data.** Users relying on these modules for redistribution or commercial use are responsible for verifying terms with the original authors.

| Registry Key(s) | Module & Validation Artifacts | Upstream Provenance & Disclosure |
|---|---|---|
| `tep`, `tep9` | `aio/sim/plants_tep.py`; offline validation artifacts `aio/sim/validate/data/tep_native.npz`, `aio/sim/validate/tep_native_build.py`, `aio/sim/validate/tep_native_adapter.c`, `aio/sim/validate/tep_native_record.py` | **Tennessee Eastman Process** (Downs & Vogel, *Comput. Chem. Eng.* 17(3), 1993; regulatory control layer after Ricker, *J. Process Control* 6(4), 1996). Equations, property constants, and initial state vector `YY` are transcribed from the `TEFUNC`/`TEINIT` routines (author archive `https://depts.washington.edu/control/LARRY/TE/temexd_mod.zip`, no explicit OSI license). Original Fortran/C sources are **not** distributed in this repository. |
| `vam` | `aio/sim/plants_vam.py`; offline validation artifacts `aio/sim/validate/data/vam_native.npz`, `aio/sim/validate/vam_native_adapter.c`, `aio/sim/validate/vam_native_record.py` | **Vinyl Acetate Monomer (VAM) Plant** (Luyben & Tyreus, *Comput. Chem. Eng.* 22, 1998; Chen, Dave, McAvoy & Luyben, *Ind. Eng. Chem. Res.* 42, 2003). Equations, constants, and base state `X0`/`U0` are transcribed routine-by-routine from the UMD C routines (`https://terpconnect.umd.edu/~mcavoy/VAC%20Material/`, `C routines` / `test_VAcPlant.m`; `aio/sim/validate/fetch.py` notes license: `none stated`). Original C/MATLAB files are **not** distributed in this repository. |
| `badgwell` | `aio/sim/plants_badgwell.py`; offline validation artifact `aio/sim/validate/data/badgwell_heater.npz` | **Badgwell's 5×5 Linear Fired Heater** (1998 MPC Workshop; Georgia Tech ChBE 4412 fired heater `mpcsim` package: `make_heater_cont.m`, `heater_mpcsim.m`; validation artifact recorded from `HeaterModelCont.mat` and `heaterData.mat`, no explicit OSI license stated). Original `.m`/`.mat` files are **not** distributed in this repository. |
| `refrig` | `aio/sim/plants_refrig.py`; offline validation artifact `aio/sim/validate/data/refrig_reference.npz` | **IFAC PID 2018 Refrigeration Benchmark** (Bejarano, Alfaya, Rodríguez, Ortega & Morilla, UNED, `http://www.dia.uned.es/~fmorilla/benchmarkPID2018`). Upstream distributes compiled S-functions only (no source code and no explicit OSI license). `plants_refrig.py` implements the simplified moving-boundary equations from the benchmark documentation, using R404A fits from `CoolProp` (v7) and component parameters identified from reference run `RSBenchmark_20170524_13_48`. `refrig_reference.npz` holds inputs/outputs from the two published reference runs (`RSBenchmark_20170524_13_48`, `RSBenchmark_20170528_22_34`). Compiled upstream binaries are **not** distributed in this repository. |

---

## 4. Offline Validation Toolchain and External Datasets (Not Bundled in Core Runtime)

The following tools and datasets are used exclusively offline for property-table fitting (`aio/sim/props_fit.py`) or independent validation (`aio/sim/validate/`). None are required core runtime dependencies, and none of the external validation datasets fetched by `aio/sim/validate/fetch.py` are redistributed in this repository:

- **`CoolProp`** (optional `[validate]` dependency; `CoolProp 7` noted in `aio/sim/plants_refrig.py`): used offline in `aio/sim/props_fit.py`, `aio/sim/validate/refprops.py`, `aio/sim/validate/columns.py`, and for R404A fits in `aio/sim/plants_refrig.py` (IAPWS-95, reference Helmholtz EOS, GERG-2008, Bell & Lemmon 2016). Not imported at simulator runtime; see upstream CoolProp package for license terms.
- **`thermo`** (optional `[validate]` dependency; `thermo 0.6.1` noted in `aio/sim/plants_htr.py`): used offline in `aio/sim/props_fit.py`, `aio/sim/validate/refprops.py`, `aio/sim/validate/columns.py`, and for flue-gas enthalpy fits in `aio/sim/plants_htr.py` (Peng-Robinson + ChemSep $k_{ij}$, ChemSep NRTL, modified UNIFAC-Dortmund `DOUFIP2016`/`DOUFSG`). Not imported at simulator runtime; see upstream `thermo` package for license terms.
- **`DaISy`** (KU Leuven, Database for the Identification of Systems): `https://ftp.esat.kuleuven.be/pub/SISTA/data/` (`process_industry/` and `thermic/`). Noted in `aio/sim/validate/fetch.py` as `free for research, cite DaISy`. Not distributed in this repository.
- **`BOPTEST`** (IBPSA Project 1): `https://github.com/ibpsa/project1-boptest` (tested against commit `a29024f`, `1.0.0-dev`, as recorded in `docs/validation.md`), licensed under **BSD-3-Clause**. Accessed via REST interface (`aio/sim/validate/boptest.py`, `aio/sim/validate/boptest_serve.py`); upstream FMUs and code are not distributed in this repository.
- **Additional external datasets downloaded on demand via `aio/sim/validate/fetch.py`** (not distributed in this repository):
  - Skogestad Column A MATLAB files (`https://skoge.folk.ntnu.no/book/matlab_m/cola/`, license: none stated);
  - Zenodo `CC BY 4.0` datasets: CSTH pilot plant (`10093059`, Thornhill, Patwardhan & Shah 2008), TUM continuous distillation (`17628963`), Batch distillation 119 runs (`20411905`, Arweiler et al.), PRONTO multiphase flow (`1341583`, Cranfield), Manipal pilot column (`22763516`);
  - Cascaded tanks benchmark (`https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks`, 4TU, `CC BY 4.0`, Schoukens, Mattsson, Wigren & Noël).
