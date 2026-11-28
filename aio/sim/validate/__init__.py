"""Validation of the simulation core against sources outside this repository (docs/validation.md).

  tools      a plant of the engine as numpy functions; a converged reference of the same equations by an
             independent integrator (LSODA); a tap on what the shell feeds a plant during a rollout
  papers     the plants written again from the papers, and the numbers the papers publish
  checks     per plant: published steady states and dynamics, equations against the second transcription,
             conservation
  numerics   the integrator as it is run, on the inputs of training episodes, against a converged reference
  colamod    Skogestad's column A: the original MATLAB code run in Octave against the engine's column
  realdata   records of real plants: measured valve curves and noise against the field layer, a real
             two-tank rig against the engine's level equations, step-response features of identified
             models among the prior's plants
  tep        Tennessee Eastman (plants_tep) against trajectories of the native code, recorded on the machine
             that has the build by tep_native_record (standalone)
  bsm        BSM1 (plants_bsm) against the benchmark report and bsm2-python, by the benchmark's protocol
  boptest    BOPTEST's building emulators through its REST interface: its reference KPIs reproduced, step
             tests of zone loops among the prior's plants, the network zero-shot in place of the test
             cases' own zone loops under BOPTEST's scoring; boptest_serve (standalone) puts the interface
             in front of BOPTEST's own TestCase inside its worker image
  fetch      where the third-party code and data come from, and a downloader
  run        all of it, as reports/validation/*.json and the table of docs/validation.md

Nothing here is imported by the simulator or the trainer. Third-party code and data are not in the
repository: `python -m aio.sim.validate.fetch` lists where each comes from and downloads it. `data/` holds
reference trajectories computed for tep and bsm (data/README.md).
"""
