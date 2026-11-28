"""AIO: one pretrained network that controls APC-scale process units from tag configuration alone.

aio.sim    the simulation core: plant and scenario prior, batched differentiable plant, references
aio.train  the learner on that core (long windows, imitation of a realizable teacher, identification
           readout, simulator gradients)
"""
