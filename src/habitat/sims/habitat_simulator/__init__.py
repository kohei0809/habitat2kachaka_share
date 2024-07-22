from habitat.core.registry import registry
from habitat.sims.habitat_simulator.real_world import RealWorld
from habitat.core.simulator import Simulator

def _try_register_real_world():
    try:
        import habitat_sim

        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        habitat_sim_import_error = e

    if has_habitat_sim:
        from habitat.sims.habitat_simulator.real_world import RealWorld
    else:
        @registry.register_simulator(name="Real-v0")
        class HabitatSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error
