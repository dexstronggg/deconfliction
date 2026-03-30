from .no_control import NoControl
from .fcfs import FCFS
from .vo_classic import VOClassic
from .vo_speed_only import VOSpeedOnly
from .rvo import RVO
from .orca_projected import ORCAProjected
from .csorca import CSORCA
from .milp import MILP
from .mappo import MAPPO
from .maddpg import MADDPG
from .masac import MASAC

ALL_ALGORITHMS = {
    "No-Control": NoControl,
    "FCFS": FCFS,
    "VO": VOClassic,
    "VO-Speed-Only (SSD)": VOSpeedOnly,
    "RVO": RVO,
    "ORCA": ORCAProjected,
    "CSORCA": CSORCA,
    "MILP": MILP,
    "MAPPO (untrained)": MAPPO,
    "MADDPG (untrained)": MADDPG,
    "MASAC (untrained)": MASAC,
}
