"""eagmodel: the Evidence-Aggregation-Growth model of competence."""
from .data import EAGData
from .estimate import fit, EAGFit
from .functions import (aggregate, competence_trajectory, kappa_from_rho, leverage, mastery,
                        rho_from_kappa, time_to_readiness)
from .simulate import SimConfig, simulate

__version__ = "0.1.0"
__all__ = ["EAGData", "EAGFit", "SimConfig", "aggregate", "competence_trajectory", "fit", "kappa_from_rho",
           "leverage", "mastery", "rho_from_kappa", "simulate", "time_to_readiness"]
