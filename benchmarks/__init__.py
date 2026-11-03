# Public benchmark registry: import available models and map CLI names to classes.
from .Dubins3D import Dubins3D, Dubins3D_scLTL
from .Dubins2D_scLTL import Dubins2D_scLTL
from .Pendulum import Pendulum
from .MountainCar import MountainCar
from .Integrators import DoubleIntegrator, DoubleIntegrator_scLTL
from .Test1D import Test1D
from .Temperature1D_reach import Temperature1D_reach
from .Temperature1D_safety import Temperature1D_safety
from .Temperature4D import Temperature4D
from .Temperature3D import Temperature3D
from .Temperature2D import Temperature2D


def create_model(args):
    from core.abstraction.model import parse_model

    model_map = {
        'Dubins2D_scLTL': Dubins2D_scLTL,
        'Dubins3D': Dubins3D,
        'Dubins3D_scLTL': Dubins3D_scLTL,
        'Pendulum': Pendulum,
        'MountainCar': MountainCar,
        'DoubleIntegrator': DoubleIntegrator,
        'DoubleIntegrator_scLTL': DoubleIntegrator_scLTL,
        'Test1D': Test1D,
        'Temperature1D_reach': Temperature1D_reach,
        'Temperature1D_safety': Temperature1D_safety,
        'Temperature2D': Temperature2D,
        'Temperature3D': Temperature3D,
        'Temperature4D': Temperature4D,
    }

    model_cls = model_map.get(args.model)
    if model_cls is None:
        raise ValueError(f"The passed model '{args.model}' could not be found")

    return parse_model(model_cls(args), args.floatprecision)
