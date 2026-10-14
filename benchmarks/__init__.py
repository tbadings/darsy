# Load the benchmarks from the subfiles
from .Dubins3D import Dubins3D
from .Dubins2D_scLTL import Dubins2D_scLTL
from .Dubins4D import Dubins4D
from .Drone4D import Drone4D
from .Drone6D import Drone6D, Drone6D_small
from .Pendulum import Pendulum
from .MountainCar import MountainCar
from .CartPole import CartPole
from .Integrators import DoubleIntegrator, DoubleIntegrator_scLTL
from .Test1D import Test1D


def create_model(args):
	from core.abstraction.model import parse_model

	model_map = {
		'Dubins2D_scLTL': Dubins2D_scLTL,
		'Dubins3D': Dubins3D,
		'Dubins4D': Dubins4D,
		'Drone4D': Drone4D,
		'Drone6D': Drone6D,
		'Drone6D_small': Drone6D_small,
		'Pendulum': Pendulum,
		'MountainCar': MountainCar,
		'CartPole': CartPole,
		'DoubleIntegrator': DoubleIntegrator,
		'DoubleIntegrator_scLTL': DoubleIntegrator_scLTL,
		'Test1D': Test1D,
	}

	model_cls = model_map.get(args.model)
	if model_cls is None:
		raise ValueError(f"The passed model '{args.model}' could not be found")

	return parse_model(model_cls(args))