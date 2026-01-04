from enum import Enum
from ..model_wrappers import *

class ModelType(Enum):
    RANDOM_FOREST = RandomForestWrapper
    ELASTIC_NET = ElasticNetWrapper