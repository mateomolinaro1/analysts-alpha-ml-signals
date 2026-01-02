from typing import Any, Dict
from sklearn.linear_model import ElasticNet
from .model_wrapper import ModelWrapper


class ElasticNetWrapper(ModelWrapper):
    """
    Implementation of ABC ModelWrapper for elastic net regressor from scikit-learn.
    """

    def create_model(self, hyper_params: Dict[str, Any]) -> None:
        """
        Create the ElasticNet model.

        Parameters
        ----------
        hyper_params : Dict[str, Any]
            Hyper-parameters for the sklearn ElasticNet.
            Example : 
            {
                "alpha": 1.0,
                "l1_ratio": 0.5
            }
        """
        self.model = ElasticNet(**hyper_params)
