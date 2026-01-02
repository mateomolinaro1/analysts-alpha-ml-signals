from typing import Any, Dict
from sklearn.ensemble import RandomForestRegressor
from .model_wrapper import ModelWrapper


class RandomForestWrapper(ModelWrapper):
    """
    Implementation of ABC ModelWrapper for random forest regressor from scikit-learn.
    """
    def create_model(self, hyper_params: Dict[str, Any]) -> None:
        """
        Create the RandomForestRegressor model.

        Parameters
        ----------
        hyper_params : Dict[str, Any]
            Hyper-parameters for the sklearn RandomForestRegressor.
            Example : 
            {
                "n_estimators": 100,
                "max_depth": 10
            }
        """
        self.model = RandomForestRegressor(**hyper_params)
