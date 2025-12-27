from abc import ABC, abstractmethod
from skops.io import dump, load
from sklearn.base import BaseEstimator as SklearnModel
import json
import numpy as np
import pandas as pd
import polars as pl
from typing import Union

MatrixLike = Union[pd.DataFrame, pl.DataFrame, np.ndarray]
VectorLike = Union[pd.Series, pl.Series, np.ndarray]


class ModelWrapper(ABC):
    """
    Abstract class reprensenting a machine learning model wrapper.

    This abstract class defines main methods that any heriting model wrapper 
    will use. For the most of them, overriding is not necessary (but possible 
    for specific use-cases).

    Model specific implementations should mainly focus on the `create_model` method
    where the instance of the underlying model should be created and configured.

    Then, for fitting, predicting and model persistence methods (save/load), the
    abstract implementations should be sufficient as it will refer to generic
    attributes of the class.

    Example usage:

    ```python
    class RidgeWrapper(ModelWrapper):
        def create_model(self):
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
    
    ridge_model = RidgeWrapper()
    ridge_model.create_model()

    ridge_model.fit(X_train, y_train)
    ridge_model.save_model("ridge_model.skops", "ridge_manifest.json")

    # Later usage
    ridge_loaded = RidgeWrapper()
    ridge_loaded.load_model("ridge_model.skops", "ridge_manifest.json")
    ridge_loaded.predict(X_test)
    ```
    """
    def __init__(self):
        self.model: SklearnModel = None
        self.manifest: dict = {}
        self.is_fitted: bool = False
        self.nb_features: int = None
        self.feature_names: list = None

    # ----------------------------------------------------------
    # |              Abstract methods to override              |
    # ----------------------------------------------------------

    @abstractmethod
    def create_model(self) -> None:
        """
        Model specific implementation to create and configure the underlying model.
        Need to be overriden by heriting classes in any case.

        This method sould set an attribute `self.model` that will be used by the
        generic methods of the class as fitting, predicting and model persistence.
        """
        raise NotImplementedError("create_model method must be implemented by the subclass.")

    # ----------------------------------------------------------
    # |           Generic sklearn-like fit & predict           |
    # ----------------------------------------------------------

    def fit(self, X: MatrixLike, y: VectorLike) -> None:
        """
        Robust and modular fitting of the underlying scikit-learn model.
        To override if necessary for specific use-cases.

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame | np.ndarray
            Input features for training.
        y : pd.Series | pl.Series | np.ndarray
            Target variable for training.
        """
        self._ensure_model()
        X_arr = self._coerce_X(X, for_fit=True)
        y_arr = self._coerce_y(y)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        self.nb_features = X_arr.shape[1] if X_arr.ndim == 2 else None

        try:
            params = self.model.get_params(deep=False)
        except Exception:
            params = None

        self.manifest.update({
            "model_class": self.model.__class__.__name__,
            "model_module": self.model.__class__.__module__,
            "n_features_in": self.nb_features,
            "feature_names": self.feature_names,
            "params": params,
        })

    def predict(self, X: MatrixLike) -> np.ndarray:
        """
        Robust and modular prediction of the underlying scikit-learn model.
        To override if necessary for specific use-cases.

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame | np.ndarray
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predictions from the model.
        """
        self._check_fitted()
        X_arr = self._coerce_X(X, for_fit=False)

        return self.model.predict(X_arr)
    
    # ----------------------------------------------------------
    # |                    Internal helpers                    |
    # ----------------------------------------------------------

    def _ensure_model(self):
        """
        Ensure that the model has been created.
        """
        if self.model is None:
            raise RuntimeError("Call create_model() before fit/predict.")

    def _check_fitted(self):
        """
        Ensure that the model has been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _coerce_X(self, X: MatrixLike, for_fit: bool) -> np.ndarray:
        """
        Modular ingestion of input features X.
        Accepts pandas.DataFrame / polars.DataFrame / numpy arrays.

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame | np.ndarray
            Input features.
        for_fit : bool
            Whether the coercion is for fitting (True) or predicting (False).

        Returns
        -------
        np.ndarray
            Coerced input features as a numpy array.
        """
        # Universal ingestion of X
        names = None
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
            arr = X.values
        elif isinstance(X, pl.DataFrame):
            names = X.columns
            arr = X.to_numpy()
        else:
            arr = np.asarray(X)

        if for_fit and names is not None:
            self.feature_names = list(names)

        # Ensuring columns and order for prediction
        if (not for_fit) and (self.feature_names is not None) and names is not None:

            missing = set(self.feature_names) - set(names)
            extra = set(names) - set(self.feature_names)
            if missing:
                raise ValueError(f"Missing columns at predict: {sorted(missing)}")
            if extra:
                raise ValueError(f"Extra columns at predict: {sorted(extra)}")
            
            if isinstance(X, pd.DataFrame):
                arr = X[self.feature_names].values
            elif isinstance(X, pl.DataFrame):
                arr = X.select(self.feature_names).to_numpy()

        return arr

    def _coerce_y(self, y: VectorLike) -> np.ndarray:
        """
        Modular ingestion of input features X.
        Accepts pandas.DataFrame / polars.DataFrame / numpy arrays.

        Parameters
        ----------
        y : pd.Series | pl.Series | np.ndarray
            Target variable for training.

        Returns
        -------
        np.ndarray
            Coerced target variable as a numpy array.
        """
        if isinstance(y, pd.Series) or isinstance(y, pl.Series):
            y = y.to_numpy().ravel()
        return np.asarray(y).ravel()
    
    # ----------------------------------------------------------
    # |                    Model persistence                   |
    # ----------------------------------------------------------

    def save_model(self, model_path: str = None, manifest_path: str = None) -> None:
        """
        Save the model to disk.
        The wanted format is :
        - 1 model file : .skops serialized for scikit-learn models
        - 1 manifest file : .json file containing metadata about the model

        Parameters
        ----------
        model_path: str
            Path to save the model file (e.g., .skops, .pt)
        manifest_path: str
            Path to save the manifest file (.json)
        """
        if model_path is None:
            model_path = self.__class__.__name__ + ".skops"
        if manifest_path is None:
            manifest_path = self.__class__.__name__ + "_manifest.json"

        dump(self.model, "saved_models/" + model_path)
        with open("saved_models/" + manifest_path, 'w') as f:
            json.dump(self.manifest, f)

    def load_model(self, model_path: str = None, manifest_path: str = None) -> None:
        """
        Load the model from disk.
        The wanted format is :
        - 1 model file : .skops serialized for scikit-learn models
        - 1 manifest file : .json file containing metadata about the model

        Parameters
        ----------
        model_path: str
            Path to load the model file (e.g., .skops, .pt)
        manifest_path: str
            Path to load the manifest file (.json)
        """
        if model_path is None:
            model_path = self.__class__.__name__ + ".skops"
        if manifest_path is None:
            manifest_path = self.__class__.__name__ + "_manifest.json"
            
        self.model = load("saved_models/" + model_path)
        with open("saved_models/" + manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.feature_names = self.manifest.get("feature_names")
        self.nb_features = self.manifest.get("nb_features")
        self.is_fitted = True
