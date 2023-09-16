# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_estimators.ipynb.

# %% ../00_estimators.ipynb 2
#| include: false
#| echo: false
from __future__ import annotations # allows multiple typing of arguments in Python versions prior to 3.10

# %% auto 0
__all__ = ['FindCluster', 'MachineControl']

# %% ../00_estimators.ipynb 9
#| include: false
import numpy as np
from .model_documentation import ggdModelDocumentation, ModelCard
from .utils import read_attr
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AffinityPropagation
from sklearn.utils.metaestimators import available_if

# %% ../00_estimators.ipynb 10
#| include: false
class FindCluster(BaseEstimator):
    "Retain only the columns of `X` that are in the same cluster as `y`."

    def __init__(
        self,
        cluster_alg:[BaseEstimator,ClusterMixin]=AffinityPropagation(), # An instance of the clustering algorithm to use
        auto_document:ggdModelDocumentation=ModelCard, # gingado Documenter template to facilitate model documentation
        random_state:int|None=None, # The random seed to be used by the algorithm, if relevant
    ):
        self.cluster_alg = cluster_alg
        self.auto_document = auto_document
        self.random_state = random_state
        if hasattr(self.cluster_alg, "random_state"):
            self.cluster_alg.set_params(random_state=self.random_state)

    def document(
        self, 
        documenter:ggdModelDocumentation|None=None # A gingado Documenter or the documenter set in `auto_document` if None.
    ):
        "Document the `FindCluster` model using the template in `documenter`"
        documenter = self.auto_document if documenter is None else documenter
        self.model_documentation = documenter()
        model_info = list(read_attr(self.cluster_alg))
        model_info = {k:v for i in model_info for k, v in i.items()}
        #self.model_documentation.read_model(self.benchmark)
        self.model_documentation.fill_model_info(model_info)

    def fit(
        self,
        X, # The population of entities, organised in columns
        y # The entity of interest
    ):
        "Fit `FindCluster`"
        temp_y_colname = "gingado_ycol"

        X[temp_y_colname] = y

        entities = X.columns
        y_mask = entities == temp_y_colname

        self.cluster_alg.fit(X.T)

        cluster = entities[self.cluster_alg.labels_ == self.cluster_alg.labels_[y_mask]]
        self.same_cluster_ = [e for e in cluster if e != temp_y_colname]
        
        self.document()
        return self

    def transform(
        self,
        X # The population of entities, organised in columns
    )->np.array: # Columns of `X` that are in the same cluster as `y`
        "Keep only the entities in `X` that belong to the same cluster as `y`"
        return X[self.same_cluster_]

    def fit_transform(
        self,
        X, # The population of entities, organised in columns
        y # The entity of interest
    )->np.array: # Columns of `X` that are in the same cluster as `y`
        "Fit a `FindCluster` object and keep only the entities in `X` that belong to the same cluster as `y`"
        self.fit(X, y)
        return self.transform(X)

# %% ../00_estimators.ipynb 38
#| include: false
import pandas as pd

from .benchmark import ggdBenchmark, RegressionBenchmark
from .model_documentation import ggdModelDocumentation, ModelCard
from sklearn.base import check_is_fitted, clone
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform

# %% ../00_estimators.ipynb 40
#| include: false
class MachineControl(BaseEstimator):
    "Synthetic controls with machine learning methods"

    def __init__(
        self,
        # An instance of the clustering algorithm to use, or None to retain all entities
        cluster_alg:[BaseEstimator,ClusterMixin]|None=AffinityPropagation(),
        # Method to weight the control entities
        estimator:BaseEstimator=RegressionBenchmark(), 
        # Algorithm for manifold learning
        manifold:BaseEstimator=TSNE(),
        # Include placebo estimations during prediction?
        with_placebo:bool=True,
        # gingado Documenter template to facilitate model documentation
        auto_document:ggdModelDocumentation=ModelCard,
        # The random seed to be used by the algorithm, if relevant
        random_state:int|None=None
    ):
        self.cluster_alg = cluster_alg
        self.estimator = estimator
        self.manifold = manifold
        self.with_placebo = with_placebo
        self.auto_document = auto_document
        self.random_state = random_state

        if hasattr(self.cluster_alg, "random_state"):
            self.cluster_alg.set_params(random_state=self.random_state)
        if hasattr(self.estimator, "random_state"):
            self.estimator.set_params(random_state=self.random_state)
        if hasattr(self.manifold, "random_state"):
            self.manifold.set_params(random_state=self.random_state)    

        self.pipeline = Pipeline([
            ('donor_pool', self.cluster_alg),
            ('estimator', self.estimator)
        ])

    def document(
        self, 
        documenter:ggdModelDocumentation|None=None # A gingado Documenter or the documenter set in `auto_document` if None.
    ):
        "Document the `MachineControl` model using the template in `documenter`"
        documenter = self.auto_document if documenter is None else documenter
        self.model_documentation = documenter()
        model_info = {}
        model_info_cluster = list(read_attr(self.cluster_alg))
        model_info['cluster'] = {k:v for i in model_info_cluster for k, v in i.items()}
        model_info_estimator = list(read_attr(self.estimator))
        model_info['estimator'] = {k:v for i in model_info_estimator for k, v in i.items()}
        model_info_manifold = list(read_attr(self.manifold))
        model_info['manifold'] = {k:v for i in model_info_manifold for k, v in i.items()}
        #self.model_documentation.read_model(self.benchmark)
        self.model_documentation.fill_model_info(model_info)

    def _create_placebo_df(
        self,
        X:pd.DataFrame, # A pandas DataFrame with data of shape (n_samples, n_control_entites)
        y:pd.DataFrame|pd.Series, # A pandas DataFrame or Series with data of shape (n_samples,)
        entity:str # A singleton column name for an entity in the control group
    ):
        X_placebo = pd.concat([X[self.donor_pool_], y], axis=1)
        y_placebo = X_placebo.pop(entity)
        return X_placebo, y_placebo

    def _fit_placebo_models(
        self, 
        X:pd.DataFrame, # A pandas DataFrame with pre-intervention data of shape (n_samples, n_control_entites)
        y:pd.DataFrame|pd.Series # A pandas DataFrame or Series with pre-intervention data of shape (n_samples,)
    ):
        self.placebo_models_ = {}
        self.placebo_score_pre_ = {}
        for entity in self.donor_pool_:
            X_pl, y_pl = self._create_placebo_df(X, y, entity)
            self.placebo_models_[entity] = clone(self.estimator)
            self.placebo_models_[entity].fit(X_pl, y_pl)
            self.placebo_score_pre_[entity] = mean_squared_error(
                y_true=y_pl, 
                y_pred=self.placebo_models_[entity].predict(X=X_pl),
                squared=False
            )

    def _select_controls(
        self,
        X:pd.DataFrame, # A pandas DataFrame with pre-intervention data of shape (n_samples, n_control_entites)
        y:pd.DataFrame|pd.Series # A pandas DataFrame or Series with pre-intervention data of shape (n_samples,)
    ): # 
        "Identifies which columns of `X` should be used as controls"
        if self.cluster_alg is None:
            self.donor_pool_ = X.columns
        else:
            Xy = pd.concat([X, y], axis=1)
            self.cluster_alg.fit(Xy.T)
            idx_y = Xy.columns == y.name
            self.donor_pool_ = [
                c for c in Xy.columns[self.cluster_alg.labels_ == self.cluster_alg.labels_[idx_y]]
                if c != y.name
            ]

    def get_controls(self):
        "Get the list of control entities"
        if hasattr(self, "donor_pool_"):
            return self.donor_pool_
        else:
            "Controls not selected yet"

    def _compare_controls(
        self,
        X:np.ndarray, # Array-like pre-intervention data of shape (n_samples, n_control_entites)
        y:np.ndarray # Array-like pre-intervention data of shape (n_samples,)
    )->[np.ndarray, np.ndarray]: # 2-d representation of the treated entity, controls, and the synthetic control
        "Calculates the 2-d manifold learning distribution and locates the distance between target and control in this distribution"
        df_manifold_learning = pd.concat([
            pd.DataFrame(X), 
            pd.DataFrame(self.machine_controls_),
            pd.DataFrame(y) # if actual data is last, it is easier to do the distance learning
        ], axis=1)
        self.manifold_embed_ = self.manifold.fit_transform(X=df_manifold_learning.T)
        self.distances_ = squareform(pdist(self.manifold_embed_))[-1,:-1] # last position in the resulting array is the dist between actual and synth control
        self.control_quality_test_ = np.percentile(self.distances_, self.distances_[-1])

    def fit(
        self,
        X:pd.DataFrame, # A pandas DataFrame with pre-intervention data of shape (n_samples, n_control_entites)
        y:pd.DataFrame|pd.Series # A pandas DataFrame or Series with pre-intervention data of shape (n_samples,)
    ):
        "Fit the `MachineControl` model"
        
        self.target_name_ = y.columns if hasattr(y, "columns") else y.name
        self._select_controls(X=X, y=y)
        
        if self.with_placebo:
            self._fit_placebo_models(X=X, y=y)

        X_donor_pool, y = self._validate_data(X[self.donor_pool_], y)
        
        self.estimator.fit(X=X_donor_pool, y=y)
        
        self.machine_controls_ = self.estimator.predict(X=X_donor_pool)

        # for the comparison part, note we use everyone, not just the selected control entities
        # this allows us to use a more robust test of whether there are many out-of-cluster entity
        # that would by itself be closer to the target entity.
        self._compare_controls(X=X.values, y=y)

        return self

    def predict(
        self,
        X:pd.DataFrame, # A pandas DataFrame with complete time series (pre- and post-intervention) of shape (n_samples, n_control_entites)
        y:pd.DataFrame|pd.Series # A pandas DataFrame or Series with complete time series of shape (n_samples,)
    ):
        "Calculate the model predictions before and after the intervention"
        check_is_fitted(self.estimator)

        X = X[self.donor_pool_]
        self.pred_ = self.estimator.predict(X=X)
        self.diff_ = y - self.pred_
        
        self.pred_ = pd.DataFrame(self.pred_, index=X.index)

        if self.with_placebo:
            self.placebo_predict_ = {}
            self.placebo_diff_ = {}
            self.placebo_score_all_ = {}
            for entity, model in self.placebo_models_.items():
                X_pl, y_pl = self._create_placebo_df(X=X, y=y, entity=entity)
                self.placebo_predict_[entity] = model.predict(X=X_pl)
                self.placebo_score_all_[entity] = mean_squared_error(
                    y_true=y_pl,
                    y_pred=self.placebo_predict_[entity],
                    squared=False
                )
                self.placebo_diff_[entity] = y_pl - self.placebo_predict_[entity]
            self.placebo_predict_ = pd.DataFrame(self.placebo_predict_, index=X.index)
            self.placebo_diff_ = pd.DataFrame(self.placebo_diff_, index=X.index)

        return self.pred_

    def intervention_effect(self):
        "Calculate the intervention effect after the cutoff date"
        pass
