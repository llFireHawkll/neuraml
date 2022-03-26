import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from neuraml.exceptions.exceptions import ModelNotFittedError

import shap


class ClsDoShapAnalysis:
    def __init__(self, model) -> None:
        self.model = model

    def _xgboost_shap_analysis(
        self,
        dataframe: pd.DataFrame,
        save_shap_artifacts: bool = True,
        show_shap_plots: bool = True,
        **kwargs
    ):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
            save_shap_artifacts (bool, optional): _description_. Defaults to True.
            show_shap_plots (bool, optional): _description_. Defaults to True.

        Raises:
            NotImplementedError: _description_
        """
        # Step-1 Setup the SHAP Tree Explainer based on the model type
        # as model_type here == xgboost
        self.shap_explainer = shap.TreeExplainer(model=self.model)

        # Step-2 Generate SHAP Values for the provided dataframe
        self.shap_values = self.shap_explainer.shap_values(X=xgb.DMatrix(dataframe))

        # Step-3 Generate SHAP Tornado Plot and Summary Bar Plot
        if show_shap_plots:
            plt.figure(figsize=(15, 7))
            shap.summary_plot(
                shap_values=self.shap_values,
                features=dataframe,
                max_display=30,
                **kwargs
            )

            # Step-4 Save SHAP artifacts
            if save_shap_artifacts:
                plt.savefig(
                    "shap_tornado_plot.png", format="png", dpi=1000, bbox_inches="tight"
                )
            else:
                raise NotImplementedError

            plt.figure(figsize=(15, 7))
            shap.summary_plot(
                shap_values=self.shap_values,
                features=dataframe,
                max_display=30,
                plot_type="bar",
                **kwargs
            )

            # Step-4 Save SHAP artifacts
            if save_shap_artifacts:
                plt.savefig(
                    "shap_summary_plot.png", format="png", dpi=1000, bbox_inches="tight"
                )
            else:
                raise NotImplementedError

    def do_shap_analysis(
        self,
        dataframe: pd.DataFrame,
        save_shap_artifacts: bool = True,
        show_shap_plots: bool = True,
        **kwargs
    ):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
            save_shap_artifacts (bool, optional): _description_. Defaults to True.
            show_shap_plots (bool, optional): _description_. Defaults to True.

        Raises:
            ModelNotFittedError: _description_
            NotImplementedError: _description_
        """
        if self.model.model_type == "xgboost":
            if self.model._fit_flag:
                self._xgboost_shap_analysis(
                    dataframe=dataframe,
                    save_shap_artifact=save_shap_artifacts,
                    show_shap_plots=show_shap_plots,
                    **kwargs
                )
            else:
                raise ModelNotFittedError()
        else:
            raise NotImplementedError
