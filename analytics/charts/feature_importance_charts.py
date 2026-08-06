"""
Feature importance charting module.
"""
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from shared.logging import get_logger

logger = get_logger(__name__)

class FeatureImportanceChart:
    """Generates feature importance charts using Plotly or Matplotlib."""
    
    def plot_feature_importance(self, feature_names: Union[List[str], Dict[str, float]], importance_scores: Optional[List[float]] = None, top_n: int = 15, title: str = "Feature Importance", engine: str = "plotly") -> Any:
        """Plots horizontal bar chart for feature importance."""
        if isinstance(feature_names, dict):
            names = list(feature_names.keys())
            scores = list(feature_names.values())
        else:
            names = feature_names
            scores = importance_scores or []

        if not names or not scores or len(names) != len(scores):
            logger.warning("Invalid data for plot_feature_importance")
            return None
            
        df = pd.DataFrame({
            "Feature": names,
            "Importance": scores
        })
        df = df.sort_values(by="Importance", ascending=True).tail(top_n)
        
        if engine == "plotly":
            fig = px.bar(df, x="Importance", y="Feature", orientation='h', title=title)
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(df["Feature"], df["Importance"], color="skyblue")
            ax.set_xlabel("Importance Score")
            ax.set_title(title)
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def plot_importance(self, feature_importances: Dict[str, float], top_n: int = 15, title: str = "Feature Importance", engine: str = "plotly") -> Any:
        """Alias for plot_feature_importance accepting a dictionary."""
        return self.plot_feature_importance(feature_importances, top_n=top_n, title=title, engine=engine)
            
    def plot_shap_summary(self, feature_names: List[str], shap_values: List[List[float]], feature_data: pd.DataFrame, engine: str = "plotly") -> Any:
        """Plots SHAP summary chart simulation."""
        if not feature_names or not shap_values or feature_data.empty:
            logger.warning("Invalid data for plot_shap_summary")
            return None
            
        if engine == "plotly":
            fig = go.Figure()
            for i, feature in enumerate(feature_names):
                if i < len(shap_values[0]):
                    vals = [row[i] for row in shap_values]
                    feature_vals = feature_data[feature].tolist() if feature in feature_data.columns else []
                    
                    fig.add_trace(go.Scatter(
                        x=vals,
                        y=[feature] * len(vals),
                        mode='markers',
                        marker=dict(
                            color=feature_vals if feature_vals else 'blue',
                            colorscale='Viridis',
                            showscale=True if i == 0 else False,
                            colorbar=dict(title="Feature value") if i == 0 else None,
                            opacity=0.5
                        ),
                        name=feature
                    ))
            fig.update_layout(title="SHAP Summary Plot", xaxis_title="SHAP value (impact on model output)")
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 8))
            for i, feature in enumerate(feature_names):
                if i < len(shap_values[0]):
                    vals = [row[i] for row in shap_values]
                    feature_vals = feature_data[feature].tolist() if feature in feature_data.columns else [0]*len(vals)
                    scatter = ax.scatter(vals, [feature]*len(vals), c=feature_vals, cmap='viridis', alpha=0.5)
                    
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title("SHAP Summary Plot")
            fig.colorbar(scatter, ax=ax, label="Feature value")
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")
