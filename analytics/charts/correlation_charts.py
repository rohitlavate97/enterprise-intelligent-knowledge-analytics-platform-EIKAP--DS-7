"""Correlation charting module using Matplotlib and Plotly."""
from typing import Optional, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from shared.logging import get_logger

logger = get_logger(__name__)


class CorrelationChart:
    """Generates correlation charts using Plotly or Matplotlib."""

    def plot_heatmap(self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = "pearson", engine: str = "plotly") -> Any:
        """Plots correlation heatmap."""
        if df.empty:
            logger.warning("Empty dataframe for plot_heatmap")
            return None

        cols_to_use = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols_to_use:
            logger.warning("No numeric columns available for heatmap")
            return None

        corr = df[cols_to_use].corr(method=method)

        if engine == "plotly":
            fig = px.imshow(corr, text_auto=True, aspect="auto", title=f"Correlation Heatmap ({method})")
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax)
            ax.set_xticks(range(len(cols_to_use)))
            ax.set_yticks(range(len(cols_to_use)))
            ax.set_xticklabels(cols_to_use, rotation=45, ha="left")
            ax.set_yticklabels(cols_to_use)
            for (i, j), val in np.ndenumerate(corr.values):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.7 else "white")
            ax.set_title(f"Correlation Heatmap ({method})", pad=20)
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def plot_scatter_matrix(self, df: pd.DataFrame, columns: List[str], color_col: Optional[str] = None, engine: str = "plotly") -> Any:
        """Plots scatter matrix."""
        if df.empty or not columns:
            logger.warning("Invalid data for plot_scatter_matrix")
            return None

        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return None

        if engine == "plotly":
            fig = px.scatter_matrix(df, dimensions=valid_cols, color=color_col, title="Scatter Matrix")
            return fig
        elif engine == "matplotlib":
            n = len(valid_cols)
            fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
            for i, col1 in enumerate(valid_cols):
                for j, col2 in enumerate(valid_cols):
                    ax = axes[i, j] if n > 1 else axes
                    if i == j:
                        ax.hist(df[col1].dropna(), bins=15, color="skyblue", edgecolor="black")
                    else:
                        ax.scatter(df[col2], df[col1], alpha=0.5, s=15)
                    if i == n - 1:
                        ax.set_xlabel(col2)
                    if j == 0:
                        ax.set_ylabel(col1)
            fig.suptitle("Scatter Matrix", y=1.02)
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def plot_pair_correlation(self, df: pd.DataFrame, col1: str, col2: str, trendline: bool = True, engine: str = "plotly") -> Any:
        """Plots pair correlation."""
        if df.empty or col1 not in df.columns or col2 not in df.columns:
            logger.warning("Invalid data for plot_pair_correlation")
            return None

        if engine == "plotly":
            trend_str = "ols" if trendline else None
            fig = px.scatter(df, x=col1, y=col2, trendline=trend_str, title=f"Correlation: {col1} vs {col2}")
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df[col1], df[col2], alpha=0.6, color="steelblue")
            if trendline:
                clean_df = df[[col1, col2]].dropna()
                if len(clean_df) > 1:
                    m, b = np.polyfit(clean_df[col1], clean_df[col2], 1)
                    x_line = np.linspace(clean_df[col1].min(), clean_df[col1].max(), 100)
                    ax.plot(x_line, m * x_line + b, color="red", linestyle="--", label="Trendline")
                    ax.legend()
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f"Correlation: {col1} vs {col2}")
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")
