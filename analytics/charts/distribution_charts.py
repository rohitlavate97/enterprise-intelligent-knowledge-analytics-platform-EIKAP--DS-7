"""Distribution charting module using Matplotlib and Plotly."""
from typing import Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from shared.logging import get_logger

logger = get_logger(__name__)


class DistributionChart:
    """Generates distribution charts using Plotly or Matplotlib."""

    def plot_histogram(self, df: pd.DataFrame, column: str, bins: int = 30, hue: Optional[str] = None, engine: str = "plotly") -> Any:
        """Plots histogram."""
        if df.empty or column not in df.columns:
            logger.warning("Invalid data for plot_histogram")
            return None

        if engine == "plotly":
            fig = px.histogram(df, x=column, color=hue, nbins=bins, title=f"Histogram of {column}")
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue and hue in df.columns:
                categories = df[hue].unique()
                for cat in categories:
                    subset = df[df[hue] == cat][column].dropna()
                    ax.hist(subset, bins=bins, alpha=0.5, label=str(cat))
                ax.legend(title=hue)
            else:
                data = df[column].dropna()
                ax.hist(data, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram of {column}")
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def plot_boxplot(self, df: pd.DataFrame, y_col: str, x_col: Optional[str] = None, engine: str = "plotly") -> Any:
        """Plots boxplot."""
        if df.empty or y_col not in df.columns:
            logger.warning("Invalid data for plot_boxplot")
            return None

        if engine == "plotly":
            fig = px.box(df, y=y_col, x=x_col, title=f"Boxplot of {y_col}")
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))
            if x_col and x_col in df.columns:
                groups = df.groupby(x_col)[y_col].apply(lambda g: g.dropna().values)
                ax.boxplot(groups.values, labels=groups.index)
                ax.set_xlabel(x_col)
            else:
                data = df[y_col].dropna().values
                ax.boxplot(data, labels=[y_col])
            ax.set_ylabel(y_col)
            ax.set_title(f"Boxplot of {y_col}")
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def plot_violin(self, df: pd.DataFrame, y_col: str, x_col: Optional[str] = None, engine: str = "plotly") -> Any:
        """Plots violin plot."""
        if df.empty or y_col not in df.columns:
            logger.warning("Invalid data for plot_violin")
            return None

        if engine == "plotly":
            fig = px.violin(df, y=y_col, x=x_col, box=True, title=f"Violin Plot of {y_col}")
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))
            if x_col and x_col in df.columns:
                groups = [g.dropna().values for _, g in df.groupby(x_col)[y_col]]
                labels = list(df.groupby(x_col).groups.keys())
                ax.violinplot(groups, showmeans=True, showmedians=True)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels)
                ax.set_xlabel(x_col)
            else:
                data = df[y_col].dropna().values
                ax.violinplot(data, showmeans=True, showmedians=True)
            ax.set_ylabel(y_col)
            ax.set_title(f"Violin Plot of {y_col}")
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def plot_kde(self, df: pd.DataFrame, column: str, engine: str = "matplotlib") -> Any:
        """Plots KDE plot."""
        if df.empty or column not in df.columns:
            logger.warning("Invalid data for plot_kde")
            return None

        data = df[column].dropna().values
        if len(data) < 2:
            return None

        if engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))
            kde = stats.gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_vals, kde(x_vals), color="steelblue", lw=2)
            ax.fill_between(x_vals, kde(x_vals), alpha=0.3, color="skyblue")
            ax.set_xlabel(column)
            ax.set_ylabel("Density")
            ax.set_title(f"KDE Plot of {column}")
            fig.tight_layout()
            return fig
        elif engine == "plotly":
            import plotly.figure_factory as ff
            fig = ff.create_distplot([data], [column], show_hist=False)
            fig.update_layout(title_text=f"KDE Plot of {column}")
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")
