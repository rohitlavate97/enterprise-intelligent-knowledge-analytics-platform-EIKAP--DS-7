"""
Trend charting module.
"""
from typing import List, Any, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared.logging import get_logger

logger = get_logger(__name__)

class TrendChart:
    """Generates trend analysis charts using Plotly or Matplotlib."""
    
    def plot_line_trend(self, df: pd.DataFrame, x_col: str, y_cols: Union[str, List[str]], title: str = "Trend Analysis", engine: str = "plotly") -> Any:
        """Plots line trend."""
        if df.empty or x_col not in df.columns:
            logger.warning("Empty dataframe or missing x_col for plot_line_trend")
            return None

        if isinstance(y_cols, str):
            cols = [y_cols]
        else:
            cols = y_cols
            
        if engine == "plotly":
            fig = go.Figure()
            for y_col in cols:
                if y_col in df.columns:
                    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', name=y_col))
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title="Value",
                xaxis=dict(rangeslider=dict(visible=True)),
                template="plotly_dark"
            )
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))
            for y_col in cols:
                if y_col in df.columns:
                    ax.plot(df[x_col], df[y_col], label=y_col)
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def plot_plotly(self, df: pd.DataFrame, x_col: str, y_col: Union[str, List[str]], title: str = "Trend Analysis") -> Any:
        """Convenience method for Plotly line trend."""
        return self.plot_line_trend(df, x_col, y_col, title=title, engine="plotly")

    def plot_matplotlib(self, df: pd.DataFrame, x_col: str, y_col: Union[str, List[str]], title: str = "Trend Analysis") -> Any:
        """Convenience method for Matplotlib line trend."""
        return self.plot_line_trend(df, x_col, y_col, title=title, engine="matplotlib")
            
    def plot_cumulative_trend(self, df: pd.DataFrame, x_col: str, y_col: str, title: str = "Cumulative Trend", engine: str = "plotly") -> Any:
        """Plots cumulative trend."""
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            logger.warning("Invalid data for plot_cumulative_trend")
            return None
            
        cum_series = df[y_col].cumsum()
        
        if engine == "plotly":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[x_col], y=cum_series, mode='lines', fill='tozeroy', name=f"Cumulative {y_col}"))
            fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=f"Cumulative {y_col}")
            return fig
        elif engine == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df[x_col], cum_series, label=f"Cumulative {y_col}")
            ax.fill_between(df[x_col], cum_series, alpha=0.3)
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(f"Cumulative {y_col}")
            ax.grid(True)
            ax.legend()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")
            
    def plot_multi_axis_trend(self, df: pd.DataFrame, x_col: str, y1_col: str, y2_col: str, engine: str = "plotly") -> Any:
        """Plots multi-axis trend."""
        if df.empty or any(col not in df.columns for col in [x_col, y1_col, y2_col]):
            logger.warning("Invalid data for plot_multi_axis_trend")
            return None
            
        if engine == "plotly":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y1_col], name=y1_col, mode='lines'), secondary_y=False)
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y2_col], name=y2_col, mode='lines'), secondary_y=True)
            fig.update_layout(title_text=f"{y1_col} and {y2_col} Trends")
            fig.update_xaxes(title_text=x_col)
            fig.update_yaxes(title_text=y1_col, secondary_y=False)
            fig.update_yaxes(title_text=y2_col, secondary_y=True)
            return fig
        elif engine == "matplotlib":
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel(x_col)
            ax1.set_ylabel(y1_col, color='tab:blue')
            ax1.plot(df[x_col], df[y1_col], color='tab:blue', label=y1_col)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel(y2_col, color='tab:orange')
            ax2.plot(df[x_col], df[y2_col], color='tab:orange', label=y2_col)
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            
            fig.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported engine: {engine}")
