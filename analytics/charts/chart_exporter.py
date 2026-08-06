"""
Chart exporter module.
"""
import base64
import io
from pathlib import Path
from typing import Union, Any
from shared.logging import get_logger

logger = get_logger(__name__)

class ChartExporter:
    """Exports Matplotlib or Plotly figures to various formats."""
    
    def _is_matplotlib(self, fig: Any) -> bool:
        """Checks if figure is matplotlib."""
        try:
            from matplotlib.figure import Figure
            return isinstance(fig, Figure)
        except ImportError:
            return False
            
    def _is_plotly(self, fig: Any) -> bool:
        """Checks if figure is plotly."""
        try:
            from plotly.graph_objs import Figure
            return isinstance(fig, Figure)
        except ImportError:
            return False

    def to_png(self, fig: Any, output_path: Union[str, Path], width: int = 1200, height: int = 800) -> Path:
        """Exports chart to PNG."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._is_matplotlib(fig):
            fig.savefig(out_path, format="png", bbox_inches="tight")
        elif self._is_plotly(fig):
            fig.write_image(str(out_path), width=width, height=height, format="png")
        else:
            raise ValueError("Unsupported figure type for PNG export.")
            
        logger.info(f"Exported chart to PNG: {out_path}")
        return out_path
        
    def to_svg(self, fig: Any, output_path: Union[str, Path]) -> Path:
        """Exports chart to SVG."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._is_matplotlib(fig):
            fig.savefig(out_path, format="svg", bbox_inches="tight")
        elif self._is_plotly(fig):
            fig.write_image(str(out_path), format="svg")
        else:
            raise ValueError("Unsupported figure type for SVG export.")
            
        logger.info(f"Exported chart to SVG: {out_path}")
        return out_path
        
    def to_html(self, fig: Any, output_path: Union[str, Path], include_plotlyjs: str = "cdn") -> Path:
        """Exports chart to HTML."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._is_plotly(fig):
            fig.write_html(str(out_path), include_plotlyjs=include_plotlyjs)
        elif self._is_matplotlib(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            html_str = f'<html><body><img src="data:image/png;base64,{b64}" alt="Matplotlib Plot"></body></html>'
            out_path.write_text(html_str, encoding="utf-8")
        else:
            raise ValueError("Unsupported figure type for HTML export.")
            
        logger.info(f"Exported chart to HTML: {out_path}")
        return out_path
        
    def to_base64(self, fig: Any, format: str = "png") -> str:
        """Exports chart to base64 string."""
        if self._is_matplotlib(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format=format, bbox_inches="tight")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        elif self._is_plotly(fig):
            img_bytes = fig.to_image(format=format)
            return base64.b64encode(img_bytes).decode("utf-8")
        else:
            raise ValueError("Unsupported figure type for base64 export.")

    def export_base64(self, fig: Any, format: str = "png") -> str:
        """Alias for to_base64."""
        return self.to_base64(fig, format=format)
