"""Excel data loader for EIKAP data pipeline."""
from pathlib import Path
from typing import Any, List, Union, Optional
import pandas as pd

from data_pipeline.loaders.base import BaseLoader
from shared.exceptions import DataLoadError


class ExcelLoader(BaseLoader):
    """Loader for Excel files (.xlsx, .xls)."""

    def _get_supported_extensions(self) -> List[str]:
        return [".xlsx", ".xls"]

    def load(self, source: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        """
        Load Excel data into a pandas DataFrame.
        
        Args:
            source: Path to the Excel file.
            **kwargs: Additional kwargs for pandas.read_excel. 
                      Support 'sheet_name' (default: first sheet).
                      If 'sheet_name=None', reads all sheets and concatenates.
            
        Returns:
            pd.DataFrame containing the loaded data.
        """
        path = self.validate_source(source)
        
        # Determine sheet_name, default is 0 (first sheet)
        sheet_name = kwargs.pop("sheet_name", 0)
        
        try:
            self.logger.info(f"Loading Excel file {path.name}, sheet: {sheet_name}")
            
            # Handling merged cells and other defaults by passing kwargs directly
            data = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
            
            if isinstance(data, dict):
                # Multiple sheets were read, concatenate them
                dfs = []
                for s_name, df in data.items():
                    # Optional: Add a column for sheet name origin
                    df['_source_sheet'] = s_name
                    dfs.append(df)
                
                final_df = pd.concat(dfs, ignore_index=True)
                self.logger.info(
                    f"Successfully loaded all sheets from {path.name}: "
                    f"{len(final_df)} total rows across {len(dfs)} sheets."
                )
            else:
                final_df = data
                self.logger.info(
                    f"Successfully loaded sheet from {path.name}: "
                    f"{len(final_df)} rows, {len(final_df.columns)} columns."
                )
                
            return self._post_load_hook(final_df, path)
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file {path.name}: {str(e)}")
            raise DataLoadError(f"Failed to load Excel file {path}: {str(e)}") from e

    def get_sheet_names(self, source: Union[str, Path]) -> List[str]:
        """
        Get the names of all sheets in the Excel file.
        
        Args:
            source: Path to the Excel file.
            
        Returns:
            List of sheet names as strings.
        """
        path = self.validate_source(source)
        try:
            xl = pd.ExcelFile(path)
            return xl.sheet_names
        except Exception as e:
            self.logger.error(f"Error reading sheet names from {path.name}: {str(e)}")
            raise DataLoadError(f"Failed to read sheet names from {path}: {str(e)}") from e

    def load_sheet(self, source: Union[str, Path], sheet_name: Union[str, int]) -> pd.DataFrame:
        """
        Load a specific sheet from an Excel file.
        
        Args:
            source: Path to the Excel file.
            sheet_name: Name or zero-indexed position of the sheet.
            
        Returns:
            pd.DataFrame containing the sheet's data.
        """
        return self.load(source, sheet_name=sheet_name)
