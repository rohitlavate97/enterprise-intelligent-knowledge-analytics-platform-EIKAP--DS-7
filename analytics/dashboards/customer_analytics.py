import pandas as pd
import numpy as np
from typing import Dict, Any
from shared.logging import get_logger

class CustomerAnalytics:
    """Analytics engine for customer segmentation, RFM, and retention."""
    
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def compute_rfm(self, df: pd.DataFrame, customer_col: str = 'user_id', date_col: str = 'date', amount_col: str = 'revenue') -> pd.DataFrame:
        """Computes Recency, Frequency, Monetary scoring (1-5 scale per dimension)."""
        self.logger.info("Computing RFM scores")
        if df.empty or not all(c in df.columns for c in [customer_col, date_col, amount_col]):
            self.logger.warning("Missing required columns for RFM calculation.")
            return pd.DataFrame()
            
        try:
            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            
            current_date = temp_df[date_col].max() + pd.Timedelta(days=1)
            
            rfm = temp_df.groupby(customer_col).agg({
                date_col: lambda x: (current_date - x.max()).days,
                customer_col: 'count',
                amount_col: 'sum'
            }).rename(columns={
                date_col: 'Recency',
                customer_col: 'Frequency',
                amount_col: 'Monetary'
            })
            
            rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1])
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
            rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
            
            rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
            return rfm.reset_index()
        except Exception as e:
            self.logger.error(f"Error computing RFM: {e}")
            return pd.DataFrame()

    def segment_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Assigns segment labels: Champions, Loyal, Potential Loyalists, At Risk, Lost."""
        self.logger.info("Segmenting customers based on RFM")
        if rfm_df.empty or 'RFM_Score' not in rfm_df.columns:
            return rfm_df
            
        def assign_segment(score_str: str) -> str:
            r, f, m = int(score_str[0]), int(score_str[1]), int(score_str[2])
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3:
                return 'Loyal'
            elif r >= 3 and f < 3:
                return 'Potential Loyalists'
            elif r < 3 and f >= 3:
                return 'At Risk'
            else:
                return 'Lost'
                
        try:
            df_segmented = rfm_df.copy()
            df_segmented['Segment'] = df_segmented['RFM_Score'].apply(assign_segment)
            df_segmented['segment'] = df_segmented['Segment']
            return df_segmented
        except Exception as e:
            self.logger.error(f"Error segmenting customers: {e}")
            return rfm_df

    def rfm_segmentation(self, df: pd.DataFrame, customer_col: str = 'user_id', date_col: str = 'date', amount_col: str = 'revenue') -> pd.DataFrame:
        """Convenience method computing RFM and segmenting in one step."""
        rfm = self.compute_rfm(df, customer_col, date_col, amount_col)
        return self.segment_customers(rfm)

    def compute_cohort_retention(self, df: pd.DataFrame, customer_col: str = 'user_id', date_col: str = 'date') -> pd.DataFrame:
        """Computes cohort retention matrix."""
        self.logger.info("Computing cohort retention")
        if df.empty or not all(c in df.columns for c in [customer_col, date_col]):
            return pd.DataFrame()
            
        try:
            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            
            temp_df['InvoiceMonth'] = temp_df[date_col].dt.to_period('M')
            temp_df['CohortMonth'] = temp_df.groupby(customer_col)['InvoiceMonth'].transform('min')
            
            def get_date_int(df_in, column):
                year = df_in[column].dt.year
                month = df_in[column].dt.month
                return year, month

            invoice_year, invoice_month = get_date_int(temp_df, 'InvoiceMonth')
            cohort_year, cohort_month = get_date_int(temp_df, 'CohortMonth')
            
            years_diff = invoice_year - cohort_year
            months_diff = invoice_month - cohort_month
            
            temp_df['CohortIndex'] = years_diff * 12 + months_diff
            
            cohort_data = temp_df.groupby(['CohortMonth', 'CohortIndex'])[customer_col].nunique().reset_index()
            cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values=customer_col)
            
            cohort_sizes = cohort_counts.iloc[:, 0]
            retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
            retention_df = retention.round(2)
            retention_df['retention'] = retention_df.mean(axis=1)
            return retention_df
        except Exception as e:
            self.logger.error(f"Error computing cohort retention: {e}")
            return pd.DataFrame()

    def cohort_retention(self, df: pd.DataFrame, customer_col: str = 'user_id', date_col: str = 'date') -> pd.DataFrame:
        """Alias for compute_cohort_retention."""
        return self.compute_cohort_retention(df, customer_col, date_col)

    def get_customer_lifetime_value_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Computes CLV distribution from a dataframe."""
        self.logger.info("Computing CLV distribution")
        if df.empty or 'ltv' not in df.columns:
            return {}
            
        try:
            ltv = df['ltv']
            percentiles = np.percentile(ltv.dropna(), [25, 50, 75, 90])
            return {
                "mean": float(ltv.mean()),
                "median": float(ltv.median()),
                "std": float(ltv.std()),
                "percentiles": {
                    "25th": float(percentiles[0]),
                    "50th": float(percentiles[1]),
                    "75th": float(percentiles[2]),
                    "90th": float(percentiles[3]),
                },
                "max": float(ltv.max()),
                "min": float(ltv.min())
            }
        except Exception as e:
            self.logger.error(f"Error computing CLV distribution: {e}")
            return {}
