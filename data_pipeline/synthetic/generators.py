import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base_generator import BaseSyntheticGenerator

class CustomerChurnGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Customer churn data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame({
            "customer_id": [f"CUST_{i}" for i in range(n_samples)],
            "tenure_months": self.rng.integers(1, 73, n_samples),
            "monthly_charges": self.rng.uniform(20.0, 120.0, n_samples),
            "contract_type": self.rng.choice(["month-to-month", "one-year", "two-year"], n_samples),
            "internet_service": self.rng.choice(["DSL", "Fiber", "None"], n_samples),
            "online_security": self.rng.choice(["Yes", "No"], n_samples),
            "tech_support": self.rng.choice(["Yes", "No"], n_samples),
            "payment_method": self.rng.choice(["Electronic", "Mailed", "Bank transfer", "Credit card"], n_samples),
            "gender": self.rng.choice(["Male", "Female"], n_samples),
            "senior_citizen": self.rng.choice([0, 1], n_samples, p=[0.85, 0.15]),
            "partner": self.rng.choice(["Yes", "No"], n_samples),
            "dependents": self.rng.choice(["Yes", "No"], n_samples),
            "num_support_tickets": self.rng.integers(0, 16, n_samples),
            "avg_monthly_usage_gb": self.rng.uniform(1.0, 500.0, n_samples),
            "last_interaction_days": self.rng.integers(0, 366, n_samples),
            "satisfaction_score": self.rng.integers(1, 6, n_samples)
        })
        df["total_charges"] = df["tenure_months"] * df["monthly_charges"]
        
        score = -2.0
        score += np.where(df["contract_type"] == "month-to-month", 1.5, 0.0)
        score -= df["tenure_months"] * 0.03
        score += df["num_support_tickets"] * 0.2
        prob = 1 / (1 + np.exp(-score))
        df["churned"] = self.rng.binomial(1, prob)
        return df

class FraudDetectionGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Transaction fraud data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        amounts = self.rng.lognormal(3, 1, n_samples)
        df = pd.DataFrame({
            "transaction_id": [f"TXN_{i}" for i in range(n_samples)],
            "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="1min"),
            "amount": amounts,
            "merchant_category": self.rng.choice(["Retail", "Travel", "Food", "Entertainment"], n_samples),
            "card_type": self.rng.choice(["Credit", "Debit"], n_samples),
            "is_international": self.rng.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "hour_of_day": self.rng.integers(0, 24, n_samples),
            "day_of_week": self.rng.integers(0, 7, n_samples),
            "customer_age": self.rng.integers(18, 80, n_samples),
            "account_age_days": self.rng.integers(1, 3000, n_samples),
            "num_transactions_24h": self.rng.integers(1, 20, n_samples),
            "avg_transaction_amount_30d": self.rng.uniform(10, 1000, n_samples),
            "distance_from_home_km": self.rng.exponential(50, n_samples),
            "is_weekend": self.rng.choice([0, 1], n_samples, p=[0.71, 0.29]),
            "device_type": self.rng.choice(["Mobile", "Desktop", "Tablet"], n_samples)
        })
        
        score = -4.0
        score += np.where(df["amount"] > 1000, 1.5, 0)
        score += df["is_international"] * 2.0
        score += np.where((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 4), 1.0, 0)
        score += np.where(df["num_transactions_24h"] > 10, 1.0, 0)
        score += np.where(df["distance_from_home_km"] > 500, 1.0, 0)
        
        prob = 1 / (1 + np.exp(-score))
        df["is_fraud"] = self.rng.binomial(1, prob)
        return df

class CreditRiskGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Credit risk data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame({
            "application_id": [f"APP_{i}" for i in range(n_samples)],
            "annual_income": self.rng.lognormal(11, 0.5, n_samples),
            "debt_to_income_ratio": self.rng.uniform(0.1, 0.8, n_samples),
            "credit_score": self.rng.integers(300, 851, n_samples),
            "employment_length_years": self.rng.integers(0, 40, n_samples),
            "loan_amount": self.rng.uniform(1000, 50000, n_samples),
            "loan_purpose": self.rng.choice(["Debt Consolidation", "Home Improvement", "Business"], n_samples),
            "home_ownership": self.rng.choice(["Rent", "Mortgage", "Own"], n_samples),
            "num_open_accounts": self.rng.integers(1, 20, n_samples),
            "num_delinquencies": self.rng.choice([0, 1, 2, 3, 4], n_samples, p=[0.7, 0.15, 0.1, 0.03, 0.02]),
            "total_credit_limit": self.rng.uniform(2000, 100000, n_samples),
            "num_hard_inquiries": self.rng.integers(0, 10, n_samples),
            "months_since_last_delinquency": self.rng.integers(0, 120, n_samples)
        })
        df["credit_utilization_ratio"] = self.rng.uniform(0.0, 1.0, n_samples)
        
        score = -3.0
        score += df["debt_to_income_ratio"] * 2.0
        score -= (df["credit_score"] - 600) / 100
        score += df["credit_utilization_ratio"] * 1.5
        score += df["num_delinquencies"] * 1.0
        
        prob = 1 / (1 + np.exp(-score))
        df["is_default"] = self.rng.binomial(1, prob)
        return df

class ProductRecommendationGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Product recommendation data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame({
            "user_id": self.rng.integers(1, 1001, n_samples),
            "product_id": self.rng.integers(1, 501, n_samples),
            "product_category": self.rng.choice(["Electronics", "Clothing", "Books", "Home"], n_samples),
            "product_price": self.rng.uniform(5, 500, n_samples),
            "rating": self.rng.integers(1, 6, n_samples),
            "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="1h"),
            "user_age": self.rng.integers(18, 70, n_samples),
            "user_gender": self.rng.choice(["M", "F", "O"], n_samples),
            "purchase_count": self.rng.integers(1, 50, n_samples),
            "view_count": self.rng.integers(1, 100, n_samples),
            "time_on_page_seconds": self.rng.exponential(60, n_samples)
        })
        
        score = -1.0
        score += (df["rating"] - 3) * 0.5
        score += (df["view_count"] / 100) * 1.0
        score += (df["time_on_page_seconds"] / 120) * 0.5
        
        prob = 1 / (1 + np.exp(-score))
        df["was_purchased"] = self.rng.binomial(1, prob)
        return df

class SentimentAnalysisGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Sentiment analysis data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        templates = [
            "This product is {adj}.",
            "I {verb} the {feature}.",
            "Very {adj} experience overall."
        ] * 17
        templates = templates[:50]
        adjs = ["great", "terrible", "okay", "amazing", "awful"]
        verbs = ["loved", "hated", "liked", "disliked", "enjoyed"]
        features = ["quality", "design", "price", "packaging"]
        
        texts = []
        for _ in range(n_samples):
            t = self.rng.choice(templates)
            texts.append(t.format(adj=self.rng.choice(adjs), verb=self.rng.choice(verbs), feature=self.rng.choice(features)))
            
        df = pd.DataFrame({
            "review_id": [f"REV_{i}" for i in range(n_samples)],
            "text": texts,
            "rating": self.rng.integers(1, 6, n_samples),
            "product_category": self.rng.choice(["Electronics", "Clothing", "Books", "Home"], n_samples),
            "reviewer_experience_level": self.rng.choice(["Beginner", "Intermediate", "Expert"], n_samples),
            "verified_purchase": self.rng.choice([True, False], n_samples),
            "helpful_votes": self.rng.integers(0, 100, n_samples),
            "review_length_chars": [len(t) for t in texts],
            "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="1h")
        })
        
        sentiments = []
        for r in df["rating"]:
            if r >= 4: sentiments.append("positive")
            elif r <= 2: sentiments.append("negative")
            else: sentiments.append("neutral")
        df["sentiment"] = sentiments
        return df

class CustomerSupportGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Customer support data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame({
            "ticket_id": [f"TKT_{i}" for i in range(n_samples)],
            "customer_id": [f"CUST_{i%1000}" for i in range(n_samples)],
            "subject": ["Support Request"] * n_samples,
            "description": ["Need help"] * n_samples,
            "category": self.rng.choice(["billing", "technical", "account", "general"], n_samples),
            "priority": self.rng.choice(["low", "medium", "high", "critical"], n_samples),
            "status": self.rng.choice(["open", "in_progress", "resolved", "closed"], n_samples),
            "created_at": pd.date_range("2023-01-01", periods=n_samples, freq="1h"),
            "resolution_time_hours": self.rng.exponential(24, n_samples),
            "agent_id": self.rng.integers(1, 50, n_samples),
            "satisfaction_rating": self.rng.integers(1, 6, n_samples),
            "num_interactions": self.rng.integers(1, 10, n_samples),
            "channel": self.rng.choice(["email", "phone", "chat", "web"], n_samples)
        })
        df["resolved_at"] = df["created_at"] + pd.to_timedelta(df["resolution_time_hours"], unit="h")
        return df

class DocumentSearchGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Document search data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "document_id": [f"DOC_{i}" for i in range(n_samples)],
            "title": [f"Document {i}" for i in range(n_samples)],
            "content": ["Sample content for document..."] * n_samples,
            "category": self.rng.choice(["policy", "procedure", "technical", "faq", "report"], n_samples),
            "author": [f"User_{self.rng.integers(1,100)}" for _ in range(n_samples)],
            "created_date": pd.date_range("2023-01-01", periods=n_samples, freq="1h"),
            "last_modified": pd.date_range("2023-02-01", periods=n_samples, freq="1h"),
            "department": self.rng.choice(["HR", "Engineering", "Sales", "Marketing"], n_samples),
            "tags": [["tag1", "tag2"] for _ in range(n_samples)],
            "word_count": self.rng.integers(200, 500, n_samples),
            "language": ["en"] * n_samples,
            "access_level": self.rng.choice(["public", "internal", "confidential"], n_samples)
        })

class ResumeScreeningGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Resume screening data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame({
            "candidate_id": [f"CAND_{i}" for i in range(n_samples)],
            "years_experience": self.rng.integers(0, 20, n_samples),
            "education_level": self.rng.choice(["high_school", "bachelors", "masters", "phd"], n_samples),
            "skills": [["Python", "SQL", "ML"] for _ in range(n_samples)],
            "previous_companies_count": self.rng.integers(1, 10, n_samples),
            "current_role": ["Software Engineer"] * n_samples,
            "industry": ["Tech"] * n_samples,
            "gpa": self.rng.uniform(2.0, 4.0, n_samples),
            "certifications_count": self.rng.integers(0, 5, n_samples),
            "languages_count": self.rng.integers(1, 4, n_samples),
            "job_applied_for": ["Data Scientist"] * n_samples,
        })
        
        score = (df["years_experience"] / 20) * 0.7 + (df["gpa"] / 4.0) * 0.3
        df["relevance_score"] = score
        
        recs = []
        for s in score:
            if s > 0.8: recs.append("shortlist")
            elif s < 0.4: recs.append("reject")
            else: recs.append("review")
        df["recommendation"] = recs
        return df

class FinancialAnalyticsGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Financial analytics data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
            "revenue": self.rng.uniform(10000, 50000, n_samples),
            "costs": self.rng.uniform(5000, 30000, n_samples),
            "profit": self.rng.uniform(1000, 20000, n_samples),
            "gross_margin": self.rng.uniform(0.1, 0.6, n_samples),
            "operating_expenses": self.rng.uniform(2000, 10000, n_samples),
            "ebitda": self.rng.uniform(1000, 15000, n_samples),
            "net_income": self.rng.uniform(500, 10000, n_samples),
            "cash_flow": self.rng.uniform(-5000, 15000, n_samples),
            "accounts_receivable": self.rng.uniform(1000, 20000, n_samples),
            "accounts_payable": self.rng.uniform(1000, 15000, n_samples),
            "inventory_value": self.rng.uniform(5000, 50000, n_samples),
            "department": self.rng.choice(["Sales", "R&D", "Marketing"], n_samples),
            "region": self.rng.choice(["NA", "EMEA", "APAC"], n_samples)
        })

class ImageClassificationGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Image classification data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "image_id": [f"IMG_{i}" for i in range(n_samples)],
            "file_path": [f"/data/img_{i}.jpg" for i in range(n_samples)],
            "category": self.rng.integers(0, 10, n_samples),
            "width": [224] * n_samples,
            "height": [224] * n_samples,
            "file_size_kb": self.rng.uniform(10, 500, n_samples),
            "format": self.rng.choice(["jpg", "png"], n_samples),
            "is_augmented": self.rng.choice([True, False], n_samples),
            "split": self.rng.choice(["train", "val", "test"], n_samples, p=[0.7, 0.15, 0.15]),
            "brightness_score": self.rng.uniform(0.0, 1.0, n_samples),
            "contrast_score": self.rng.uniform(0.0, 1.0, n_samples),
            "has_background_noise": self.rng.choice([True, False], n_samples)
        })

class OCRGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "OCR metadata"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "document_id": [f"DOC_{i}" for i in range(n_samples)],
            "page_number": self.rng.integers(1, 10, n_samples),
            "text_content": ["Extracted text..."] * n_samples,
            "confidence_score": self.rng.uniform(0.0, 1.0, n_samples),
            "language": ["en"] * n_samples,
            "has_tables": self.rng.choice([True, False], n_samples),
            "has_images": self.rng.choice([True, False], n_samples),
            "processing_time_ms": self.rng.uniform(10, 5000, n_samples),
            "text_density": self.rng.uniform(0.0, 1.0, n_samples),
            "document_type": self.rng.choice(["invoice", "receipt", "letter", "form"], n_samples),
            "num_words": self.rng.integers(10, 1000, n_samples),
            "error_rate": self.rng.uniform(0.0, 0.5, n_samples)
        })

class KnowledgeAssistantGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Knowledge assistant Q&A pairs"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "question_id": [f"Q_{i}" for i in range(n_samples)],
            "question": ["What is the policy?"] * n_samples,
            "answer": ["The policy is..."] * n_samples,
            "source_document_id": [f"DOC_{i%100}" for i in range(n_samples)],
            "confidence_score": self.rng.uniform(0.0, 1.0, n_samples),
            "category": self.rng.choice(["HR", "IT", "Finance"], n_samples),
            "is_grounded": self.rng.choice([True, False], n_samples, p=[0.9, 0.1]),
            "citations": [["doc_1", "doc_2"] for _ in range(n_samples)],
            "response_time_ms": self.rng.uniform(100, 2000, n_samples)
        })

class MultiDocQAGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Multi-document QA data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "query_id": [f"Q_{i}" for i in range(n_samples)],
            "query": ["Compare X and Y"] * n_samples,
            "relevant_doc_ids": [["DOC_1", "DOC_2"] for _ in range(n_samples)],
            "answer": ["X is better because..."] * n_samples,
            "num_sources": self.rng.integers(1, 5, n_samples),
            "avg_relevance_score": self.rng.uniform(0.0, 1.0, n_samples),
            "query_type": self.rng.choice(["factual", "comparative", "analytical"], n_samples),
            "complexity": self.rng.choice(["simple", "moderate", "complex"], n_samples)
        })

class SalesAnalyticsGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "Sales data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "sale_id": [f"SALE_{i}" for i in range(n_samples)],
            "date": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
            "product_id": [f"PROD_{i%100}" for i in range(n_samples)],
            "product_name": ["Product"] * n_samples,
            "category": self.rng.choice(["Software", "Hardware", "Services"], n_samples),
            "quantity": self.rng.integers(1, 100, n_samples),
            "unit_price": self.rng.uniform(10, 1000, n_samples),
            "total_amount": self.rng.uniform(10, 100000, n_samples),
            "discount_percentage": self.rng.uniform(0.0, 0.5, n_samples),
            "customer_segment": self.rng.choice(["enterprise", "mid-market", "smb", "consumer"], n_samples),
            "region": self.rng.choice(["NA", "EMEA", "APAC"], n_samples),
            "sales_rep_id": self.rng.integers(1, 50, n_samples),
            "channel": self.rng.choice(["direct", "online", "partner"], n_samples),
            "is_returned": self.rng.choice([True, False], n_samples, p=[0.05, 0.95])
        })

class KPIDashboardGenerator(BaseSyntheticGenerator):
    def get_description(self) -> str: return "KPI dashboard data"
    def generate(self, n_samples: int = 10000, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
            "metric_name": self.rng.choice(["revenue", "users", "conversion_rate", "churn_rate", "nps", "cac", "ltv", "mrr", "arr"], n_samples),
            "metric_value": self.rng.uniform(0, 100000, n_samples),
            "previous_period_value": self.rng.uniform(0, 100000, n_samples),
            "change_percentage": self.rng.uniform(-0.5, 0.5, n_samples),
            "target_value": self.rng.uniform(0, 100000, n_samples),
            "is_on_target": self.rng.choice([True, False], n_samples),
            "department": self.rng.choice(["Sales", "Marketing", "Product"], n_samples),
            "region": self.rng.choice(["Global", "NA", "EMEA"], n_samples)
        })
