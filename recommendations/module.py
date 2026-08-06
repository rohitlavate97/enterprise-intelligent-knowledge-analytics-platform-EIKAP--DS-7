import time
import pandas as pd
from typing import Dict, Any, List, Union, Type
from pydantic import Field, BaseModel

from module_registry.base import EIKAPModule, ModuleMetadata, MaturityLabel, ModuleCategory
from module_registry.schemas import BaseInputSchema, BaseOutputSchema
from recommendations.hybrid import HybridRecommender
from recommendations.evaluation import RecommendationEvaluator
from shared.logging import get_logger
from shared.exceptions import PipelineError

logger = get_logger(__name__)

class RecommendationInput(BaseInputSchema):
    user_id: Union[str, int] = Field(default="", description="The ID of the user to get recommendations for.")
    top_k: int = Field(default=10, description="Number of recommendations to retrieve.")
    cf_weight: float = Field(default=0.5, description="Weight for Collaborative Filtering.")
    user_history_item_ids: List[Any] = Field(default_factory=list, description="User history items for cold start fallback.")

class RecommendationOutput(BaseOutputSchema):
    user_id: Union[str, int]
    recommendations: List[Dict[str, Any]]
    metrics: Dict[str, float]
    execution_time_ms: float

class RecommendationModule(EIKAPModule):
    """
    EIKAP Recommendation Engine Module.
    Integrates HybridRecommender and evaluation metrics.
    """

    def __init__(self, cf_weight: float = 0.5):
        super().__init__()
        self._metadata = ModuleMetadata(
            name="recommendation_engine",
            version="0.1.0",
            description="Hybrid Recommendation & Personalized Analytics Engine",
            category=ModuleCategory.MACHINE_LEARNING,
            maturity=MaturityLabel.STANDARD,
            author="EIKAP Team",
            tags=["recsys", "hybrid", "collaborative filtering", "content-based"]
        )
        self.recommender = HybridRecommender(cf_weight=cf_weight)
        self.evaluator = RecommendationEvaluator()
        self.is_fitted = False

    @property
    def metadata(self) -> ModuleMetadata:
        return self._metadata

    @property
    def input_schema(self) -> Type[BaseModel]:
        return RecommendationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return RecommendationOutput

    def train(self, data: pd.DataFrame, items: pd.DataFrame = None, **kwargs) -> Dict[str, Any]:
        """Fits hybrid recommender on interactions DataFrame and item catalog DataFrame."""
        try:
            self.recommender.fit(
                interactions_df=data,
                items_df=items,
                user_col="user_id",
                item_col="item_id",
                rating_col="rating",
                item_text_col="description"
            )
            self.is_fitted = True
            logger.info("RecommendationModule trained successfully.")
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Failed to train RecommendationModule: {e}")
            raise PipelineError(f"Failed to train RecommendationModule: {e}")

    def predict(self, input_data: RecommendationInput, **kwargs) -> RecommendationOutput:
        """Routes user recommendation requests through HybridRecommender, returns validated output."""
        start_time = time.time()
        
        if not self.is_fitted:
            raise PipelineError("Module must be trained before calling predict().")
            
        try:
            recs = self.recommender.recommend(
                user_id=input_data.user_id,
                user_history_item_ids=input_data.user_history_item_ids,
                top_k=input_data.top_k,
                cf_weight=input_data.cf_weight
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return RecommendationOutput(
                prediction=recs,
                maturity_label=self.metadata.maturity.value,
                module_name=self.metadata.name,
                request_id=input_data.request_id,
                user_id=input_data.user_id,
                recommendations=recs,
                metrics={"num_recommendations": float(len(recs))},
                execution_time_ms=execution_time_ms
            )
        except Exception as e:
            logger.error(f"Prediction failed in RecommendationModule: {e}")
            raise PipelineError(f"Prediction failed: {e}")

    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        """Returns breakdown of collaborative vs content-based recommendation contributions."""
        if not self.is_fitted:
            raise PipelineError("Module must be trained before calling explain().")
            
        cf_score = 0.0
        cb_score = 0.0
        
        user_id = input_data.user_id if hasattr(input_data, 'user_id') else kwargs.get("user_id")
        user_history_item_ids = input_data.user_history_item_ids if hasattr(input_data, 'user_history_item_ids') else kwargs.get("user_history_item_ids")
        recommended_item_id = kwargs.get("recommended_item_id")
        
        # Check CF Score
        cf_recs = self.recommender.cf_recommender.recommend_for_user(user_id=user_id, top_k=100)
        for r in cf_recs:
            if r.get("item_id") == recommended_item_id:
                cf_score = r.get("score", 0.0)
                break
                
        # Check CB Score
        if user_history_item_ids:
            cb_recs = self.recommender.cb_recommender.recommend_for_user_profile(user_item_ids=user_history_item_ids, top_k=100)
            for r in cb_recs:
                if r.get("item_id") == recommended_item_id:
                    cb_score = r.get("similarity_score", 0.0)
                    break
                    
        return {
            "cf_contribution": cf_score * self.recommender.cf_weight,
            "cb_contribution": cb_score * (1.0 - self.recommender.cf_weight)
        }

    def evaluate(self, test_data: Dict[Any, List[Any]], k: int = 10, **kwargs) -> Dict[str, float]:
        """Evaluates recommendation Precision@K, Recall@K, MAP@K, NDCG@K on test set."""
        if not self.is_fitted:
            raise PipelineError("Module must be trained before calling evaluate().")
            
        return self.evaluator.evaluate_all(test_data, recommender=self.recommender, k=k)

    def health_check(self) -> Dict[str, Any]:
        """Verifies SciPy, Scikit-Learn recommendation dependencies."""
        health = {"status": "healthy"}
        try:
            import scipy
            health["scipy_version"] = scipy.__version__
        except ImportError:
            health["status"] = "unhealthy"
            health["scipy_error"] = "scipy is not installed"
            
        try:
            import sklearn
            health["sklearn_version"] = sklearn.__version__
        except ImportError:
            health["status"] = "unhealthy"
            health["sklearn_error"] = "scikit-learn is not installed"
            
        return health
