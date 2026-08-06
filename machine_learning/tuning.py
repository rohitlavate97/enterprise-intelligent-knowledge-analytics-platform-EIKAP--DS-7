from typing import Any, Dict
from sklearn.model_selection import GridSearchCV

class HyperparameterTuner:
    def __init__(self, estimator: Any, param_grid: Dict[str, list], cv: int = 5, scoring: str = "f1_macro"):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_estimator_ = None
        self.best_params_ = {}

    def tune(self, X: Any, y: Any) -> Any:
        grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        return self.best_estimator_
