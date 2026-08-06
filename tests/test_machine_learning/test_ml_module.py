import pytest
from machine_learning.module import MachineLearningModule
from machine_learning.schemas import MLInput, MLOutput
from module_registry.base import ModuleCategory, MaturityLabel
from module_registry.contract import UniversalModuleContract

def test_ml_module_metadata(ml_module):
    meta = ml_module.metadata
    assert meta.name == "machine_learning"
    assert meta.version == "0.1.0"
    assert meta.category == ModuleCategory.MACHINE_LEARNING
    assert meta.maturity == MaturityLabel.STANDARD
    assert "scikit-learn" in meta.dependencies

def test_ml_module_predict_churn(ml_module, churn_data):
    X_train, X_test, y_train, y_test = churn_data
    
    # Train
    train_res = ml_module.train({"X": X_train, "y": y_train}, use_case="churn")
    assert train_res["status"] == "success"
    
    # Predict
    input_data = MLInput(use_case="churn", feature_data={"X": X_test})
    output = ml_module.predict(input_data)
    
    assert isinstance(output, MLOutput)
    assert output.use_case == "churn"
    assert len(output.predictions) == len(X_test)
    assert output.execution_time_ms > 0

def test_ml_module_compliance_contract(ml_module, churn_data):
    X_train, X_test, y_train, y_test = churn_data
    
    # Needs to be trained first
    ml_module.train({"X": X_train, "y": y_train}, use_case="churn")
    
    input_data = MLInput(use_case="churn", feature_data={"X": X_test})
    sample_prediction = ml_module.predict(input_data).predictions
    sample_output = ml_module.predict(input_data).model_dump()
    
    contract = UniversalModuleContract()
    report = contract.run_full_check(
        module=ml_module,
        sample_input=input_data,
        sample_prediction=sample_prediction,
        sample_output=sample_output,
        train_data={"X": X_train, "y": y_train},
        test_data={"X": X_test, "y": y_test},
        target_ms=5000.0
    )
    
    assert report.all_passed is True
