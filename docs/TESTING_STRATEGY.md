
# Testing Strategy – Experiment 8

## Objective
To test data processing, validation, model training,
and model prediction quality using pytest.

## Test Files

1. test_preprocess.py
   Tests duplicate removal and missing-value handling.

2. test_features.py
   Tests feature calculations and zero division.

3. test_validate.py
   Tests valid and invalid input data.

4. test_training_pipeline.py
   Tests model training, prediction shape,
   and probability outputs.

5. test_model_quality.py
   Tests prediction labels and minimum model accuracy.

## Test Execution
Command: python -m pytest -v

## Coverage Report
Command: python -m pytest --cov=src --cov-report=term-missing

## Result
Record the actual number of passed and failed tests
and the coverage percentage after execution.
