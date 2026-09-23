# Data Pipeline

## Pipeline Stages

| Stage | Purpose | Input | Output |
|---|---|---|---|
| Collect | Collect Iris dataset | sklearn Iris dataset | iris_raw.csv |
| Preprocess | Clean and preprocess data | iris_raw.csv | iris_preprocessed.csv |
| Feature Engineering | Create additional features | iris_preprocessed.csv | iris_features.csv |
| Validate | Validate schema, nulls and ranges | iris_features.csv | Validation result |

## Pipeline Flow

Collect
↓
Preprocess
↓
Feature Engineering
↓
Validate

## Validation Rules

- Check expected columns
- Check for null values
- Check valid species values
- Check numeric feature ranges