# MetisCraft manual

A MetisCraft recipe brings together data preparation, training, calibration and evaluation for a decision model. If you already have a run directory, go directly to prediction and serving below.

## Installation and first run

Follow the [installation guide](../installation.md) to install the package. The Cookbook’s [first model tutorial](../cookbook/first-model.md) walks through a CPU run from preparing example data to loading the trained artifact.

## Data, training and calibration

1. [Prepare your data](../data.md). A Record describes the input, task and label. Assign separate data to training, calibration and evaluation.
2. [Train and evaluate](../training.md). Choose a base model and readout, fit temperature on the calibration set and read metrics on the evaluation set. Look up individual settings in the [recipe reference](../recipe-reference.md).

## Prediction and serving

To check which readout a recipe uses, what training saves and how calibration is tied to an artifact, see [model architecture and training artifacts](model-and-artifacts.md).

[Load and serve an artifact](../serving.md) covers loading a run, predicting batches of Records and starting the HTTP service. The `/v1/systemone` endpoint accepts `noul`, `choice` and `score`. For `multi_label`, use a scalar-readout artifact with the Python predictor or the batch Record JSONL interface.

## Reference

The [interface and artifact reference](../interfaces.md) describes judgment types, response fields and Python entry points. Check [compatibility](../compatibility.md) for environment and artifact-version requirements and [limitations](../limitations.md) for supported use cases.
