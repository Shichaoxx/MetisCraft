# Cookbook

From a recorded response to a decision model of your own.

<div class="cookbook-guide" markdown="1">

<figure class="cookbook-avatar"><img src="../../visuals/figures/metis-guide-chibi.png" alt="Metis holding a star-chart book, your Cookbook guide" width="160" height="240"></figure>

<div class="cookbook-guide-copy" markdown="1">

## Metis · Your guide

See a decision. Run the example. Bring your own data.

</div>
</div>

## Choose a starting point

<div class="start-grid cookbook-paths" markdown="1">

<div markdown="1">

### 01 · See a decision

Follow a customer message through two decisions, with the full request, recorded response and evaluation evidence.

[Open the recorded example](decision.md) →

</div>
<div markdown="1">

### 02 · Run your first model

No GPU or weight download. Use a tiny random model to learn training, calibration, evaluation and HTTP calls.

[Start the CPU tutorial](first-model.md) →

</div>
<div markdown="1">

### 03 · Bring your own task

Replace the labels and pretrained base, then produce an artifact you can load, evaluate and serve.

[Use your labeled data](own-task.md) →

</div>
</div>

## From a question to an application

Define the possible answers, collect examples with labels, and keep separate data for training, calibration and evaluation. A recipe produces an artifact you can load in Python, predict with in batch, or serve over HTTP. Your application decides what action to take from that answer.

For customer routing, this might mean sending a message to the security queue and escalating cases likely to be urgent. The [recorded example](decision.md) shows both decisions, along with a comparison to a baseline that does not read the message.

## Reference when you need it

- [Installation](../installation.md) and [supported scope](../limitations.md).
- [Record JSONL and dataset manifests](../data.md).
- [Base preparation and recipe settings](../training.md).
- [Prediction, HTTP and task routing](../serving.md).
- [Recipe field reference](../recipe-reference.md) and [Concepts & Tensors](../tensor-analysis/index.md).

[中文](../../zh/cookbook/index.md)
