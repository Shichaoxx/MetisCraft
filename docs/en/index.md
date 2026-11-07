# Your task. A model that decides.

<div class="home-lead" markdown="1">

Train typed decision models on your own data. One recipe connects training, calibration, evaluation and serving.

[Get started](cookbook/first-model.md){ .md-button .md-button--primary }
[Explore the Cookbook](cookbook/index.md){ .md-button }

</div>

## Start with a recorded decision

> My card was stolen this morning and there are two payments on it that I did not make.

<div class="decision-result" markdown="1">

**security** · Route to the security team

`confidence: 0.9999995054983101` · `urgent: 0.9999314182034214`

A constructed message, answered by a recorded customer-routing model. Probabilities are not accuracy scores.

[Open the full example](cookbook/decision.md) →

</div>

## Choose your next step

<div class="start-grid" markdown="1">

<div markdown="1">

### 01 · See a decision

See the input, the typed output and the evidence behind it.

[Understand a decision](cookbook/decision.md)

</div>
<div markdown="1">

### 02 · Run the workflow

Run your first CPU example without a GPU or weight download.

[Run your first model](cookbook/first-model.md)

</div>
<div markdown="1">

### 03 · Bring your task

Replace the data and base model, keeping the same training and serving path.

[Bring your own task](cookbook/own-task.md)

</div>
</div>

## Typed judgments. Recorded evidence.

Use yes/no, choice, ordered score and multiple labels through Python or batch prediction. HTTP `/v1/systemone` supports the first three.

[Recipes and results](limitations.md) · [Data format](data.md) · [Train and calibrate](training.md) · [Predict and serve](serving.md)

<picture>
  <source media="(max-width: 600px)" srcset="../visuals/figures/pipeline-mobile.svg">
  <img src="../visuals/figures/pipeline.svg" alt="Data, train, calibrate, evaluate, serve">
</picture>
