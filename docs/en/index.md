# A toolkit for JevLike models.

<div class="home-lead" markdown="1">

Train, reproduce and deploy JevLike decision models. One recipe connects training, calibration, evaluation and serving.

[Get started](cookbook/first-model.md){ .md-button .md-button--primary }
[Explore the Cookbook](cookbook/index.md){ .md-button }

**0.1.0rc1 is available as a GitHub pre-release.** [Download distributions](https://github.com/Shichaoxx/MetisCraft/releases/tag/0.1.0rc1) · [Installation](installation.md)

[Intro: JevLike, tasks, data and model architecture](tensor-analysis/introduction.md)

</div>

## Tev1 4B reproduction

<div class="reproduction-results" markdown="1">

<div markdown="1">

<span class="result-label">Main decisions · 1,000 questions</span>

**87.3%**

873/1,000 correct · original Tev1 **88.0%**

</div>
<div markdown="1">

<span class="result-label">Policy transfer · 300 questions</span>

**99.33%**

298/300 correct · original Tev1 **100%**

</div>
</div>

Both predeclared accuracy checks passed. Single seed, the same public development questions; original training history is unavailable and calibration can worsen NLL.

[Reproduction record and limits](reproductions/tev1.md) · [Training recipe](../../recipes/tev1/README_en.md)

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

## Understand the computation

Follow [Concepts & Tensors](tensor-analysis/index.md) from the System One contract to answer-token and scalar-head readouts. The [five-minute technical walkthrough](tensor-analysis/walkthrough.md) connects editable vector figures, numerical examples and source references.

[What is JevLike? Task, data and architecture](tensor-analysis/introduction.md)
