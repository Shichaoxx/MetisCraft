---
hide:
  - toc
---

<div class="home-hero" markdown="1">
<div class="home-intro" markdown="1">

# A toolkit for *JevLike* models.

Train, reproduce and deploy typed decision models. One recipe connects training, calibration, evaluation and serving.

<div class="home-actions" markdown="1">

[Get started](cookbook/first-model.md){ .md-button .md-button--primary }
[Explore the Cookbook](cookbook/index.md)

</div>

[0.1.0 release](https://github.com/Shichaoxx/MetisCraft/releases/tag/v0.1.0) · [Installation](installation.md)

</div>
<div class="home-specimen" markdown="1">

## Start with a recorded decision

<blockquote lang="en">My card was stolen this morning and there are two payments on it that I did not make.</blockquote>

<svg class="choice-readout" viewBox="0 0 335 156" role="img" aria-label="Of six support queues, the model chose security."><use href="../visuals/figures/choice-readout.svg#readout"></use></svg>

<div class="specimen-caption" markdown="1">

A constructed message scored by the customer-routing model. [Inspect the request and response](cookbook/decision.md)

</div>

<details class="specimen-probabilities"><summary>Recorded probabilities</summary>
<dl><dt>queue.confidence</dt><dd>0.9999995054983101</dd><dt>urgent.noul</dt><dd>0.9999314182034214</dd></dl>
</details>

</div>
</div>

## Tev1 4B reproduction

<div class="home-results" markdown="1">
<div markdown="1">

### Main decisions

**87.3%**

873 of 1,000 correct · Original Tev1 **88.0%**

</div>
<div markdown="1">

### Policy transfer

**99.33%**

298 of 300 correct · Original Tev1 **100%**

</div>
</div>

Both predeclared accuracy checks passed on the same public development questions with a single seed. The original training history is unavailable, and calibration can worsen negative log-likelihood.

[Reproduction record and limits](reproductions/tev1.md) · [Training recipe](../../recipes/tev1/README_en.md)

## Choose your next step

<div class="home-paths" markdown="1">

- [Understand a decision](cookbook/decision.md) — Inspect all six queue probabilities and the separate urgency prediction.
- [Run your first model](cookbook/first-model.md) — Complete a CPU example without downloading weights.
- [Bring your own task](cookbook/own-task.md) — Replace the data and base model, keeping the same training and serving path.

</div>

## Typed judgments. Recorded evidence.

Use yes/no, choice, ordered score and multiple labels through Python or batch prediction. HTTP `/v1/systemone` supports the first three.

[Recipes and results](limitations.md) · [Data format](data.md) · [Train and calibrate](training.md) · [Predict and serve](serving.md)

<picture>
  <source media="(max-width: 600px)" srcset="../visuals/figures/pipeline-mobile.svg">
  <img src="../visuals/figures/pipeline.svg" alt="Data, train, calibrate, evaluate, serve">
</picture>

## Understand the computation

Start with [JevLike tasks, data and model architecture](tensor-analysis/introduction.md). Follow [Concepts & Tensors](tensor-analysis/index.md) from the System One contract to answer-token and scalar-head readouts. The [five-minute walkthrough](tensor-analysis/walkthrough.md) connects editable vector figures, numerical examples and source references.
