# Cookbook

Inspect a recorded support decision, train a model on a CPU or examine a reproduction run. Follow the [installation guide](../installation.md) before starting a training tutorial.

## Tutorials { #choose-a-starting-point }

<div class="start-grid cookbook-paths" markdown="1">

<div markdown="1">

### Customer routing { #01--see-a-decision }

Follow a customer message through two decisions, with the full request, recorded response and evaluation evidence.

[Open the recorded example](decision.md) →

</div>
<div markdown="1">

### CPU training { #02--run-your-first-model }

No GPU or weight download. Use a tiny random model to learn training, calibration, evaluation and HTTP calls.

[Start the CPU tutorial](first-model.md) →

</div>
<div markdown="1">

### Custom tasks { #03--bring-your-own-task }

Replace the labels and pretrained base, then produce an artifact you can load, evaluate and serve.

[Use your labeled data](own-task.md) →

</div>
</div>

## Experiment recipes and results { #from-a-question-to-an-application }

[Banking77 customer routing](../../../recipes/banking77_support/README_en.md) derives queue and urgency labels from 77 original intents and documents the data splits, training commands and held-out results. The [GoEmotions recipe](../../../recipes/go_emotions/README_en.md) trains a multi-label model: one text can have several emotion labels.

The [Tev1 4B reproduction](../reproductions/tev1.md) documents both evaluation sets, compares results with the authors’ numbers and describes differences in training and calibration. The [training recipe](../../../recipes/tev1/README_en.md) gives the steps for a new run.

<div class="cookbook-guide" data-metis-theme="light" data-metis-light-src="../../visuals/figures/metis-guide-chibi.webp" data-metis-light-full-src="../../visuals/figures/metis-guide-chibi.png" data-metis-dark-src="../../visuals/figures/metis-guide-night.webp" data-metis-dark-full-src="../../visuals/figures/metis-guide-night.png" data-metis-light-alt="Metis in her academy outfit, holding a star-chart book" data-metis-dark-alt="Metis in her night outfit, holding a star-chart book" markdown="1">

<figure class="cookbook-avatar"><a data-metis-full href="../../visuals/figures/metis-guide-chibi.png" target="_blank" rel="noopener" aria-label="View Metis's illustration in a new tab"><img data-metis-portrait loading="lazy" decoding="async" src="../../visuals/figures/metis-guide-chibi.webp" alt="Metis in her academy outfit, holding a star-chart book" width="160" height="160"></a></figure>

<div class="cookbook-guide-copy" markdown="1">

## Metis’s note on experiment logs { #metis--your-guide }

Open `report.md` in the run directory after training. For the same metrics in machine-readable form, look in `eval/metrics.json`.

</div>
</div>

## Manual and reference { #reference-when-you-need-it }

- [Installation](../installation.md) and [supported scope](../limitations.md).
- [Runtime, interface and artifact compatibility](../compatibility.md).
- [Record JSONL and dataset manifests](../data.md).
- [Base preparation and recipe settings](../training.md).
- [Prediction, HTTP and task routing](../serving.md).
- [Recipe field reference](../recipe-reference.md) and [Concepts & Tensors](../tensor-analysis/index.md).

[中文](../../zh/cookbook/index.md)
