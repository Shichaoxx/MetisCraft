# Cookbook

Turn a business question into a decision you can test and use in an application. These guides move from a recorded example to a local CPU workflow, then to your own labeled data.

<div class="cookbook-guide" markdown="1">

<p><img src="../../visuals/figures/avatar.png" alt="Metis, your Cookbook guide" width="96"></p>

<div markdown="1">

## Metis · Choose your next step

- **See what you can build:** [Understand a decision](decision.md).
- **Run the workflow:** [Your first model on CPU](first-model.md).
- **Use your own labels:** [Bring your own task](own-task.md).
- **Go deeper:** [Principles](../../cookbook/principles.md) and [best practices](../../cookbook/best-practices.md) (Chinese).

</div>
</div>

## Start with an outcome

| Your goal | Guide | What you will have |
|---|---|---|
| Understand the input, output and business use | [Understand a decision](decision.md) | A complete customer-routing request, its recorded model response, and a way to read the evidence |
| Check your installation and the full workflow | [Your first model on CPU](first-model.md) | A local artifact, evaluation report and HTTP response, using a tiny random model |
| Train for a task you already label | [Bring your own task](own-task.md) | A versioned dataset, a recipe linked to your pretrained base, and a plan for evaluating the result |

The customer-routing example uses a trained model and a recorded evaluation. The CPU guide uses synthetic data and random weights to check that the software works. Use the former to understand the kind of result the toolkit can produce; use the latter to learn the commands.

## From a question to an application

Define the possible answers, collect examples with labels, and keep separate data for training, calibration and evaluation. A recipe produces an artifact you can load in Python, predict with in batch, or serve over HTTP. Your application decides what action to take from that answer.

For customer routing, this might mean sending a message to the security queue and escalating cases likely to be urgent. The [recorded example](decision.md) shows both decisions, along with a comparison to a baseline that does not read the message.

## Reference when you need it

- [Installation](../installation.md) and [supported scope](../limitations.md).
- [Record JSONL and dataset manifests](../data.md).
- [Base preparation and recipe settings](../training.md).
- [Prediction, HTTP and task routing](../serving.md).
- [Recipe field reference](../../cookbook/recipe-reference.md) and [readout diagrams](../../readouts/README.md) (Chinese).

[中文](../../zh/cookbook/index.md)
