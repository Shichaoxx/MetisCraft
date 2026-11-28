# Candidate scores and probabilities

Two options for a question each receive a score. To interpret the scores as probabilities, first identify the question they belong to and whether both options can be true. We compare two readouts, then work through a batch of questions with different numbers of candidates.

## Where a candidate score comes from

**MetisCraft scalar-list.** For a choice question, the prompt builder records the last token position in each candidate segment. When the prompt order changes, it maps these positions back to the original candidate order. The model gathers the hidden vector at each position and runs the same small network (MLP) on each vector to produce a scalar. Each candidate has its own readout position, and all use the same scoring parameters. [Candidate positions](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/modeling/prompts.py#L220-L235) · [Hidden-state gather](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/modeling/model.py#L79-L81) · [Scalar head](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/modeling/model.py#L14-L29)

**Kev pointer.** Each option ends with `</opt>`, and each question has a `<decide>` marker. The readout projects the decision marker’s hidden state into a query \(q\) and each option-end state into a key \(k_i\). At temperature 1, the score is \(q^\top k_i / \sqrt{d_p}\), where \(d_p\) is the projection dimension. Candidates in one question share its query. When processing multiple questions together, the code uses `owner` to select the query for each candidate. [Option and decision markers](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L107-L126) · [Pointer head and batched scoring](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L205-L223)

## Two questions by hand

Set the temperature to 1. The logits below are teaching values. Question 1 has two valid options, A and B. Its third column contains a deliberately large padding score. Question 2 has three options.

| Question | Scores across three columns | Valid options | Single-choice probabilities |
| --- | --- | --- | --- |
| 1 | ln 3, 0, 90 | A, B | 0.75, 0.25 |
| 2 | 0, ln 2, 0 | A, B, C | 0.25, 0.50, 0.25 |

For a single choice, softmax divides each exponentiated score by the sum over **valid candidates in the same question**. Question 1 has weights 3 and 1, so A gets 3/4. Question 2 uses its own denominator, 1 + 2 + 1. Each row sums to one.

Now use the two valid scores in question 1 for a `multi_label` task in which both labels can apply. An independent sigmoid gives 3/(1+3) = 0.75 and 1/(1+1) = 0.50. Their sum is 1.25. Each value expresses the probability of its own label. Reusing the scores here illustrates task semantics. The Kev choice path above uses a per-question softmax. [MetisCraft probability conversion](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/judgments/__init__.py#L112-L118) · [Kev grouping and normalization](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L494-L505)

<figure class="score-experiment" hidden>
  <figcaption>Change B’s score and watch the probability of A</figcaption>
  <p>A’s logit is fixed at ln 3. Vary B’s logit. Both readouts use the same input scores.</p>
  <label for="score-b">B’s logit <output for="score-b">0.0</output></label>
  <input id="score-b" type="range" min="-3" max="3" step="0.1" value="0">
  <dl>
    <div><dt>Single choice · P(A)</dt><dd><span class="score-track" aria-hidden="true"><i></i></span><span data-probability="choice">75.0%</span></dd></div>
    <div><dt>Multi-label · P(A)</dt><dd><span class="score-track" aria-hidden="true"><i></i></span><span>75.0%</span></dd></div>
  </dl>
</figure>
<script src="/MetisCraft/assets/candidate-scores.js" defer></script>

## Check the calculation in Python

This example uses the Python standard library. Subtracting the largest valid score in a row before exponentiation avoids overflow and preserves the relative probabilities.

```python
from math import exp, log

scores = [[log(3), 0, 90], [0, log(2), 0]]
valid = [[True, True, False], [True, True, True]]

for row, mask in zip(scores, valid):
    logits = [score for score, keep in zip(row, mask) if keep]
    largest = max(logits)
    weights = [exp(score - largest) for score in logits]
    total = sum(weights)
    print([round(weight / total, 4) for weight in weights])

print([round(1 / (1 + exp(-score)), 4) for score in scores[0][:2]])
```

The three lines print `[0.75, 0.25]`, `[0.25, 0.5, 0.25]` and `[0.75, 0.5]`, respectively.

1. **Check candidate IDs and valid columns.** Question 1 should return only A and B. Change the padding score from 90 to −90. Its probabilities should stay at 0.75 and 0.25. MetisCraft builds a candidate mask from each question’s target count, masks invalid columns for training and slices off padding at inference. This mask acts on the output candidate axis. The input attention mask addresses token positions. [Candidate mask](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/train/collate.py#L76-L87) · [Training loss](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/judgments/__init__.py#L83-L97) · [Valid inference scores](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/evaluate/infer.py#L41-L50)
2. **Check question grouping.** Hold both rows of scores fixed and compute question 1 alone, then with question 2. Its result should be identical. If the entire batch sums to one while a question’s valid choice probabilities sum to less than one, check whether candidates from other questions entered its denominator.
3. **Check the task.** Use the sum-to-one check for a question with mutually exclusive options. For independent labels, check each sigmoid on its own. With A’s score held fixed, changing only B’s score should leave A’s probability unchanged.

This exercise verifies the calculation for supplied logits. It uses no weights from either model and provides no comparison of predictive performance.
