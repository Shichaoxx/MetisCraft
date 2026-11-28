// In this two-candidate exercise exp(logit A) = 3 for both readouts.
(() => {
  const figure = document.querySelector(".score-experiment");
  if (!figure) return;
  const slider = figure.querySelector("input");
  const score = figure.querySelector("output");
  const choice = figure.querySelector('[data-probability="choice"]');
  function update() {
    const logitB = Number(slider.value);
    const probability = 3 / (3 + Math.exp(logitB));
    score.value = logitB.toFixed(1);
    choice.textContent = (100 * probability).toFixed(1) + "%";
    figure.style.setProperty("--choice-probability", choice.textContent);
  }
  slider.addEventListener("input", update);
  update();
  figure.hidden = false;
})();
