// Enhance the published score table only when its complete experimental record is available.
(async () => {
  const figure = document.querySelector(".candidate-edit");
  if (!figure) return;
  const controls = Array.from(figure.querySelectorAll('input[name="readout"]'));
  const scores = figure.querySelectorAll("[data-score]");
  const branches = figure.querySelectorAll(".candidate-edit-branches path");
  const source = document.getElementById("candidate-edit-data");
  try {
    const response = await fetch(source.href);
    if (!response.ok) return;
    const record = await response.json();
    if (!controls.every(control => {
      const delta = record.scores[control.value]?.delta;
      return delta?.length === scores.length && delta.every(Number.isFinite);
    })) return;

    figure.querySelector("[data-before]").textContent = record.replacement.from;
    figure.querySelector("[data-after]").textContent = record.replacement.to;
    function select(control) {
      const delta = record.scores[control.value].delta;
      delta.forEach((value, index) => {
        const sign = value > 0 ? "+" : "−";
        scores[index].textContent = value === 0 ? "0" : sign + Math.abs(value).toFixed(6);
        branches[index].classList.toggle("changed", value !== 0);
      });
      figure.querySelector(".candidate-edit-note").textContent = control.dataset.note;
    }
    for (const control of controls) {
      control.addEventListener("change", () => select(control));
    }
    select(controls.find(control => control.checked));
    figure.hidden = false;
    document.querySelector(".candidate-edit-table").open = false;
  } catch {
    // Offline readers can use the initially expanded table below the figure.
  }
})();
