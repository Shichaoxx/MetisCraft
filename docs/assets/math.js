/* Shared by the documentation and the project showcase. No remote math assets. */
(function () {
  "use strict";
  window.renderMetisMath = function (root = document) {
    if (!window.katex) return;
    root.querySelectorAll(".arithmatex:not([data-math-rendered])").forEach(function (element) {
      const source = element.textContent.replace(/^\\\[|\\\]$|^\\\(|\\\)$/g, "");
      try {
        window.katex.render(source, element, {
          displayMode: element.tagName === "DIV",
          throwOnError: true,
          strict: "ignore",
          trust: false
        });
        element.dataset.mathRendered = "true";
      } catch (error) {
        element.dataset.mathRendered = "false";
        element.classList.add("math-render-error");
        console.error("MetisCraft formula failed to render", source, error);
      }
    });
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { window.renderMetisMath(); });
  } else {
    window.renderMetisMath();
  }
  if (typeof document$ !== "undefined") document$.subscribe(function () { window.renderMetisMath(); });
}());
