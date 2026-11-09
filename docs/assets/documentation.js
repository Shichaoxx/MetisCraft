(function () {
  "use strict";
  function isDarkTheme() {
    const scheme = document.body?.getAttribute("data-md-color-scheme") || document.documentElement.getAttribute("data-md-color-scheme");
    if (scheme) return scheme === "slate";
    const theme = document.documentElement.getAttribute("data-theme") || document.body?.getAttribute("data-theme");
    if (theme === "dark" || theme === "light") return theme === "dark";
    if (document.documentElement.classList.contains("dark")) return true;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function syncBrand() {
    const dark = isDarkTheme();
    const src = "/MetisCraft/assets/docs/visuals/figures/mark-" + (dark ? "dark" : "light") + ".svg";
    document.querySelectorAll(".md-logo img").forEach(function (img) { img.src = src; });
    document.querySelectorAll('link[rel~="icon"]').forEach(function (icon) { icon.href = src; });
  }
  function syncPortraits() {
    const theme = isDarkTheme() ? "dark" : "light";
    document.querySelectorAll("[data-metis-theme]").forEach(function (guide) {
      const src = guide.getAttribute("data-metis-" + theme + "-src");
      const alt = guide.getAttribute("data-metis-" + theme + "-alt");
      if (!src || !alt) return;
      guide.dataset.metisTheme = theme;
      guide.querySelectorAll("img[data-metis-portrait]").forEach(function (portrait) {
        if (portrait.getAttribute("src") !== src) portrait.setAttribute("src", src);
        portrait.alt = alt;
      });
      guide.querySelectorAll("a[data-metis-full]").forEach(function (link) { link.setAttribute("href", src); });
    });
  }
  function syncAppearance() { syncBrand(); syncPortraits(); }
  function hydrate() {
    syncAppearance();
    const links = (window.METIS_DOCUMENTATION_LANGUAGES || {})[location.pathname];
    if (links) {
      document.querySelectorAll(".md-select a, a[hreflang]").forEach(function (anchor) {
        const lang = anchor.getAttribute("hreflang") || (anchor.getAttribute("href") || "").match(/\/(en|zh)\//)?.[1];
        if (links[lang]) {
          anchor.setAttribute("href", links[lang]);
          // The policy already resolves the exact paired page. Native navigation
          // avoids Zensical interpreting this page URL as a language-site root.
          anchor.setAttribute("target", "_self");
        }
      });
    }
    if (window.renderMetisMath) window.renderMetisMath();
    document.querySelectorAll(".md-typeset img[src$='.svg'], .md-typeset img[src*='/tensor-analysis/']").forEach(function (img) {
      if (img.dataset.figureReady) return;
      img.dataset.figureReady = "true";
      img.tabIndex = 0;
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", (document.documentElement.lang.startsWith("zh") ? "放大图示：" : "Enlarge figure: ") + img.alt);
      img.addEventListener("click", function () { showFigure(img); });
      img.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); showFigure(img); }
      });
    });
  }
  function showFigure(img) {
    let dialog = document.getElementById("metis-figure-dialog");
    const zh = document.documentElement.lang.startsWith("zh");
    if (!dialog) {
      dialog = document.createElement("dialog");
      dialog.id = "metis-figure-dialog";
      dialog.innerHTML = '<div class="figure-toolbar"><strong></strong><button type="button"></button></div><div class="figure-canvas"><img alt=""></div><a target="_blank" rel="noopener"></a>';
      dialog.querySelector("button").onclick = function () { dialog.close(); };
      dialog.addEventListener("click", function (event) { if (event.target === dialog) dialog.close(); });
      document.body.appendChild(dialog);
    }
    dialog.querySelector("strong").textContent = img.alt;
    const close = dialog.querySelector("button");
    close.textContent = zh ? "关闭 ×" : "Close ×";
    const enlarged = dialog.querySelector("img");
    enlarged.src = img.src;
    enlarged.alt = img.alt;
    const original = dialog.querySelector("a");
    original.href = img.src;
    original.textContent = zh ? "在新标签页查看原尺寸 ↗" : "Open original size in a new tab ↗";
    dialog.showModal();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", hydrate); else hydrate();
  if (typeof document$ !== "undefined") document$.subscribe(hydrate);
  const themeObserver = new MutationObserver(syncAppearance);
  const themeAttributes = {attributes: true, attributeFilter: ["data-md-color-scheme", "data-theme", "class"]};
  themeObserver.observe(document.documentElement, themeAttributes);
  if (document.body) themeObserver.observe(document.body, themeAttributes);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncAppearance);
}());
