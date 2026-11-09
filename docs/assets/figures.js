/* Shared theme selection; geometry stays in the exported vector sources. */
(function () {
  "use strict";
  function variants(src) {
    const url = new URL(src, document.baseURI);
    if (!/\/tensor-analysis\/figures\/[^/]+\.(en|zh)(\.dark)?\.svg$/.test(url.pathname) &&
        !/\/visuals\/figures\/(pipeline|judgments)(-mobile)?(-zh)?(\.dark)?\.svg$/.test(url.pathname)) return null;
    url.pathname = url.pathname.replace(/(?:\.dark)?\.svg$/, ".svg");
    const light = url.href;
    url.pathname = url.pathname.replace(/\.svg$/, ".dark.svg");
    return {light: light, dark: url.href};
  }
  function sync(root, dark) {
    (root || document).querySelectorAll("img[src]").forEach(function (img) {
      // Responsive picture sources take precedence over the img fallback.
      const picture = img.closest("picture");
      if (picture) picture.querySelectorAll("source[srcset]").forEach(function (source) {
        const pair = variants(source.getAttribute("srcset"));
        if (pair) source.srcset = pair[dark ? "dark" : "light"];
      });
      const pair = variants(img.getAttribute("src"));
      if (!pair) return;
      img.dataset.tensorFigure = "true";
      img.dataset.figureTheme = dark ? "dark" : "light";
      const src = pair[dark ? "dark" : "light"];
      if (img.src !== src) img.src = src;
      const dialog = img.closest("dialog");
      if (dialog) dialog.querySelectorAll("a[data-figure-original]").forEach(function (a) { a.href = src; });
    });
  }
  window.MetisFigures = {sync: sync};
}());
