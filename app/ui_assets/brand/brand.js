/* Replaces the stock chat UI's logo and name with the Cognitia Engineering mark and adds a demo notice.
   Re-applies after React re-renders. Local assets only. */
(function () {
  var NOTICE = "Demo on a public filing: Apple 10-K FY2025 · recorded replies or live model · not investment advice";
  function apply() {
    var h1 = document.querySelector('[data-sidebar="header"] h1');
    if (h1 && !h1.querySelector(".cg-brand")) {
      h1.textContent = "";
      var wrap = document.createElement("span");
      wrap.className = "cg-brand";
      wrap.innerHTML =
        '<img class="cg-brand-light" src="/brand/cognitia-mark_v1.svg" alt="" />' +
        '<img class="cg-brand-dark" src="/brand/cognitia-mark-on-dark_v1.svg" alt="" />' +
        "<span>Cognitia Engineering</span>";
      h1.appendChild(wrap);
    }
    var header = document.querySelector('[data-sidebar="header"]');
    if (header && !header.querySelector(".cg-notice")) {
      var n = document.createElement("div");
      n.className = "cg-notice";
      n.textContent = NOTICE;
      header.appendChild(n);
    }
  }
  new MutationObserver(apply).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", apply);
  apply();
})();
