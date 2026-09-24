/* Light/dark theme toggle. Persists the choice, follows the OS preference until the user
 * overrides it, and restyles the live Plotly charts via retheme() (from charts.js). */

const SUN_ICON =
  '<circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="4"/>' +
  '<line x1="12" y1="20" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="6.34" y2="6.34"/>' +
  '<line x1="17.66" y1="17.66" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="4" y2="12"/>' +
  '<line x1="20" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="6.34" y2="17.66"/>' +
  '<line x1="17.66" y1="6.34" x2="19.07" y2="4.93"/>';
const MOON_ICON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';

function initTheme() {
  const btn = document.getElementById("theme-toggle");
  const icon = document.getElementById("theme-toggle-icon");
  const apply = (dark) => {
    document.body.classList.toggle("dark-theme", dark);
    if (icon) icon.innerHTML = dark ? SUN_ICON : MOON_ICON;
  };
  let dark = false;
  const saved = localStorage.getItem("theme");
  if (saved) dark = saved === "dark";
  else if (window.matchMedia) dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  apply(dark);
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
      if (!localStorage.getItem("theme")) { apply(e.matches); retheme(); }
    });
  }
  if (btn) {
    btn.onclick = () => {
      const d = !document.body.classList.contains("dark-theme");
      localStorage.setItem("theme", d ? "dark" : "light");
      apply(d);
      retheme();
    };
  }
}
