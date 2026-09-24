"use strict";

const CABIN_CONSENT_KEY = "tabbench-bio-cabin-consent";
const cabinEnabled = localStorage.getItem(CABIN_CONSENT_KEY) !== "denied";

if (cabinEnabled) {
  const script = document.createElement("script");
  script.src = "https://scripts.withcabin.com/hello.js";
  script.async = true;
  script.defer = true;
  document.head.append(script);
}

const preferenceButton = document.querySelector("#analytics-preferences");
if (preferenceButton) {
  preferenceButton.textContent = cabinEnabled ? "Turn off analytics" : "Turn on analytics";
  document.querySelector("#analytics-status").textContent = cabinEnabled
    ? "Analytics is on in this browser."
    : "Analytics is off in this browser.";
  preferenceButton.addEventListener("click", () => {
    localStorage.setItem(CABIN_CONSENT_KEY, cabinEnabled ? "denied" : "granted");
    window.location.reload();
  });
}
