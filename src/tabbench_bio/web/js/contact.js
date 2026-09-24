"use strict";

const revealContactEmail = document.getElementById("reveal-contact-email");
revealContactEmail.addEventListener("click", () => {
  const address = atob("anVsZXMua3JldWVyQHVuaS10dWViaW5nZW4uZGU=");
  const link = document.createElement("a");
  link.href = "mailto:" + address;
  link.textContent = address;
  revealContactEmail.replaceWith(link);
  link.focus();
}, { once: true });
document.getElementById("contact-email-fallback").hidden = true;
revealContactEmail.hidden = false;
