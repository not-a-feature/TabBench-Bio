const CITATION_SOURCES = [
  { path: "CITATION.cff", target: "citation-cff" },
  { path: "CITATION.bib", target: "citation-bibtex" },
];

function setCitationStatus(message) {
  document.getElementById("citation-status").textContent = message;
}

function fallbackCopy(text) {
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("The browser did not permit copying");
}

function copyCitation(button) {
  const text = document.getElementById(button.dataset.copyTarget).textContent;
  const copied = () => {
    button.textContent = "Copied";
    setCitationStatus(`${button.closest("article").querySelector("h3").textContent} copied to clipboard.`);
    window.setTimeout(() => { button.textContent = "Copy"; }, 1800);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(copied, () => {
      fallbackCopy(text);
      copied();
    });
  } else {
    fallbackCopy(text);
    copied();
  }
}

async function initializeCitations() {
  const responses = await Promise.all(CITATION_SOURCES.map(({ path }) => fetch(path, { cache: "no-cache" })));
  responses.forEach((response) => {
    if (!response.ok) throw new Error(`Could not load citation metadata (HTTP ${response.status})`);
  });
  const texts = await Promise.all(responses.map((response) => response.text()));
  CITATION_SOURCES.forEach(({ target }, index) => {
    document.getElementById(target).textContent = texts[index].trim();
  });
  document.querySelectorAll(".citation-copy").forEach((button) => {
    button.disabled = false;
    button.addEventListener("click", () => copyCitation(button));
  });
}

initializeCitations().catch((error) => {
  const message = document.getElementById("load-error");
  message.hidden = false;
  message.textContent = error.message;
  console.error(error);
});
