(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("form"), fileInput = $("file"), drop = $("drop");
  const result = $("result"), loading = $("loading"), errorBox = $("error");
  const csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;
  let file = null, objectUrl = null, ctrl = null;

  const modelo = () => form.querySelector("[name=modelo]:checked").value;
  const nombreModelo = () =>
    form.querySelector("[name=modelo]:checked").closest("label").querySelector(".seg-title").textContent;

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = !msg;
  }

  function setFile(f) {
    if (!f) return;
    file = f;
    $("drop-text").innerHTML = `<b>${f.name.replace(/</g, "&lt;")}</b> · clic para cambiar`;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    $("player").src = objectUrl;
    run();
  }

  async function run() {
    if (!file) return;
    if (ctrl) ctrl.abort();
    ctrl = new AbortController();
    showError("");
    loading.hidden = false;
    result.classList.add("busy");

    const fd = new FormData();
    fd.append("audio", file);
    fd.append("modelo", modelo());
    try {
      const r = await fetch("predecir/", {
        method: "POST", body: fd, headers: { "X-CSRFToken": csrf }, signal: ctrl.signal,
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Error desconocido");
      render(d);
    } catch (e) {
      if (e.name === "AbortError") return;
      showError(e.message);
    } finally {
      loading.hidden = true;
      result.classList.remove("busy");
    }
  }

  function render(d) {
    result.hidden = false;
    $("dur").textContent = `${d.duracion} s`;
    $("onda").src = d.onda;
    $("espec").src = d.espectrograma;
    $("repr").textContent = modelo() === "cnn2d" ? "log-mel · 64×173" : "MFCC + delta";
    $("modelo-usado").textContent = nombreModelo();
    $("clase").textContent = d.etiqueta;
    $("conf").textContent = `Confianza ${(d.confianza * 100).toFixed(1)} %`;

    const ul = $("bars");
    ul.innerHTML = "";
    d.ranking.forEach((it, i) => {
      const li = document.createElement("li");
      if (i === 0) li.className = "top";
      li.innerHTML = `<span class="lbl"><span></span><span class="pct"></span></span><span class="track"><span class="fill"></span></span>`;
      li.querySelector(".lbl span").textContent = it.etiqueta;
      li.querySelector(".pct").textContent = (it.prob * 100).toFixed(1) + "%";
      ul.appendChild(li);
      requestAnimationFrame(() => (li.querySelector(".fill").style.width = it.prob * 100 + "%"));
    });
  }

  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  form.querySelectorAll("[name=modelo]").forEach((r) => r.addEventListener("change", run));

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
})();
