/* =====================================================================
   Charles Lippens, portfolio version 4, « le dossier »
   JavaScript vanilla, sans dépendance, aucun cookie ni traceur.
   Modules : thème clair / sombre, sommaire actif au défilement,
   filtres du registre des projets, impression du dossier complet.
   ===================================================================== */
(function () {
  "use strict";

  /* ---------- 1. Thème clair / sombre ---------- */
  function initTheme() {
    var root = document.documentElement;
    var bouton = document.querySelector(".theme-toggle");
    var meta = document.querySelector('meta[name="theme-color"]');
    function appliquer(theme, memoriser) {
      var sombre = theme === "dark";
      root.setAttribute("data-theme", theme);
      if (bouton) {
        bouton.setAttribute("aria-pressed", sombre ? "true" : "false");
        bouton.setAttribute("aria-label", sombre ? "Activer le thème papier (clair)" : "Activer le thème encre (sombre)");
      }
      if (meta) meta.setAttribute("content", sombre ? "#171610" : "#f4efe4");
      if (memoriser) { try { localStorage.setItem("theme", theme); } catch (e) { /* stockage indisponible */ } }
      document.dispatchEvent(new CustomEvent("themechange", { detail: { theme: theme } }));
    }
    appliquer(root.getAttribute("data-theme") || "light", false);
    if (bouton) {
      bouton.addEventListener("click", function () {
        appliquer(root.getAttribute("data-theme") === "dark" ? "light" : "dark", true);
      });
    }
  }

  /* ---------- 2. Sommaire : entrée active selon la section visible ---------- */
  function initSommaire() {
    var liens = Array.prototype.slice.call(document.querySelectorAll(".sommaire a[href^='#']"));
    if (!liens.length || !("IntersectionObserver" in window)) return;
    var parId = {};
    liens.forEach(function (a) { parId[a.getAttribute("href").slice(1)] = a; });
    var sections = Object.keys(parId).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting ? e.intersectionRatio : 0; });
      var meilleur = null, score = 0;
      sections.forEach(function (s) { if (visible[s.id] > score) { score = visible[s.id]; meilleur = s.id; } });
      if (!meilleur) return;
      liens.forEach(function (a) { a.classList.toggle("is-active", a === parId[meilleur]); });
    }, { rootMargin: "-20% 0px -55% 0px", threshold: [0, 0.1, 0.25, 0.5] });
    sections.forEach(function (s) { io.observe(s); });
  }

  /* ---------- 3. Filtres du registre ---------- */
  function initFiltres() {
    var boutons = Array.prototype.slice.call(document.querySelectorAll(".filtre"));
    var projets = Array.prototype.slice.call(document.querySelectorAll(".registre .projet"));
    var vide = document.querySelector(".registre-vide");
    if (!boutons.length || !projets.length) return;
    function appliquer(cle) {
      var visibles = 0;
      projets.forEach(function (p) {
        var cats = (p.getAttribute("data-cat") || "").split(/\s+/);
        var ok = cle === "tous" || cats.indexOf(cle) !== -1;
        p.classList.toggle("cache", !ok);
        if (ok) visibles++;
      });
      if (vide) vide.classList.toggle("visible", visibles === 0);
    }
    boutons.forEach(function (b) {
      b.addEventListener("click", function () {
        boutons.forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        appliquer(b.getAttribute("data-filtre"));
      });
    });
  }

  /* ---------- 4. Impression : ouvrir toutes les entrées, puis rétablir ---------- */
  function initImpression() {
    var lien = document.querySelector(".imprimer");
    if (lien) lien.addEventListener("click", function (e) {
      e.preventDefault();
      // J1 : charger les figures différées avant d'imprimer, sinon elles sortent vides
      var images = Array.prototype.slice.call(document.querySelectorAll('img[loading="lazy"]'));
      images.forEach(function (img) { img.loading = "eager"; });
      Promise.all(images.map(function (img) { return img.decode ? img.decode().catch(function () {}) : null; }))
        .then(function () { window.print(); });
    });
    var ouverts = [];
    window.addEventListener("beforeprint", function () {
      ouverts = [];
      document.querySelectorAll("details.projet").forEach(function (d) {
        ouverts.push(d.open);
        d.open = true;
      });
    });
    window.addEventListener("afterprint", function () {
      document.querySelectorAll("details.projet").forEach(function (d, i) { d.open = ouverts[i] || false; });
    });
  }

  /* ---------- 5. Année du colophon ---------- */
  function initAnnee() {
    document.querySelectorAll("[data-annee]").forEach(function (el) { el.textContent = String(new Date().getFullYear()); });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    initSommaire();
    initFiltres();
    initImpression();
    initAnnee();
  });
})();
