#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application finale : détection automatique de faux billets
==========================================================

Projet 12, parcours Data Analyst, ONCFM
Auteur : Charles Lippens
Date   : août 2026

Le script recharge le pipeline scikit-learn entraîné dans le notebook livrable
(StandardScaler suivi d'une LogisticRegression) et détermine, pour chaque billet
fourni, s'il est vrai ou faux, avec la probabilité associée.

Trois modes d'utilisation
-------------------------

1) Mode fichier : un CSV contenant un ou plusieurs billets

   python Lippens_Charles_2_Application_terminal_082026.py --input billets_production.csv

2) Mode arguments : les six dimensions d'un billet passées en ligne de commande

   python Lippens_Charles_2_Application_terminal_082026.py \\
       --diagonal 171.81 --height_left 104.86 --height_right 104.95 \\
       --margin_low 4.52 --margin_up 2.89 --length 112.83

3) Mode interactif : lancé sans argument, le script demande les dimensions
   une par une

   python Lippens_Charles_2_Application_terminal_082026.py

Options complémentaires
-----------------------

  --output FICHIER.csv   exporte les résultats au format CSV
  --seuil 0.7            change le seuil de décision (0,5 par défaut)
  --refuser-manquants    refuse un fichier comportant des valeurs manquantes
                         au lieu de les imputer
  --model CHEMIN         chemin explicite vers le fichier .joblib
  --graphique [IMG.png]  mode visuel des résultats finaux, en plus du terminal :
                         ouvre une fenêtre à deux panneaux, probabilités par
                         billet et projection sur le plan factoriel, et
                         enregistre l'image (défaut : graphique_detection.png)

Mode visuel des résultats finaux
--------------------------------
L'option --graphique montre les résultats autrement que dans la console. À
gauche, les résultats finaux : la probabilité d'être vrai de chaque billet, en
barres colorées selon le verdict, avec la ligne du seuil de décision (au-delà
de 30 billets, les barres deviennent un histogramme des probabilités). À
droite, le contexte : le plan factoriel du notebook, reconstruit depuis
billets.csv (imputation, standardisation, analyse en composantes principales),
avec les billets analysés en étoiles colorées selon le verdict. Quand un écran
est disponible, la fenêtre s'ouvre à la fin de l'exécution ; dans tous les cas
l'image est enregistrée en PNG à côté des résultats. L'option demande
matplotlib et billets.csv, livrés avec l'application ; si l'un des deux manque,
le script le signale et rend quand même ses verdicts.

Valeurs manquantes
------------------
Le jeu d'entraînement comportait 37 valeurs manquantes sur margin_low. Le cas
peut donc se reproduire en production. Par défaut, le script impute ces valeurs
avec le même modèle de régression linéaire que le notebook, dont les
coefficients sont embarqués dans le fichier joblib, et signale chaque ligne
concernée. L'option --refuser-manquants rétablit un comportement strict.

Environnement
-------------
Modèle entraîné et sérialisé sous scikit-learn 1.6.1. Le script fonctionne avec
des versions ultérieures, scikit-learn émettant alors un avertissement de
compatibilité sans conséquence sur les prédictions. Dépendances : pandas, numpy,
scikit-learn, joblib, et matplotlib pour la seule option --graphique.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FEATURES = ["diagonal", "height_left", "height_right",
            "margin_low", "margin_up", "length"]

LIBELLES = {
    "diagonal": "diagonale du billet (mm)",
    "height_left": "hauteur mesurée à gauche (mm)",
    "height_right": "hauteur mesurée à droite (mm)",
    "margin_low": "marge inférieure (mm)",
    "margin_up": "marge supérieure (mm)",
    "length": "longueur du billet (mm)",
}

DEFAULT_MODEL_NAME = "modele_detection_billets.joblib"
DEFAULT_TRAIN_NAME = "billets.csv"
# Dossier du dépôt : le modèle et le jeu d'entraînement y sont cherchés si cette
# application de repli est lancée depuis un dossier voisin plutôt que depuis le dépôt.
DEPOT_NAME = "Detection_faux_billets_Lippens_Charles"
DEFAULT_GRAPHIQUE_NAME = "graphique_detection.png"
SEUIL_DEFAUT = 0.5

# Bornes de vraisemblance issues du jeu d'entraînement (min et max observés,
# élargis d'une marge de sécurité). Servent uniquement à alerter l'opérateur.
BORNES_PLAUSIBLES = {
    "diagonal": (170.0, 174.0),
    "height_left": (102.0, 106.0),
    "height_right": (102.0, 106.0),
    "margin_low": (2.0, 8.0),
    "margin_up": (1.5, 5.0),
    "length": (108.0, 116.0),
}


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
def locate_model(explicit: str | None = None) -> Path:
    """Cherche le fichier modèle et lève une erreur explicite s'il est introuvable."""
    if explicit:
        chemin = Path(explicit)
        if not chemin.exists():
            raise FileNotFoundError(f"Modèle introuvable : {chemin}")
        return chemin

    ici = Path(__file__).resolve().parent
    candidats = [ici / DEFAULT_MODEL_NAME,
                 Path.cwd() / DEFAULT_MODEL_NAME,
                 ici.parent / DEFAULT_MODEL_NAME,
                 ici.parent / DEPOT_NAME / DEFAULT_MODEL_NAME]
    for candidat in candidats:
        if candidat.exists():
            return candidat
    raise FileNotFoundError(
        f"Modèle introuvable. Placez '{DEFAULT_MODEL_NAME}' à côté du script, "
        f"gardez le dossier '{DEPOT_NAME}' voisin, ou indiquez son chemin avec --model."
    )


def load_artefact(model_path: Path) -> dict:
    """
    Charge le fichier joblib et renvoie toujours un dictionnaire normalisé.

    Le notebook d'août 2026 sauvegarde un dictionnaire contenant le pipeline,
    le modèle d'imputation et des métadonnées. Les versions antérieures
    sauvegardaient directement le pipeline : ce cas reste pris en charge.
    """
    objet = joblib.load(model_path)

    if isinstance(objet, dict) and "pipeline" in objet:
        artefact = dict(objet)
    else:
        artefact = {"pipeline": objet, "features": FEATURES,
                    "imputation": None, "seuil": SEUIL_DEFAUT, "metadata": {}}

    artefact.setdefault("features", FEATURES)
    artefact.setdefault("imputation", None)
    artefact.setdefault("seuil", SEUIL_DEFAUT)
    artefact.setdefault("metadata", {})

    print(f"[OK] Modèle chargé depuis : {model_path}")
    meta = artefact["metadata"]
    if meta:
        details = []
        if meta.get("sklearn_version"):
            details.append(f"entraîné sous scikit-learn {meta['sklearn_version']}")
        if meta.get("n_entrainement"):
            details.append(f"{meta['n_entrainement']} billets d'entraînement")
        if details:
            print("     " + ", ".join(details) + ".")
    return artefact


# ---------------------------------------------------------------------------
# Lecture et contrôle du fichier d'entrée
# ---------------------------------------------------------------------------
def read_csv_smart(csv_path: Path) -> pd.DataFrame:
    """Lit un CSV en détectant le séparateur, le point-virgule et la virgule."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig") as flux:
        entete = flux.readline()

    separateur = ";" if entete.count(";") > entete.count(",") else ","
    donnees = pd.read_csv(csv_path, sep=separateur, encoding="utf-8-sig")
    donnees.columns = [str(c).strip() for c in donnees.columns]

    if donnees.empty:
        raise ValueError(f"Le fichier {csv_path} ne contient aucune ligne de données.")
    return donnees


def imputer_manquants(donnees: pd.DataFrame, imputation: dict | None) -> tuple[pd.DataFrame, list]:
    """
    Complète les valeurs manquantes de margin_low avec le modèle de régression
    linéaire embarqué dans le fichier joblib. Renvoie le tableau complété et la
    liste des identifiants concernés.
    """
    manquants = donnees[FEATURES].isna()
    if not manquants.any().any():
        return donnees, []

    colonnes_touchees = [c for c in FEATURES if manquants[c].any()]

    if imputation is None:
        raise ValueError(
            "Le fichier contient des valeurs manquantes sur "
            f"{', '.join(colonnes_touchees)}, et le modèle chargé n'embarque pas "
            "de règle d'imputation. Complétez le fichier ou réexportez le modèle "
            "depuis le notebook."
        )

    cible = imputation["cible"]
    predicteurs = imputation["predicteurs"]
    coefficients = imputation["coefficients"]

    hors_perimetre = [c for c in colonnes_touchees if c != cible]
    if hors_perimetre:
        raise ValueError(
            "Valeurs manquantes sur "
            f"{', '.join(hors_perimetre)} : seule la colonne '{cible}' peut être "
            "imputée automatiquement. Complétez ces colonnes dans le fichier."
        )

    if donnees[predicteurs].isna().any().any():
        raise ValueError(
            f"Impossible d'imputer '{cible}' : certaines des variables "
            f"explicatives ({', '.join(predicteurs)}) sont elles-mêmes manquantes."
        )

    donnees = donnees.copy()
    lignes = manquants[cible]
    estimation = np.full(int(lignes.sum()), float(coefficients.get("const", 0.0)))
    for variable in predicteurs:
        estimation += float(coefficients[variable]) * donnees.loc[lignes, variable].to_numpy()

    donnees.loc[lignes, cible] = estimation
    identifiants = donnees.loc[lignes, "id"].tolist() if "id" in donnees.columns \
        else [f"ligne {i + 1}" for i in np.flatnonzero(lignes.to_numpy())]
    return donnees, identifiants


def validate_dataframe(donnees: pd.DataFrame) -> pd.DataFrame:
    """Vérifie la présence des six colonnes attendues et ajoute un identifiant si besoin."""
    absentes = [c for c in FEATURES if c not in donnees.columns]
    if absentes:
        raise ValueError(
            f"Colonnes manquantes dans le fichier d'entrée : {absentes}. "
            f"Colonnes attendues : {FEATURES}."
        )

    donnees = donnees.copy()
    for colonne in FEATURES:
        donnees[colonne] = pd.to_numeric(donnees[colonne], errors="coerce")

    if "id" not in donnees.columns:
        donnees.insert(0, "id", [f"billet_{i + 1}" for i in range(len(donnees))])
    return donnees


def signaler_valeurs_douteuses(donnees: pd.DataFrame) -> list:
    """Repère les dimensions hors des plages physiquement plausibles."""
    alertes = []
    for _, ligne in donnees.iterrows():
        for variable, (mini, maxi) in BORNES_PLAUSIBLES.items():
            valeur = ligne[variable]
            if pd.notna(valeur) and not (mini <= valeur <= maxi):
                alertes.append(f"{ligne['id']} : {variable} = {valeur} "
                               f"(hors de la plage attendue {mini} à {maxi})")
    return alertes


# ---------------------------------------------------------------------------
# Prédiction et restitution
# ---------------------------------------------------------------------------
def predict_dataframe(pipeline, donnees: pd.DataFrame, seuil: float) -> pd.DataFrame:
    """Applique le pipeline et renvoie un tableau enrichi des prédictions."""
    matrice = donnees[FEATURES]
    probabilites = pipeline.predict_proba(matrice)[:, 1]
    predictions = (probabilites >= seuil).astype(int)

    resultat = donnees[["id"] + FEATURES].copy()
    resultat["prediction"] = predictions
    resultat["probabilite_vrai"] = probabilites.round(4)
    resultat["verdict"] = np.where(predictions == 1, "VRAI BILLET", "FAUX BILLET")
    return resultat


def pretty_print(resultat: pd.DataFrame, seuil: float) -> None:
    """Affichage console formaté."""
    largeur_id = max(8, int(resultat["id"].astype(str).map(len).max()))
    largeur_totale = largeur_id + 44

    print()
    print("Résultats de la détection")
    if abs(seuil - SEUIL_DEFAUT) > 1e-9:
        print(f"Seuil de décision : {seuil:.2f} (valeur par défaut : {SEUIL_DEFAUT:.2f})")
    print("-" * largeur_totale)
    print(f"{'id':<{largeur_id}}  {'prediction':>10}  {'proba_vrai':>11}  verdict")
    print("-" * largeur_totale)
    for _, ligne in resultat.iterrows():
        print(f"{str(ligne['id']):<{largeur_id}}  "
              f"{int(ligne['prediction']):>10}  "
              f"{ligne['probabilite_vrai']:>11.4f}  "
              f"{ligne['verdict']}")
    print("-" * largeur_totale)

    total = len(resultat)
    vrais = int((resultat["prediction"] == 1).sum())
    print(f"Total : {total} billet(s), {vrais} vrai(s) et {total - vrais} faux")

    incertains = resultat[(resultat["probabilite_vrai"] > 0.20)
                          & (resultat["probabilite_vrai"] < 0.80)]
    if not incertains.empty:
        print()
        print("Verdicts peu tranchés, une vérification manuelle est conseillée :")
        for _, ligne in incertains.iterrows():
            print(f"  {ligne['id']} : probabilité {ligne['probabilite_vrai']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Visualisation : les billets analysés sur le plan factoriel de l'entraînement
# ---------------------------------------------------------------------------
def locate_train() -> Path:
    """Cherche billets.csv, le jeu d'entraînement livré avec l'application."""
    ici = Path(__file__).resolve().parent
    for candidat in (ici / DEFAULT_TRAIN_NAME,
                     Path.cwd() / DEFAULT_TRAIN_NAME,
                     ici.parent / DEFAULT_TRAIN_NAME,
                     ici.parent / DEPOT_NAME / DEFAULT_TRAIN_NAME):
        if candidat.exists():
            return candidat
    raise FileNotFoundError(
        f"Jeu d'entraînement '{DEFAULT_TRAIN_NAME}' introuvable : placez-le à "
        "côté du script pour générer le graphique."
    )


def generer_graphique(artefact: dict, resultat: pd.DataFrame, chemin_png: Path,
                      seuil: float) -> None:
    """
    Mode visuel des résultats finaux, sur deux panneaux : à gauche la
    probabilité de chaque billet en barres colorées selon le verdict, avec la
    ligne du seuil (histogramme au-delà de 30 billets) ; à droite le plan
    factoriel du notebook, reconstruit sur les 1 500 billets d'entraînement,
    avec les billets analysés en étoiles. La fenêtre s'ouvre quand un écran est
    disponible, et l'image est enregistrée en PNG dans tous les cas.
    """
    import os
    try:
        import matplotlib
        # Sans écran, on bascule sur le moteur de rendu hors fenêtre.
        interactif = (sys.platform in ("win32", "darwin")
                      or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
        if not interactif:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("matplotlib n'est pas installé (pip install matplotlib).") from exc

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train = read_csv_smart(locate_train())
    absentes = [c for c in FEATURES + ["is_genuine"] if c not in train.columns]
    if absentes:
        raise ValueError(f"Colonnes manquantes dans billets.csv : {absentes}.")
    for colonne in FEATURES:
        train[colonne] = pd.to_numeric(train[colonne], errors="coerce")
    if "id" not in train.columns:
        train.insert(0, "id", [f"train_{i + 1}" for i in range(len(train))])
    train, _ = imputer_manquants(train, artefact["imputation"])

    vrais = train["is_genuine"].astype(str).str.strip().str.lower().isin(
        ("true", "1", "vrai", "genuine"))

    scaler = StandardScaler().fit(train[FEATURES])
    acp = PCA(n_components=2).fit(scaler.transform(train[FEATURES]))
    plan_train = acp.transform(scaler.transform(train[FEATURES]))
    plan_nouveaux = acp.transform(scaler.transform(resultat[FEATURES]))
    part1, part2 = acp.explained_variance_ratio_[:2] * 100

    predits_vrais = resultat["prediction"].to_numpy() == 1
    probas = resultat["probabilite_vrai"].to_numpy()
    total, n_vrais = len(resultat), int(predits_vrais.sum())

    fig, (axe_res, axe_plan) = plt.subplots(
        1, 2, figsize=(13.5, 6.8), gridspec_kw={"width_ratios": [1.0, 1.35]})
    fig.canvas.manager.set_window_title("Détection de faux billets, résultats")

    # Panneau de gauche : les résultats finaux, billet par billet.
    couleurs = np.where(predits_vrais, "#2e7d32", "#c62828")
    if total <= 30:
        positions = np.arange(total)[::-1]
        axe_res.barh(positions, probas, color=couleurs, edgecolor="black", linewidth=0.4)
        axe_res.set_yticks(positions)
        axe_res.set_yticklabels(resultat["id"].astype(str))
        for pos, proba, est_vrai in zip(positions, probas, predits_vrais):
            couleur = "#1b5e20" if est_vrai else "#b71c1c"
            axe_res.annotate(f"{proba:.4f}".replace(".", ",") + f"  {'VRAI' if est_vrai else 'FAUX'}",
                             (min(proba + 0.02, 0.72), pos), va="center", fontsize=8,
                             color=couleur, fontweight="bold",
                             bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                                   "edgecolor": couleur, "linewidth": 0.5, "alpha": 0.9})
        axe_res.set_xlim(0, 1)
        axe_res.set_xlabel("Probabilité d'être un vrai billet")
    else:
        axe_res.hist([probas[predits_vrais], probas[~predits_vrais]], bins=20,
                     stacked=True, color=["#2e7d32", "#c62828"],
                     label=["Verdicts vrais", "Verdicts faux"])
        axe_res.set_xlabel("Probabilité d'être un vrai billet")
        axe_res.set_ylabel("Nombre de billets")
        axe_res.legend(fontsize=8)
    # La zone des verdicts peu tranchés, celle que la console signale aussi.
    axe_res.axvspan(0.20, 0.80, color="#f9a825", alpha=0.10, zorder=0)
    axe_res.annotate("zone d'incertitude (0,20 à 0,80)",
                     (0.50, 0.015), xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=7, color="#8d6e00")
    axe_res.axvline(seuil, color="#37474f", linestyle="--", linewidth=1.2)
    axe_res.annotate("seuil = " + f"{seuil:.2f}".replace(".", ","), (seuil, 0.97),
                     xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=8,
                     bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                           "edgecolor": "#37474f", "linewidth": 0.6})
    axe_res.set_title(f"Résultats finaux : {total} billet(s), {n_vrais} vrai(s), {total - n_vrais} faux")
    axe_res.grid(True, axis="x", linewidth=0.3, alpha=0.5)

    # Panneau de droite : le contexte, les billets posés sur le plan factoriel.
    axe_plan.scatter(plan_train[vrais.to_numpy(), 0], plan_train[vrais.to_numpy(), 1],
                     s=14, c="#2e7d32", alpha=0.25,
                     label="Vrais billets, entraînement")
    axe_plan.scatter(plan_train[~vrais.to_numpy(), 0], plan_train[~vrais.to_numpy(), 1],
                     s=14, c="#c62828", alpha=0.25,
                     label="Faux billets, entraînement")
    for masque, couleur, libelle in (
            (predits_vrais, "#1b5e20", "Billet analysé, verdict vrai"),
            (~predits_vrais, "#b71c1c", "Billet analysé, verdict faux")):
        if masque.any():
            axe_plan.scatter(plan_nouveaux[masque, 0], plan_nouveaux[masque, 1],
                             s=340, marker="*", c=couleur, edgecolors="black",
                             linewidths=0.8, zorder=5, label=libelle)
    if total <= 30:
        for (x, y), (_, ligne) in zip(plan_nouveaux, resultat.iterrows()):
            axe_plan.annotate(f"{ligne['id']}\np = " + f"{ligne['probabilite_vrai']:.2f}".replace(".", ","),
                              (x, y), textcoords="offset points", xytext=(9, 7),
                              fontsize=8, zorder=6)
    axe_plan.set_title("Billets analysés sur le plan factoriel de l'entraînement")
    axe_plan.set_xlabel("F1 (" + f"{part1:.1f}".replace(".", ",") + " % de la variance)")
    axe_plan.set_ylabel("F2 (" + f"{part2:.1f}".replace(".", ",") + " % de la variance)")
    axe_plan.grid(True, linewidth=0.3, alpha=0.5)
    axe_plan.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(chemin_png, dpi=150)
    print(f"[OK] Graphique enregistré vers : {chemin_png}")
    if interactif:
        print("[OK] Ouverture de la fenêtre de résultats, fermez-la pour terminer.")
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Saisie d'un billet unique
# ---------------------------------------------------------------------------
def build_single_billet(args) -> pd.DataFrame:
    """Construit un tableau d'une ligne à partir des arguments de la ligne de commande."""
    valeurs = {variable: getattr(args, variable) for variable in FEATURES}
    absentes = [variable for variable, valeur in valeurs.items() if valeur is None]
    if absentes:
        raise ValueError(
            "Les six dimensions sont obligatoires en mode arguments. "
            f"Manquantes : {absentes}. Lancez le script sans argument pour "
            "utiliser le mode interactif."
        )
    return pd.DataFrame([{"id": "billet_saisi", **valeurs}])


def saisie_interactive() -> pd.DataFrame:
    """Demande les six dimensions à l'opérateur, une par une."""
    print()
    print("Mode interactif : saisie des dimensions d'un billet")
    print("Appuyez sur Entrée sans rien saisir pour annuler.")
    print("-" * 55)

    valeurs = {}
    for variable in FEATURES:
        while True:
            brut = input(f"  {variable:<13} ({LIBELLES[variable]}) : ").strip()
            if brut == "":
                raise ValueError("Saisie annulée par l'opérateur.")
            try:
                valeur = float(brut.replace(",", "."))
            except ValueError:
                print("    Valeur non numérique, recommencez.")
                continue
            mini, maxi = BORNES_PLAUSIBLES[variable]
            if not (mini <= valeur <= maxi):
                reponse = input(f"    {valeur} sort de la plage attendue "
                                f"({mini} à {maxi}). Confirmer ? [o/N] ").strip().lower()
                if reponse not in ("o", "oui", "y", "yes"):
                    continue
            valeurs[variable] = valeur
            break
    return pd.DataFrame([{"id": "billet_saisi", **valeurs}])


# ---------------------------------------------------------------------------
# Interface en ligne de commande
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    analyseur = argparse.ArgumentParser(
        prog="Lippens_Charles_2_Application_terminal_082026",
        description="Détection automatique de faux billets, application finale ONCFM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python Lippens_Charles_2_Application_terminal_082026.py "
            "--input billets_production.csv\n"
            "  python Lippens_Charles_2_Application_terminal_082026.py "
            "--input billets.csv --output resultats.csv\n"
            "  python Lippens_Charles_2_Application_terminal_082026.py "
            "--diagonal 171.81 --height_left 104.86 --height_right 104.95 "
            "--margin_low 4.52 --margin_up 2.89 --length 112.83\n"
            "  python Lippens_Charles_2_Application_terminal_082026.py    (mode interactif)\n"
        ),
    )
    analyseur.add_argument("--input", "-i", help="Chemin vers un fichier CSV de billets.")
    analyseur.add_argument("--output", "-o", help="Chemin du fichier CSV de sortie.")
    analyseur.add_argument("--model", "-m", help="Chemin explicite vers le modèle .joblib.")
    analyseur.add_argument("--seuil", "-s", type=float, default=None,
                           help="Seuil de décision entre 0 et 1 (0,5 par défaut).")
    analyseur.add_argument("--refuser-manquants", action="store_true",
                           dest="refuser_manquants",
                           help="Refuse le fichier au lieu d'imputer les valeurs manquantes.")
    analyseur.add_argument("--graphique", "-g", nargs="?", const=DEFAULT_GRAPHIQUE_NAME,
                           default=None, metavar="IMG.png",
                           help="Mode visuel des résultats finaux : ouvre une fenêtre à deux "
                                "panneaux, probabilités par billet et projection factorielle, "
                                f"et enregistre l'image (défaut : {DEFAULT_GRAPHIQUE_NAME}).")

    groupe = analyseur.add_argument_group("Saisie d'un billet unique")
    for variable in FEATURES:
        groupe.add_argument(f"--{variable}", type=float, default=None,
                            help=f"{LIBELLES[variable].capitalize()}.")
    return analyseur.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        artefact = load_artefact(locate_model(args.model))
        pipeline = artefact["pipeline"]

        seuil = args.seuil if args.seuil is not None else float(artefact["seuil"])
        if not 0.0 < seuil < 1.0:
            raise ValueError(f"Le seuil doit être strictement compris entre 0 et 1 (reçu : {seuil}).")

        dimensions_fournies = any(getattr(args, v) is not None for v in FEATURES)

        if args.input:
            donnees = validate_dataframe(read_csv_smart(Path(args.input)))
            if args.refuser_manquants:
                manquants = donnees[FEATURES].isna().sum()
                if int(manquants.sum()) > 0:
                    detail = ", ".join(f"{c} ({int(n)})" for c, n in manquants.items() if n > 0)
                    raise ValueError(f"Valeurs manquantes détectées : {detail}.")
            else:
                donnees, imputes = imputer_manquants(donnees, artefact["imputation"])
                if imputes:
                    print(f"[INFO] {len(imputes)} valeur(s) manquante(s) imputée(s) par "
                          "régression linéaire (même modèle que le notebook) : "
                          + ", ".join(str(i) for i in imputes[:10])
                          + (" ..." if len(imputes) > 10 else ""))
        elif dimensions_fournies:
            donnees = validate_dataframe(build_single_billet(args))
        else:
            donnees = validate_dataframe(saisie_interactive())

        restant = int(donnees[FEATURES].isna().sum().sum())
        if restant:
            raise ValueError(f"{restant} valeur(s) non numérique(s) ou manquante(s) "
                             "subsistent après contrôle. Vérifiez le fichier d'entrée.")

        for alerte in signaler_valeurs_douteuses(donnees):
            print(f"[ALERTE] {alerte}")

        resultat = predict_dataframe(pipeline, donnees, seuil)
        pretty_print(resultat, seuil)

        if args.output:
            colonnes = ["id", "prediction", "probabilite_vrai", "verdict"] + FEATURES
            resultat[colonnes].to_csv(args.output, index=False)
            print(f"[OK] Résultats exportés vers : {args.output}")

        if args.graphique:
            try:
                generer_graphique(artefact, resultat, Path(args.graphique), seuil)
            except Exception as erreur:
                # Le graphique est un plus : son échec ne doit jamais invalider
                # les verdicts déjà rendus.
                print(f"[ALERTE] Graphique non généré : {erreur}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n[INTERROMPU] Aucune prédiction effectuée.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, KeyError) as erreur:
        print(f"[ERREUR] {erreur}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
