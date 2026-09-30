# -*- coding: utf-8 -*-
"""
Détection de faux billets : l'application, version tableau de bord (Streamlit).
Projet 12, ONCFM, Charles Lippens, août 2026. Livrable 2 du dépôt, version 5.

Tout se fait depuis le navigateur, sans ligne de commande : c'est la forme
demandée par le mentor en séance de mentorat. L'application terminal d'origine
(Lippens_Charles_2_Application_terminal_082026.py) est livrée dans le même
dépôt, à côté de ce fichier, comme repli sans navigateur, et rend les mêmes
verdicts.

Ce que propose l'interface :
  1. charger un fichier CSV de billets au format de billets_production.csv,
     directement depuis la page (ou utiliser le fichier d'exemple fourni) ;
  2. laisser l'application sélectionner automatiquement le meilleur modèle,
     la régression logistique du fichier joblib officiel ;
  3. comparer les quatre modèles du projet sur le protocole du notebook
     et sur le fichier chargé ;
  4. choisir soi-même le modèle qui rend les verdicts, si on le souhaite.

Lancement, depuis ce dossier, après une installation unique de Streamlit
(pip install streamlit) :
    streamlit run Lippens_Charles_2_Application_082026.py

Version 2 : le seuil de décision affiche explicitement sa valeur par défaut,
0,50, celle du livrable, et rappelle le levier documenté à 0,69.
Version 3 : plus de texte parasite au-dessus du premier graphique (une
expression nue que Streamlit affichait), les deux graphiques sont alignés à la
même hauteur, et les dimensions hors des plages plausibles sont signalées,
comme dans l'application terminal.
Version 4 : version livrée dans le dépôt sous le nom du livrable 2 ; libellés
mis à jour, fichier de résultats nommé resultats.csv.
Version 5 : l'application terminal rejoint le dépôt, à côté de ce fichier, le
support de présentation y est livré en PDF, et le random forest de comparaison
reprend tous les réglages du notebook, min_samples_leaf=2 compris.

Dépendances : streamlit, pandas, numpy, scikit-learn, joblib, matplotlib.
Le fichier modele_detection_billets.joblib et billets.csv sont cherchés dans
le dossier du script, puis dans le dossier courant, puis dans un dossier
voisin nommé Detection_faux_billets_Lippens_Charles.

Les verdicts rendus en mode automatique, au seuil de 0,5, sont identiques à
ceux du notebook et de l'application terminal : mêmes probabilités, mêmes
conclusions, même fichier resultats.csv.
"""

from pathlib import Path
import io

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constantes, reprises du notebook et de l'application terminale
# ---------------------------------------------------------------------------
FEATURES = ["diagonal", "height_left", "height_right",
            "margin_low", "margin_up", "length"]
MODEL_NAME = "modele_detection_billets.joblib"
TRAIN_NAME = "billets.csv"
EXEMPLE_NAME = "billets_production.csv"
RANDOM_STATE = 42
SEUIL_DEFAUT = 0.5

NOM_AUTO = "Meilleur modèle, sélection automatique (régression logistique)"
NOM_RL = "Régression logistique"
NOM_KNN = "K plus proches voisins (KNN, k = 13)"
NOM_RF = "Random forest (300 arbres)"
NOM_KM = "K-Means par les centroïdes"

BORNES_PLAUSIBLES = {
    "diagonal": (170.0, 174.0), "height_left": (102.0, 106.0),
    "height_right": (102.0, 106.0), "margin_low": (2.0, 8.0),
    "margin_up": (1.5, 5.0), "length": (108.0, 116.0),
}

VERT = "#2ECC71"
ROUGE = "#E74C3C"
BLEU = "#1F4E79"
AMBRE = "#F5B041"


def fr(x, dec=4):
    """Écrit un nombre à la française, virgule décimale."""
    return f"{x:.{dec}f}".replace(".", ",")


def locate(name):
    """Cherche un fichier dans le dossier du script, le dossier courant et
    le dossier des livrables voisin."""
    ici = Path(__file__).resolve().parent
    candidats = [ici / name, Path.cwd() / name,
                 ici.parent / "Detection_faux_billets_Lippens_Charles" / name]
    for c in candidats:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Fichier introuvable : {name}. Placez-le à côté du script ou dans "
        "le dossier Detection_faux_billets_Lippens_Charles voisin.")


# ---------------------------------------------------------------------------
# Chargement du modèle officiel et des données d'entraînement
# ---------------------------------------------------------------------------
def charger_artefact():
    """Recharge le fichier joblib officiel et le normalise en dictionnaire."""
    objet = joblib.load(locate(MODEL_NAME))
    if isinstance(objet, dict) and "pipeline" in objet:
        artefact = dict(objet)
    else:
        artefact = {"pipeline": objet, "imputation": None}
    artefact.setdefault("imputation", None)
    artefact.setdefault("seuil", SEUIL_DEFAUT)
    return artefact


def charger_entrainement(artefact):
    """Recharge billets.csv, impute la marge basse manquante avec les
    coefficients du joblib, et renvoie X (1 500 x 6) et y."""
    train = lire_csv(locate(TRAIN_NAME))
    for c in FEATURES:
        train[c] = pd.to_numeric(train[c], errors="coerce")
    train, _ = imputer_manquants(train, artefact["imputation"])
    X = train[FEATURES].to_numpy(float)
    y = train["is_genuine"].astype(int).to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Lecture, contrôle et imputation du fichier chargé
# ---------------------------------------------------------------------------
def lire_csv(source):
    """Lit un CSV en laissant pandas détecter le séparateur, virgule ou
    point-virgule. Accepte un chemin ou un contenu téléversé."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    donnees = pd.read_csv(source, sep=None, engine="python")
    donnees.columns = [c.strip() for c in donnees.columns]
    return donnees


def valider(donnees):
    """Vérifie les colonnes attendues et pose un identifiant si besoin."""
    absentes = [c for c in FEATURES if c not in donnees.columns]
    if absentes:
        raise ValueError(
            "Colonnes manquantes dans le fichier : " + ", ".join(absentes)
            + ". Les six mesures sont attendues : " + ", ".join(FEATURES) + ".")
    donnees = donnees.copy()
    if "id" not in donnees.columns:
        donnees.insert(0, "id", [f"billet_{i + 1}" for i in range(len(donnees))])
    for c in FEATURES:
        donnees[c] = pd.to_numeric(donnees[c], errors="coerce")
    return donnees


def imputer_manquants(donnees, imputation):
    """Complète margin_low avec la régression embarquée dans le joblib,
    comme le fait l'application terminal. Renvoie le tableau et les identifiants
    des lignes complétées."""
    manquants = donnees[FEATURES].isna()
    if not manquants.any().any():
        return donnees, []
    touchees = [c for c in FEATURES if manquants[c].any()]
    if imputation is None:
        raise ValueError("Valeurs manquantes sur " + ", ".join(touchees)
                         + " et aucune règle d'imputation dans le modèle.")
    cible = imputation["cible"]
    hors = [c for c in touchees if c != cible]
    if hors:
        raise ValueError("Seule la colonne " + cible + " peut être imputée ; "
                         "complétez : " + ", ".join(hors) + ".")
    donnees = donnees.copy()
    lignes = manquants[cible]
    coefs = imputation["coefficients"]
    estimation = np.full(int(lignes.sum()), float(coefs.get("const", 0.0)))
    for v in imputation["predicteurs"]:
        estimation += float(coefs[v]) * donnees.loc[lignes, v].to_numpy(float)
    donnees.loc[lignes, cible] = estimation
    if "id" in donnees.columns:
        ids = donnees.loc[lignes, "id"].tolist()
    else:
        ids = [f"ligne {i + 1}" for i in np.flatnonzero(lignes.to_numpy())]
    return donnees, ids


def signaler_hors_plage(donnees):
    """Liste les billets dont une mesure sort des plages plausibles observées
    à l'entraînement ; simple alerte, jamais bloquante."""
    alertes = []
    for c, (bas, haut) in BORNES_PLAUSIBLES.items():
        hors = donnees[(donnees[c] < bas) | (donnees[c] > haut)]
        for _, ligne in hors.iterrows():
            alertes.append(f"{ligne['id']} : {c} = {fr(float(ligne[c]), 2)} mm, "
                           f"hors de la plage {fr(bas, 1)} à {fr(haut, 1)} mm")
    return alertes


# ---------------------------------------------------------------------------
# Le K-Means exploité comme classifieur par ses centroïdes (notebook 8.4)
# ---------------------------------------------------------------------------
class KMeansCentroides:
    """Reproduit le quatrième modèle du notebook : un K-Means à deux groupes
    sur données standardisées, chaque groupe recevant la classe majoritaire,
    la prédiction suivant le centroïde le plus proche."""

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.km = KMeans(n_clusters=2, n_init=10,
                         random_state=RANDOM_STATE).fit(Xs)
        self.mapping = {}
        for c in [0, 1]:
            self.mapping[c] = int(pd.Series(y[self.km.labels_ == c]).mode()[0])
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        dists = np.linalg.norm(Xs[:, None, :] - self.km.cluster_centers_[None, :, :], axis=2)
        inv = 1.0 / (dists + 1e-9)
        soft = inv / inv.sum(axis=1, keepdims=True)
        colonnes_vrai = [c for c in [0, 1] if self.mapping[c] == 1]
        p1 = soft[:, colonnes_vrai].sum(axis=1)
        return np.column_stack([1 - p1, p1])


# ---------------------------------------------------------------------------
# Entraînements mis en cache : une seule fois par session du navigateur
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Préparation des modèles de comparaison...")
def preparer_modeles():
    """Prépare tout ce qui demande un entraînement.

    Deux familles distinctes, fidèles au notebook :
      - pour le tableau des métriques, les modèles sont entraînés sur les
        1 200 billets d'entraînement et évalués sur les 300 billets de test,
        découpage stratifié à graine fixée, exactement le protocole du
        notebook, ce qui redonne ses chiffres ;
      - pour rendre des verdicts sur un fichier chargé, chaque modèle est
        réentraîné sur les 1 500 billets, comme le pipeline final officiel,
        la régression logistique restant celle du joblib.
    """
    artefact = charger_artefact()
    X, y = charger_entrainement(artefact)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    protocole = {
        NOM_RL: make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000)).fit(X_tr, y_tr),
        NOM_KNN: make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=13)).fit(X_tr, y_tr),
        NOM_RF: RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                       random_state=RANDOM_STATE).fit(X_tr, y_tr),
        NOM_KM: KMeansCentroides().fit(X_tr, y_tr),
    }

    lignes = []
    for nom, modele in protocole.items():
        proba = modele.predict_proba(X_te)[:, 1]
        pred = (proba >= SEUIL_DEFAUT).astype(int)
        faux_passes = int(((y_te == 0) & (pred == 1)).sum())
        vrais_rejetes = int(((y_te == 1) & (pred == 0)).sum())
        lignes.append({
            "Modèle": nom,
            "Exactitude": fr(accuracy_score(y_te, pred)),
            "Score F1": fr(f1_score(y_te, pred)),
            "Faux interceptés (sur 100)": int(recall_score(y_te, pred, pos_label=0) * 100),
            "Faux laissés passer": faux_passes,
            "Vrais rejetés à tort": vrais_rejetes,
        })
    tableau_metriques = pd.DataFrame(lignes)

    complets = {
        NOM_KNN: make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=13)).fit(X, y),
        NOM_RF: RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                       random_state=RANDOM_STATE).fit(X, y),
        NOM_KM: KMeansCentroides().fit(X, y),
    }

    scaler_plan = StandardScaler().fit(X)
    pca_plan = PCA(n_components=2).fit(scaler_plan.transform(X))
    fond = {"points": pca_plan.transform(scaler_plan.transform(X)), "y": y,
            "variance": pca_plan.explained_variance_ratio_ * 100,
            "scaler": scaler_plan, "pca": pca_plan}

    return artefact, complets, tableau_metriques, fond


def predicteur(nom, artefact, complets):
    """Renvoie l'objet qui sait produire des probabilités pour ce nom de
    modèle. Le meilleur modèle et la régression logistique renvoient le
    pipeline officiel du joblib."""
    if nom in (NOM_AUTO, NOM_RL):
        return artefact["pipeline"]
    return complets[nom]


def predire(donnees, modele, seuil):
    """Applique le modèle et construit le tableau des résultats, dans le
    même ordre de colonnes que le fichier resultats.csv du notebook."""
    proba = modele.predict_proba(donnees[FEATURES].to_numpy(float))[:, 1]
    pred = (proba >= seuil).astype(int)
    resultat = donnees[["id"]].copy()
    resultat["prediction"] = pred
    resultat["probabilite_vrai"] = np.round(proba, 4)
    resultat["verdict"] = np.where(pred == 1, "VRAI BILLET", "FAUX BILLET")
    for c in FEATURES:
        resultat[c] = donnees[c].to_numpy()
    return resultat


# ---------------------------------------------------------------------------
# Les deux graphiques, dans le style du mode visuel de l'application terminal
# ---------------------------------------------------------------------------
FIGSIZE = (7.5, 4.8)


def figure_barres(resultat, seuil):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    n = len(resultat)
    if n <= 30:
        couleurs = [VERT if v == "VRAI BILLET" else ROUGE for v in resultat["verdict"]]
        ax.bar(range(n), resultat["probabilite_vrai"], color=couleurs)
        ax.scatter(range(n), resultat["probabilite_vrai"], c=couleurs, s=42,
                   zorder=4, edgecolor="white", linewidths=0.8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(resultat["id"], rotation=0 if n <= 8 else 45,
                           ha="center" if n <= 8 else "right", fontsize=8)
        ax.set_xlabel("Billet analysé")
    else:
        vrais = resultat.loc[resultat["prediction"] == 1, "probabilite_vrai"]
        faux = resultat.loc[resultat["prediction"] == 0, "probabilite_vrai"]
        ax.hist([faux, vrais], bins=20, stacked=True, color=[ROUGE, VERT],
                label=["Faux billets", "Vrais billets"])
        ax.set_xlabel("Probabilité d'être un vrai billet")
        ax.set_ylabel("Nombre de billets")
        ax.legend(fontsize=8)
    if n <= 30:
        ax.axhspan(0.20, 0.80, color=AMBRE, alpha=0.15)
        ax.axhline(seuil, color=BLEU, linestyle="--", linewidth=1.2)
        ax.text(n - 0.5, seuil + 0.02, "seuil " + fr(seuil, 2), color=BLEU,
                ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probabilité d'être un vrai billet")
    ax.set_title("Probabilités et verdicts par billet")
    fig.tight_layout()
    return fig


def figure_plan(resultat, fond, artefact):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pts, y = fond["points"], fond["y"]
    ax.scatter(pts[y == 1, 0], pts[y == 1, 1], s=10, alpha=0.18, color=VERT,
               label="Vrais billets d'entraînement")
    ax.scatter(pts[y == 0, 0], pts[y == 0, 1], s=10, alpha=0.18, color=ROUGE,
               label="Faux billets d'entraînement")
    X_new = resultat[FEATURES].to_numpy(float)
    proj = fond["pca"].transform(fond["scaler"].transform(X_new))
    couleurs = [VERT if v == "VRAI BILLET" else ROUGE for v in resultat["verdict"]]
    ax.scatter(proj[:, 0], proj[:, 1], marker="*", s=260, c=couleurs,
               edgecolor="white", linewidths=0.9, zorder=5,
               label="Billets analysés")
    if len(resultat) <= 30:
        for i, ident in enumerate(resultat["id"]):
            ax.annotate(str(ident), (proj[i, 0], proj[i, 1]),
                        textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel("F1 (" + fr(fond["variance"][0], 1) + " % de la variance)")
    ax.set_ylabel("F2 (" + fr(fond["variance"][1], 1) + " % de la variance)")
    ax.set_title("Billets analysés sur le plan factoriel de l'entraînement")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# La page
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Détection de faux billets, ONCFM",
                       layout="wide")
    st.title("Détection de faux billets")
    st.caption("Projet 12, ONCFM. L'application de détection en version "
               "tableau de bord : mêmes modèles, mêmes verdicts que le notebook, "
               "dans le navigateur. Charles Lippens, août 2026.")

    try:
        artefact, complets, tableau_metriques, fond = preparer_modeles()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Réglages")
        choix = st.selectbox("Modèle qui rend les verdicts",
                             [NOM_AUTO, NOM_RL, NOM_KNN, NOM_RF, NOM_KM])
        if choix == NOM_AUTO:
            st.caption("La régression logistique du fichier joblib officiel, "
                       "retenue à l'évaluation sur quatre critères : "
                       "performance, stabilité, lisibilité et coût.")
        seuil = st.slider("Seuil de décision, 0,50 par défaut : probabilité "
                          "au-dessus de laquelle un billet est déclaré vrai",
                          0.05, 0.95, float(artefact.get("seuil", SEUIL_DEFAUT)), 0.01)
        st.caption("Valeur par défaut : 0,50, celle du notebook et du fichier "
                   "du modèle. Le notebook documente aussi le levier à 0,69, "
                   "qui intercepte tous les faux du jeu de test au prix de deux "
                   "vrais contrôlés de plus ; l'arbitrage appartient à l'ONCFM.")
        if abs(seuil - SEUIL_DEFAUT) > 1e-9:
            st.warning("Seuil modifié à " + fr(seuil, 2) + " : les verdicts "
                       "ci-contre ne sont plus ceux du seuil par défaut à 0,50.")

    st.subheader("1. Charger un fichier de billets")
    st.write("Le fichier attendu est un CSV au format de billets_production.csv : "
             "les six mesures en millimètres, diagonal, height_left, height_right, "
             "margin_low, margin_up, length, et une colonne id facultative. "
             "Séparateur virgule ou point-virgule.")
    televerse = st.file_uploader("Déposer le fichier CSV ici", type=["csv"])
    exemple = st.toggle("Utiliser le fichier d'exemple billets_production.csv",
                        value=televerse is None)

    if televerse is not None:
        brut = lire_csv(televerse.getvalue())
        source = televerse.name
    elif exemple:
        brut = lire_csv(locate(EXEMPLE_NAME))
        source = EXEMPLE_NAME
    else:
        st.info("Déposez un fichier CSV ou activez le fichier d'exemple.")
        st.stop()

    try:
        donnees = valider(brut)
        donnees, imputes = imputer_manquants(donnees, artefact["imputation"])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Fichier lu : {source}, {len(donnees)} billet(s), "
               "six mesures contrôlées.")
    if imputes:
        st.warning("Marge basse manquante complétée par la régression "
                   "embarquée pour : " + ", ".join(map(str, imputes)) + ".")
    alertes = signaler_hors_plage(donnees)
    if alertes:
        st.warning("Dimension hors des plages plausibles de l'entraînement, "
                   "à vérifier sur la machine de mesure : " + " ; ".join(alertes) + ".")

    st.subheader("2. Verdicts du modèle : " + choix + " (seuil " + fr(seuil, 2) + ")")
    modele = predicteur(choix, artefact, complets)
    resultat = predire(donnees, modele, seuil)
    nb_vrais = int((resultat["prediction"] == 1).sum())
    nb_faux = len(resultat) - nb_vrais
    incertains = int(resultat["probabilite_vrai"].between(0.20, 0.80).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Vrais billets", nb_vrais)
    c2.metric("Faux billets", nb_faux)
    c3.metric("Verdicts peu tranchés (0,20 à 0,80)", incertains)

    affichage = resultat.copy()
    affichage["probabilite_vrai"] = affichage["probabilite_vrai"].map(lambda p: fr(p))
    st.dataframe(affichage, use_container_width=True, hide_index=True)
    st.download_button("Télécharger les résultats en CSV",
                       resultat.to_csv(index=False).encode("utf-8"),
                       file_name="resultats.csv", mime="text/csv")
    if choix == NOM_AUTO and abs(seuil - SEUIL_DEFAUT) < 1e-9:
        st.caption("Mode automatique au seuil 0,50 : ces verdicts sont "
                   "identiques à ceux du notebook et de l'application terminal.")

    st.subheader("3. Mode visuel des résultats")
    g1, g2 = st.columns(2, gap="medium", vertical_alignment="top")
    with g1:
        st.pyplot(figure_barres(resultat, seuil), use_container_width=True)
    with g2:
        st.pyplot(figure_plan(resultat, fond, artefact), use_container_width=True)

    st.subheader("4. Comparaison des quatre modèles")
    st.write("Sur le protocole du notebook, modèles entraînés sur les 1 200 "
             "billets d'entraînement et évalués sur les 300 billets de test, "
             "seuil 0,50 :")
    st.dataframe(tableau_metriques, use_container_width=True, hide_index=True)
    st.write("Et sur le fichier chargé, les verdicts des quatre modèles "
             "côte à côte :")
    cote = donnees[["id"]].copy()
    for nom in [NOM_RL, NOM_KNN, NOM_RF, NOM_KM]:
        cote[nom] = predire(donnees, predicteur(nom, artefact, complets),
                            seuil)["verdict"].values
    st.dataframe(cote, use_container_width=True, hide_index=True)
    accord = (cote[[NOM_RL, NOM_KNN, NOM_RF, NOM_KM]].nunique(axis=1) == 1)
    if bool(accord.all()):
        st.caption("Les quatre modèles sont unanimes sur chaque billet.")
    else:
        st.caption(f"Modèles unanimes sur {int(accord.sum())} billet(s) "
                   f"sur {len(cote)} ; les désaccords sont les billets à "
                   "regarder de près.")

    st.caption("Le choix final du projet reste la régression logistique : "
               "performance équivalente au random forest, meilleure stabilité "
               "en validation croisée, verdicts lisibles coefficient par "
               "coefficient, et modèle le plus économe, le principe de "
               "parcimonie à performance égale.")


def _sous_streamlit():
    """Vrai uniquement quand le script tourne via streamlit run."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _sous_streamlit():
    main()
elif __name__ == "__main__":
    print("Application Streamlit : lancez-la avec")
    print("    streamlit run Lippens_Charles_2_Application_082026.py")
