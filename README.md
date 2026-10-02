# Charles Lippens, Data Analyst : dossier de compétences

Site statique (HTML, CSS et JavaScript, sans framework ni étape de construction), publié sur GitHub Pages :
https://charleslippensdata.github.io/

Le dossier présente la mission du projet 13 (le notebook BottleNeck amélioré avec l'IA), la veille et ses critères,
le registre des quatorze projets du parcours Data Analyst d'OpenClassrooms avec leurs livrables, les compétences et le
contact. Il se lit à l'écran, en thème clair ou sombre, et s'imprime en A4.

## Structure

- `index.html` : la page unique, avec son sommaire.
- `assets/style.css` et `assets/app.js` : mise en page, thèmes, filtres du registre, impression.
- `assets/img/` : les cinq figures, redessinées à partir des sorties du notebook, la capture de la veille, l'icône et
  l'image de partage.
- `assets/fonts/` : Fraunces, Newsreader et IBM Plex Mono, sous licence SIL Open Font License (textes des licences
  dans le même dossier).
- `livrables/` : notebooks et documents du projet 13 (en PDF, et en Word pour les sept documents), CV, livrables des
  projets 1 à 12 dans `livrables/projets/`.
- `404.html`, `robots.txt`, `sitemap.xml`, `.nojekyll` : fichiers d'hébergement.

## Autres présentations

Le même contenu existe en deux autres présentations, publiées chacune dans son dépôt :

- version 3, une page unique : https://charleslippensdata.github.io/portfolio-v3/
  (dépôt `portfolio-v3`) ;
- version 2, un site de six pages : https://charleslippensdata.github.io/portfolio-v2/
  (dépôt `portfolio-v2`).

Ce dossier, la version 4, reste la référence ; les trois versions renvoient aux mêmes documents, publiés ici dans `livrables/`.

## Consulter en local

Depuis ce dossier : `python -m http.server 8000`, puis ouvrir http://localhost:8000 dans un navigateur.

## Données et confidentialité

Aucun cookie, aucune mesure d'audience ; le thème choisi reste dans le navigateur du visiteur. La page ne charge
aucune ressource d'un autre serveur. Les jeux de données des projets sont fournis par OpenClassrooms ou ouverts.
