README.md
========================================


Organisation des données
------------------------
|
|--samples/          *# Répertoire contenant les fichiers musicaux*


Fichiers source
----------------

Le fichier *algorithm.py* contient deux classes:
- la classe Encoding permet d'extraire l'empreinte des fichiers musicaux et/ou 
  des extraits musicaux
- la classe Matching permet de comparer deux empreintes

*database.py* est un script Python permettant de calculer les 
signatures des morceaux de la base de donnée et de générer un fichier 
(extension .pickle) regroupant les caractéristiques de ces morceaux.

Le fichier *demo.py* permet enfin de sélectionner aléatoirement un extrait dans 
la base de morceaux et cherche ensuite à quel morceau cet extrait correspond.


Base de donnée
----------------

Les morceaux de la base de donnée se trouvent dans le dossier 'samples'. 
Tous les morceaux sont enregistrés au format .wav à une fréquence de 55 kHz.


Librairies nécessaires
----------------------

- numpy
- scipy
- scikit-image
