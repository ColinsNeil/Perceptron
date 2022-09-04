def Perceptron(entrees, poids):
    """
    Desc : Un perceptron

    Paramètres : 
            entrees : Liste des entrées
            poids : Liste des poids de chaque entrée

    Retourne :
            sortie : 1 si le produit scalaire entre les entrees et des poids est positif et 0 sinon
    """
    nb_entrees = len(entrees)
    prod_scal = 0
    sortie = 0

    for i in range(nb_entrees):
        prod_scal += entrees[i] * poids[i]
    
    if (prod_scal > 0):
        sortie = 1

    return sortie

def MiseAJourPoids(entrees, poids, vit_appr, res_att, estim):
    """
    Desc : Fonction qui permet de mettre à jour tout les poids des entrées

    Paramètres :
            entrees : Liste des entrées
            poids : Liste des poids de chaque entrée
            vit_appr : Vitesse d'apprentissage du perceptron
            res_att : Résultat attendu
            estim : Estimation du perceptron
    
    Retourne :
            nouv_poids : Nouveaux poids
    """
    nb_poids = len(poids)
    nouv_poids = poids

    for i in range(nb_poids):
        nouv_poids[i] = poids[i] + vit_appr * (res_att - estim) * entrees[i]

    return nouv_poids

def CalculerErreur(liste_estim, liste_res_att):
    """
    Desc : Fonction qui calcul l'écart type entre les estimations du perceptron et les résultats attendu

    Paramètres :
            liste_estim : Liste des estimations du perceptron pour chaque exemples
            liste_res_att : Liste des résultats attendu pour chaques exemples

    Retourne :
            res = L'écart type entre les résultats attendu et les écarts types
    """
    nb_donnees = len(liste_estim)
    res = 0

    for i in range(nb_donnees):
        res += abs(liste_res_att[i] - liste_estim[i])
    
    res = res / nb_donnees

    return res

def Apprentissage(jeu_donnees):
    """
    Desc : Algorithme d'apprentissage du perceptron

    Paramètre :
            jeu_donnees : Un jeu de données

    Retourne :
            poids : Les poids correctes pour le perceptron
    """
    nb_donnees = len(jeu_donnees)
    nb_poids = len(jeu_donnees[0][0]) + 1

    poids = [0] * nb_poids
    liste_estim = [0] * nb_donnees
    liste_res_att = [0] * nb_donnees

    while True :
        for i in range(nb_donnees):
            entrees = [1] + jeu_donnees[i][0]
            res_att = jeu_donnees[i][1]

            liste_res_att[i] = res_att

            estim = Perceptron(entrees, poids)

            liste_estim[i] = estim

            poids = MiseAJourPoids(entrees, poids, 0.1, res_att, estim)

        erreur = CalculerErreur(liste_estim, liste_res_att)

        if (erreur < 0.1):
            break
    
    return poids