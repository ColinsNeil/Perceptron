def Estimation(entrees, poids):
    """
    Desc : Fait le produit scalaire des entrées avec les poids pour faire une prédiction

    Paramètres : 
            entrees : Liste des entrées
            poids : Liste des poids de chaque entrée

    Retourne :
            estim : Estimation par rapport aux entrées et aux poids
    """
    nb_entrees = len(entrees)
    estim = 0

    for i in range(nb_entrees):
        estim += entrees[i] * poids[i]
    
    return estim

def MiseAJourPoids(entrees, poids, vit_appr, res_att):
    """
    Desc : Fonction qui permet de mettre à jour tout les poids des entrées

    Paramètres :
            entrees : Liste des entrées
            poids : Liste des poids de chaque entrée
            vit_appr : Vitesse d'apprentissage du perceptron
            res_att : Résultat attendu
    
    Retourne :
            nouv_poids : Nouveaux poids
    """
    nb_poids = len(poids)
    nouv_poids = poids

    estim = Estimation(entrees, poids)

    for i in range(nb_poids):
        nouv_poids[i] = poids[i] + vit_appr * (res_att - estim)

    return nouv_poids