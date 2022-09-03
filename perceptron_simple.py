def Estimation(entrees, poids):
    """
    Desc : Fait le produit scalaire des entrées avec les poids pour faire une prédiction

    Paramètres : 
            entrees : Nombre totales des entrées
            poids : poids de chaque entrée

    Retourne :
            res : Prédiction par rapport aux entrées et aux poids
    """
    nb_entrees = len(entrees)
    res = 0

    for i in range(nb_entrees):
        res += entrees[i] * poids[i]
    
    return res