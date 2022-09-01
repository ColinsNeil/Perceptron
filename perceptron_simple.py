data_set = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (1, 3, 3),
    (1, 3, 4),
    (1, 4, 3),
    (1, 4, 4)
    ]

data_set_res = [
    1,
    1,
    1,
    0,
    0,
    0,
    0
]

def estimation(poids, caracteres):
    nb_poids = len(poids)
    res = 0

    for i in range(nb_poids):
        res += poids[i] * caracteres[i]

    return res

def miseAJourPoids(poids, caractere, estim, res):
    vitesse_apprentissage = 0.1

    return poids + vitesse_apprentissage * ( res - estim ) * caractere

def apprentissage(exemples):
    nb_poids = len(exemples[0])
    nb_exemples = len(exemples)

    poids = []
    estim = []

    for i in range(nb_poids):
        poids.append(0)

    for i in range(nb_exemples):
        exemple = exemples[i]

        estim.append(estimation(poids, exemple))

        for j in range(nb_poids):
            poids[j] = miseAJourPoids(poids[j], exemple[j], estim[i], data_set_res[i])

    print(poids)

    """

    x1 = 4
    x2 = 4
    print(1 * poids[0] + x1 * poids[1] + x2 * poids[2])

    """


apprentissage(data_set)