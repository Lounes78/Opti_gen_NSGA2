# 2éme méthode  :
import numpy as np 
import random
import matplotlib.pyplot as plt

data = np.loadtxt('marche_train.txt')  
temps = data[:, 0]
position_x = data[:, 1] 

class Individu:
    def __init__(self, x: np.ndarray):
        self.x = x  # Variables de décision
        self.objectifs = None  # Valeurs des objectifs
        
        self.rang = float('inf')  # Rang basé sur la domination, on l initialise à l infini
        self.distance_de_crowding = 0  # crowding distance)
        
        self.solutions_dominees = []  # Solutions dominées par cet individu
        self.nombre_dominations = 0  # Nombre de solutions qui dominent cet individu


def nsga2(parametres_probleme, taille_pop, nb_generations, bornes):
    # Initialisation de la population et évaluation initiale
    population = initialiser_population(taille_pop, bornes)
    for individu in population:
        individu.objectifs = evaluer_individu(individu, parametres_probleme)
    
    # Tri initial non dominé
    fronts = tri_non_domine_rapide(population)
    for front in fronts:
        calculer_distance_crowding(front)
    
    
    # Boucle principale des générations
    for generation in range(nb_generations):
        # Création des descendants
        descendants = []
        while len(descendants) < taille_pop:
            parent1 = selection_par_tournoi(population, 2)
            parent2 = selection_par_tournoi(population, 2)
            enfant1, enfant2 = croisement(parent1, parent2, cr=0.9)
            mutation(enfant1, bornes, mr=0.1)
            mutation(enfant2, bornes, mr=0.1)
            descendants.extend([enfant1, enfant2])
        
        # Évaluation des descendants
        for individu in descendants:
            individu.objectifs = evaluer_individu(individu, parametres_probleme)
        
        # Combinaison des populations et tri pour la prochaine génération
        population_combinee = population + descendants
        fronts = tri_non_domine_rapide(population_combinee)
        
        
        ######################
        # Sélection de la prochaine génération
        ######################
        
        population = []
        index_front = 0
        while index_front < len(fronts) and len(population) + len(fronts[index_front]) <= taille_pop:
            calculer_distance_crowding(fronts[index_front])
            population.extend(fronts[index_front])
            index_front += 1
        
        # Ajout des individus restants pour compléter la population
        if len(population) < taille_pop and index_front < len(fronts):
            calculer_distance_crowding(fronts[index_front])
            fronts[index_front].sort(key=lambda x: -x.distance_de_crowding)
            population.extend(fronts[index_front][:taille_pop - len(population)])
        
        # Affichage toutes les 10 générations
        if generation % 10 == 0:
            print(f"Génération {generation}")
    
    return population


def evaluer_individu(individu: Individu, parametres_probleme: dict):
    capacite_batterie, seuil_puissance = individu.x
    capacite_sortie = capacite_batterie * 0.1
    capacite_entree = capacite_batterie * 0.1
    niveau_batterie = capacite_batterie / 2.0
    liste_chute_tension = []
    
    for idx in range(len(parametres_probleme['temps'])):
        position_x = parametres_probleme['position_x'][idx]
        demande_train = parametres_probleme['Ptrain'][idx]
        RCeq = parametres_probleme['calculer_Req'](position_x)
        
        P_batterie, P_LAC, P_rheostat, niveau_batterie = parametres_probleme['gestion_puissances'](
            seuil_puissance,
            demande_train,
            niveau_batterie,
            capacite_batterie,
            capacite_sortie,
            capacite_entree,
            parametres_probleme['Dt']
        )
        
        delta = parametres_probleme['Vsst']**2 - 4 * RCeq * P_LAC
        if delta >= 0:
            vitesse_train = 0.5 * (parametres_probleme['Vsst'] + np.sqrt(delta))
        else:
            vitesse_train = 0
            
        chute_tension = parametres_probleme['Vsst'] - vitesse_train
        liste_chute_tension.append(chute_tension)
    
    return capacite_batterie, max(liste_chute_tension)


def initialiser_population(taille_pop, bornes):
    return [Individu(np.array([random.uniform(bornes[i][0], bornes[i][1]) for i in range(len(bornes))])) 
            for _ in range(taille_pop)]


def tri_non_domine_rapide(population):
    fronts = [[]]
    for p in population:
        p.solutions_dominees = []
        p.nombre_dominations = 0
        
        for q in population:
            if all(p.objectifs[i] <= q.objectifs[i] for i in range(len(p.objectifs))) and \
               any(p.objectifs[i] < q.objectifs[i] for i in range(len(p.objectifs))):
                p.solutions_dominees.append(q)
            elif all(q.objectifs[i] <= p.objectifs[i] for i in range(len(p.objectifs))) and \
                 any(q.objectifs[i] < p.objectifs[i] for i in range(len(p.objectifs))):
                p.nombre_dominations += 1
        
        if p.nombre_dominations == 0:
            p.rang = 0
            fronts[0].append(p)
    
    i = 0
    while i < len(fronts) and fronts[i]:
        prochain_front = []
        for p in fronts[i]:
            for q in p.solutions_dominees:
                q.nombre_dominations -= 1
                if q.nombre_dominations == 0:
                    q.rang = i + 1
                    prochain_front.append(q)
        i += 1
        fronts.append(prochain_front)
    
    return [front for front in fronts if front]


def calculer_distance_crowding(front):
    if len(front) <= 2:
        for individu in front:
            individu.distance_de_crowding = float('inf')
        return
    
    n_objectifs = len(front[0].objectifs)
    for individu in front:
        individu.distance_de_crowding = 0
    
    for m in range(n_objectifs):
        front.sort(key=lambda x: x.objectifs[m])
        front[0].distance_de_crowding = float('inf')
        front[-1].distance_de_crowding = float('inf')
        
        intervalle_objectif = front[-1].objectifs[m] - front[0].objectifs[m]
        if intervalle_objectif == 0:
            continue
            
        for i in range(1, len(front)-1):
            front[i].distance_de_crowding += (front[i+1].objectifs[m] - front[i-1].objectifs[m]) / intervalle_objectif



def selection_par_tournoi(population, taille_tournoi):
    tournoi = random.sample(population, taille_tournoi)
    return min(tournoi, key=lambda x: (x.rang, -x.distance_de_crowding))


def croisement(parent1, parent2, cr):
    if random.random() < cr:
        alpha = random.random()
        enfant1_x = alpha * parent1.x + (1 - alpha) * parent2.x
        enfant2_x = (1 - alpha) * parent1.x + alpha * parent2.x
        return Individu(enfant1_x), Individu(enfant2_x)
    return Individu(parent1.x.copy()), Individu(parent2.x.copy())



def mutation(individu, bornes, mr):
    for i in range(len(individu.x)):
        if random.random() < mr:
            individu.x[i] = random.uniform(bornes[i][0], bornes[i][1])

def afficher_resultats(population):
    recherche = np.array([ind.x for ind in population])
    plt.figure(figsize=(10, 6))
    plt.scatter(recherche[:, 0], recherche[:, 1], c='blue')
    plt.xlabel('Capacité Batterie (KWh)')
    plt.ylabel('Puissance de Seuil (w)')
    plt.title('domaine de recherche ')
    plt.grid(True)
    plt.show()

    objectifs = np.array([ind.objectifs for ind in population])
    objectifs0_KWh = objectifs[:, 0]/ (3600.0*1000)
    plt.figure(figsize=(10, 6))
    plt.scatter(objectifs0_KWh, objectifs[:, 1], c='blue')
    plt.xlabel('Capacité Batterie (KWh)')
    plt.ylabel('Chute de Tension Max (V)')
    plt.title('Front de Pareto')
    plt.grid(True)
    plt.show()