from tqdm import tqdm
import numpy as np 

# algorithme Monte carlo : 
def monte_carlo_iteration(
    capacity_range,
    P_seuil_range,
    gestion_puissances,
    temps,
    position_x,
    Ptrain,
    Vsst,
    calculer_Req,
    Dt
):    

    """
    Entrées :
        - capacity_range : Intervalle des capacités de la batterie (min, max).
        - P_seuil_range : Intervalle des puissances seuil de la LAC (min, max).
        - gestion_puissances : Fonction pour gérer les puissances (batterie, LAC, rhéostat).
        - temps : Tableau des instants de temps (en secondes).
        - position_x : Tableau des positions du train (en mètres).
        - Ptrain : Tableau des puissances demandées par le train (en Watts).
        - Vsst : Tension aux bornes des sous-stations (en Volts).
        - calculer_Req : Fonction pour calculer la résistance équivalente.
        - Dt : Pas de temps (en secondes).
    Sorties :
        - Tuple contenant :
            - battery_capacity : Capacité de la batterie simulée (en Joules).
            - P_seuil : Puissance seuil simulée (en Watts).
            - chute_de_tension_max : Chute de tension maximale (en Volts).
    """
    
    battery_capacity = np.random.uniform(capacity_range[0], capacity_range[1])
    P_seuil = np.random.uniform(P_seuil_range[0], P_seuil_range[1])
    
    battery_output_capacity = battery_capacity * 0.1
    battery_input_capacity = battery_capacity * 0.1
    
    battery_level = battery_capacity / 2.0
    chute_de_tension_list = []
    

    for idx in range(len(temps)):
        x = position_x[idx]
        train_demand = Ptrain[idx]
        
        RCeq = calculer_Req(x)
        
        P_battery, P_LAC, P_rheostat, battery_level = gestion_puissances(
            P_seuil,
            train_demand,
            battery_level,
            battery_capacity,
            battery_output_capacity,
            battery_input_capacity,
            Dt
        )
        
        delta = Vsst**2 - 4 * RCeq * P_LAC
        if delta >= 0:
            Vtrain = 0.5 * (Vsst + np.sqrt(delta))
        else:
            Vtrain = 0
            
        chute_tension = Vsst - Vtrain
        chute_de_tension_list.append(chute_tension)
    
    chute_de_tension_max = max(chute_de_tension_list)
    return (battery_capacity, P_seuil, chute_de_tension_max)


def run_monte_carlo(n_iterations, capacity_range, P_seuil_range, *args):
    """
    Entrées :
        - n_iterations : Nombre d'itérations à exécuter.
        - capacity_range : Intervalle des capacités de la batterie (min, max).
        - P_seuil_range : Intervalle des puissances seuil de la LAC (min, max).
        - *args : Arguments supplémentaires pour la fonction `monte_carlo_iteration`.
    Sorties :
        - results : Liste des résultats de chaque itération Monte Carlo. Chaque élément est un tuple
                    contenant (battery_capacity, P_seuil, chute_de_tension_max).
    """
    results = []
    
    for _ in tqdm(range(n_iterations), desc="Monte Carlo Iterations"):
        result = monte_carlo_iteration(capacity_range, P_seuil_range, *args)
        results.append(result)
        
    return results


def find_non_dominated_solutions(results):
    """
    Entrées :
        - results : Liste des solutions obtenues (chaque élément est un tuple contenant 
                    (battery_capacity, P_seuil, chute_de_tension_max)).
    Sorties :
        - non_dominated : Liste des solutions non dominées. Chaque élément est un tuple contenant
                          (battery_capacity, P_seuil, chute_de_tension_max).
    """
    non_dominated = []
    
    for i, sol_i in enumerate(results):
        cap_i, p_seuil_i, chute_i = sol_i
        
        dominated = False
        
        for j, sol_j in enumerate(results):
            if i == j:
                continue
            
            cap_j, p_seuil_j, chute_j = sol_j
            
            # Check if sol_j dominates sol_i
            if (cap_j <= cap_i and chute_j <= chute_i) and (cap_j < cap_i or chute_j < chute_i):
                dominated = True
                break
        
        if not dominated:
            non_dominated.append(sol_i)
    
    return non_dominated
    non_dominated = find_non_dominated_solutions(results)