
import numpy as np 
import random 

# les données : 
RLAC_per_m = 95e-6
Rrail_per_m = 10e-6
Rsst = 33e-3


# fonction pour calculer R de Lac
def calculer_RLAC(x):
    """
    Entrées :
        - x : Distance (en mètres) depuis une sous-station.
    Sorties :
        - Résistance linéique totale (en Ohms) pour la distance donnée.
    """
    return RLAC_per_m * x

# fonction pour calculer R de rail 
def calculer_Rrail(x):
    """
    Entrées :
        - x : Distance (en mètres) depuis une sous-station.
    Sorties :
        - Résistance du rail totale (en Ohms) pour la distance donnée.
    """
    return Rrail_per_m * x

# Fonction pour calculer Req
def calculer_Req(x):
    """
    Entrées :
        - x : Position actuelle du train (en mètres).
    Sorties :
        - Req : Résistance équivalente (en Ohms) vue par le train.
    """
    RLAC1 = calculer_RLAC(x)
    Rrail1 = calculer_Rrail(x)
    RLAC2 = calculer_RLAC(5000 - x)  
    Rrail2 = calculer_Rrail(5000 - x)
    R1 = RLAC1 + Rrail1 + Rsst
    R2 = RLAC2 + Rrail2 + Rsst
    return ( (R1*R2)/(R1 + R2 ) )

# Fonction pour calculer la puissance mécanique
def calculer_puissance_mecanique(temps, position_x):
    """
    Entrées :
        - temps : Tableau de temps (en secondes).
        - position_x : Tableau des positions du train (en mètres).
    Sorties :
        - Pmechanique : Puissance mécanique (en Watts) calculée pour chaque instant.
    """
    vitesse = np.gradient(position_x, temps)
    acceleration = np.gradient(vitesse, temps)

    # Paramètres pour le calcul de la force résistive
    M = 70000 
    Ao = 780
    A1 = 6.4
    Bo = 0
    B1 = 0.14
    Co = 0.3634
    C1 = 0

    # Force résistive
    Fresistive = (Ao + A1 * M/1000) + (Bo + B1 * M/1000) * 3.6*vitesse + (Co + C1 * M/1000) * (3.6*vitesse)**2

    # Force motrice
    Fmotrice = M * acceleration + M * 9.81 * np.sin(0) + Fresistive  # En supposant une pente nulle pour le moment

    Pmechanique = Fmotrice * vitesse
    return Pmechanique

# fonction pour la gestion de batterie : 
def gestion_puissances(P_seuil, train_demand, battery_level, battery_capacity, battery_output_capacity, battery_input_capacity, Dt=1, battery_efficiency=0.9):
    """
    Entrées :
        - P_seuil : Puissance seuil de la LAC (en Watts).
        - train_demand : Puissance demandée par le train (en Watts).
        - battery_level : Niveau d'énergie actuel de la batterie (en Joules).
        - battery_capacity : Capacité totale de la batterie (en Joules).
        - battery_output_capacity : Puissance maximale de sortie de la batterie (en Watts).
        - battery_input_capacity : Puissance maximale d'entrée de la batterie (en Watts).
        - Dt : Intervalle de temps (en secondes).
        - battery_efficiency : Rendement de la batterie (sans unité).
    Sorties :
        - P_battery : Puissance fournie ou absorbée par la batterie (en Watts).
        - P_LAC : Puissance fournie par la LAC (en Watts).
        - P_rheostat : Puissance dissipée par le rhéostat (en Watts).
        - battery_level : Niveau d'énergie mis à jour de la batterie (en Joules).
    """
    P_LAC = 0.0
    P_battery = 0.0
    P_rheostat = 0.0
    
    # Phase de consommation élevée (accélération ou forte demande)
    if train_demand >= P_seuil:
        # Puissance cible que la batterie doit fournir :
        target_battery_power = train_demand - P_seuil
        max_battery_power = min(battery_output_capacity, battery_level / Dt)
        requested_discharge = min(target_battery_power, max_battery_power)

        P_battery = requested_discharge
                
        # Si la batterie ne peut pas fournir toute la différence, on ajuste la LAC :
        if P_battery < target_battery_power:
            # Complément de puissance à fournir par la LAC
            P_LAC = P_seuil + (target_battery_power - P_battery)
        else:
            # La batterie couvre entièrement la différence : la LAC fournit uniquement P_seuil
            P_LAC = P_seuil
        
        # Dans tous les cas, le rhéostat ne consomme pas ici
        P_rheostat = 0


    elif 0 < train_demand < P_seuil:
        # Demande modérée : la LAC couvre cette demande entièrement
        P_LAC = train_demand
        P_battery = 0
        P_rheostat = 0


    else:
        # train_demand <= 0 --> récupération d'énergie vers la batterie ou dissipation
        P_LAC = 0
        if battery_level < battery_capacity:
            # Propose de stocker |train_demand| dans la batterie dans la limite de battery_input_capacity et de battery_level
            proposed_battery_power = max(train_demand, -battery_input_capacity / Dt)
            new_energy_if_charged = battery_level + (-proposed_battery_power * Dt * battery_efficiency)


            if new_energy_if_charged > battery_capacity:
                max_charge = battery_capacity - battery_level
                P_battery = -max_charge / (Dt*battery_efficiency)
                P_rheostat = -(train_demand - P_battery)

            else:
                P_battery = proposed_battery_power
                if (P_battery == -battery_input_capacity) or (P_battery == -battery_level / Dt):
                    P_rheostat = -(train_demand - P_battery)
                else:
                    P_rheostat = 0
                
        else:
            # Batterie pleine
            P_battery = 0
            P_rheostat = -train_demand


    if P_battery > 0:
        # Discharging: from battery's perspective, it must supply P_battery / battery_efficiency
        battery_level -= (P_battery / battery_efficiency) * Dt
    else:
        # Charging (P_battery <= 0): battery gains -P_battery * battery_efficiency
        battery_level += (-P_battery * battery_efficiency) * Dt


    # On devrait pas dépasser la capacité de la batterie MAIS BON .... 
    if battery_level > battery_capacity:
        print(battery_level)
        battery_level = battery_capacity
        print(str(battery_level) + '\n')
        
           
    return P_battery, P_LAC, P_rheostat, battery_level
