import numpy as np
import matplotlib.pyplot as plt

# Données du problème
Vsst = 790  
Rsst = 33e-3  
RLAC_per_m = 131e-6  
Rrail_per_m = 18e-6  

# Charger les données de la marche du train
data = np.loadtxt('marche_train.txt')  
temps = data[:, 0]
position_x = data[:, 1] * 1000


def calculer_RLAC(x):
    return RLAC_per_m * x

def calculer_Rrail(x):
    return Rrail_per_m * x

def calculer_Req(x):
    RLAC1 = calculer_RLAC(x)
    Rrail1 = calculer_Rrail(x)
    RLAC2 = calculer_RLAC(5000 - x)  # Longueur totale de 5 km
    Rrail2 = calculer_Rrail(5000 - x)
    return ( (Rsst + RLAC1 + Rrail1)*(Rsst + RLAC2 + Rrail2) / ((Rsst + RLAC1 + Rrail1)+(Rsst + RLAC2 + Rrail2)) )

def calculer_puissance_mecanique(temps, position_x):
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
    Fresistive = (Ao + A1 * M) + (Bo + B1 * M) * vitesse + (Co + C1 * M) * vitesse**2

    # Force motrice
    Fmotrice = M * acceleration + M * 9.81 * np.sin(0) + Fresistive  # En supposant une pente nulle pour le moment


    Pmechanique = Fmotrice * vitesse
    return Pmechanique

Pmechanique = calculer_puissance_mecanique(temps, position_x)

# Calculer la puissance électrique consommée (à chaque temps t)
Ptrain = Pmechanique / 0.80 + 35000  # Rendement de 80% et consommation constante de 35 kW


# Calculer Vtrain au cours du temps
Vtrain = []
for i in range(len(temps)):
    x = position_x[i]
    Req = calculer_Req(x)
    P = Ptrain[i]
    delta = Vsst**2 - 4 * Req * P
    if delta >= 0:
        V = 0.5 * (Vsst + np.sqrt(delta))
        Vtrain.append(V)
    else:
        Vtrain.append(np.nan) 

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(temps, Vtrain)
plt.title('Evolution de la tension aux bornes du train (Vtrain) au cours du temps')
plt.xlabel('Temps (s)')
plt.ylabel('Tension (V)')
plt.grid(True)
plt.show()