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
i1 = []
i2 = []
i = []
P1 = []
P2 = []

for idx in range(len(temps)):
    x = position_x[idx]
    Req = calculer_Req(x)
    P = Ptrain[idx]
    delta = Vsst**2 - 4 * Req * P
    if delta >= 0:
        V = 0.5 * (Vsst + np.sqrt(delta))
        Vtrain.append(V)
        i_value = (Vsst - V) / Req
        i.append(i_value)
        # Correction des courants i1 et i2 pour mieux correspondre à la distribution des résistances
        i1_value = (Vsst - V) / (Rsst + calculer_Rrail(x) + calculer_Rrail(x))
        i2_value = (Vsst - V) / (Rsst + calculer_Rrail(5000 - x) + calculer_Rrail(5000 - x))
        P1_value = (Rsst + calculer_Rrail(x) + calculer_Rrail(x)) * i1_value**2 
        P2_value = (Rsst + calculer_Rrail(5000- x) + calculer_Rrail(5000-x)) * i2_value**2 
        i1.append(i1_value)
        i2.append(i2_value)
        P1.append(P1_value)
        P2.append(P2_value)

    else:
        Vtrain.append(np.nan) 
        i_value = Vsst / Req
        i1_value = Vsst / (Rsst + calculer_Rrail(x) + calculer_Rrail(x))
        i2_value = Vsst / (Rsst + calculer_Rrail(5000 - x) + calculer_Rrail(5000 - x))
        P1_value = (Rsst + calculer_Rrail(x) + calculer_Rrail(x)) * i1_value**2 
        P2_value = (Rsst + calculer_Rrail(5000- x) + calculer_Rrail(5000-x)) * i2_value**2 
        i.append(i_value)
        i1.append(i1_value)
        i2.append(i2_value)
        P1.append(P1_value)
        P2.append(P2_value)

# Tracer les courants (i, i1, i2)
plt.figure(figsize=(10, 6))
plt.plot(temps, i, label='Courant total i', color='blue')
plt.plot(temps, i1, label='Courant i1 (1ère sous-station)', color='green')
plt.plot(temps, i2, label='Courant i2 (2ème sous-station)', color='red')
plt.title('Courants au cours du temps')
plt.xlabel('Temps (s)')
plt.ylabel('Courant (A)')
plt.legend()
plt.grid(True)
plt.show()

# Tracer les puissances (P1 et P2)
plt.figure(figsize=(10, 6))
plt.plot(temps, P1, label='Puissance fournie par la 1ère sous-station', color='green')
plt.plot(temps, P2, label='Puissance fournie par la 2ème sous-station', color='red')
plt.title('Puissance fournie par chaque sous-station au cours du temps')
plt.xlabel('Temps (s)')
plt.ylabel('Puissance (W)')
plt.legend()
plt.grid(True)
plt.show()

# Tracer la tension Vtrain
plt.figure(figsize=(10, 6))
plt.plot(temps, Vtrain, label='Tension aux bornes du train (Vtrain)', color='purple')
plt.title('Evolution de la tension aux bornes du train (Vtrain) au cours du temps')
plt.xlabel('Temps (s)')
plt.ylabel('Tension (V)')
plt.legend()
plt.grid(True)
plt.show()
