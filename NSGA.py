
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np 
import matplotlib.pyplot as plt
from fonctions import gestion_puissances

# from pymoo.core.problem import ElementwiseProblem
# import numpy as np
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.optimize import minimize
# import matplotlib.pyplot as plt

class BatteryOptimizationProblem(ElementwiseProblem):
    def __init__(self, temps, position_x, Ptrain, Vsst, calculer_Req, Dt, gestion_puissances):
        super().__init__(
            n_var=2,
            n_obj=2,
            xl=np.array([0, 0]),
            xu=np.array([50000000, 1000000])
        )
        self.temps = temps
        self.position_x = position_x
        self.Ptrain = Ptrain
        self.Vsst = Vsst
        self.calculer_Req = calculer_Req
        self.Dt = Dt
        self.gestion_puissances = gestion_puissances

    def _evaluate(self, x, out, *args, **kwargs):
        battery_capacity, P_seuil = x
        battery_output_capacity = battery_capacity * 0.1
        battery_input_capacity = battery_capacity * 0.1
        battery_level = battery_capacity / 2.0
        chute_de_tension_list = []

        for idx in range(len(self.temps)):
            x = self.position_x[idx]
            train_demand = self.Ptrain[idx]
            RCeq = self.calculer_Req(x)

            P_battery, P_LAC, P_rheostat, battery_level = self.gestion_puissances(
                P_seuil,
                train_demand,
                battery_level,
                battery_capacity,
                battery_output_capacity,
                battery_input_capacity,
                self.Dt
            )

            delta = self.Vsst**2 - 4 * RCeq * P_LAC
            if delta >= 0:
                Vtrain = 0.5 * (self.Vsst + np.sqrt(delta))
            else:
                Vtrain = 0

            chute_tension = self.Vsst - Vtrain
            chute_de_tension_list.append(chute_tension)

        out["F"] = np.array([battery_capacity, max(chute_de_tension_list)])

