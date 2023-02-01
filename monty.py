import pgmpy

#En su editor de python cree un nuevo archivo monty.py e 
#incluya los objetos BayesianNetwork y TabularCPD

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

#Defina ahora la estructura de la red incluyendo los arcos y nodos

model = BayesianNetwork([("C","A"),("U","A")])

#Defina las CPDs de C y U

cpd_c = TabularCPD(variable="C", variable_card=3, values=[[0.33], [0.33], [0.33]])
cpd_u = TabularCPD(variable="U", variable_card=3, values=[[0.33], [0.33], [0.33]])

#Defina la CPD de A

cpd_a = TabularCPD(
    variable="A",
    variable_card=3,
    values=[
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
    ],
    evidence=["C", "U"],
    evidence_card=[3, 3],
)

#Asocie las 3 CPDs a su modelo

model.add_cpds(cpd_c, cpd_u, cpd_a)

#Revise que su modelo est ́e completo

model.check_model()

#Importe la clase VariableElimination del paquete de inferencia y cree un objeto de
#esta clase

from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

# Suponga que Ud selecciona la puerta 1 y el animador la puerta 3. ¿Cual es la probabilidad 
# de que el carro este detras de cada una de las puertas?

posterior_p =infer.query(["C"], evidence = {"U": 0, "A": 2})
print(posterior_p)