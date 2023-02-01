import pgmpy

#En su editor de python cree un nuevo archivo alarma.py e implemente el modelo creado 
# para el problema de la alarma antirrobo. Utilice las clases BayesianNetwork, TabularCPD 
# y VariableElimination de pgmpy.

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

#Defina ahora la estructura de la red incluyendo los arcos y nodos

model_2 = BayesianNetwork([("R","Al"),("S","Al"),("Al","J"),("Al","M")])

#Defina las CPDs de R y S

cpd_r = TabularCPD(variable="R", variable_card=2, values=[[0.01], [0.99]])
cpd_s = TabularCPD(variable="S", variable_card=2, values=[[0.02], [0.98]])

#Defina la CPD de A

cpd_al = TabularCPD(
    variable="Al",
    variable_card=2,
    values=[
        [0.95, 0.94, 0.29, 0.001],
        [0.05, 0.06, 0.71, 0.999],
    ],
    evidence=["R", "S"],
    evidence_card=[2, 2],
)

#Defina la CPD de J

cpd_j = TabularCPD(
    variable="J",
    variable_card=2,
    values=[
        [0.9, 0.05],
        [0.1, 0.95],
    ],
    evidence=["Al"],
    evidence_card=[2],
)

#Defina la CPD de M

cpd_m = TabularCPD(
    variable="M",
    variable_card=2,
    values=[
        [0.7, 0.01],
        [0.3, 0.99],
    ],
    evidence=["Al"],
    evidence_card=[2],
)

#Asocie las 5 CPDs a su modelo

model_2.add_cpds(cpd_r, cpd_s, cpd_al, cpd_j, cpd_m)

#Revise que su modelo est ́e completo

model_2.check_model()

print(model_2.get_independencies())

#Importe la clase VariableElimination del paquete de inferencia y cree un objeto de
#esta clase

from pgmpy.inference import VariableElimination

infer_2 = VariableElimination(model_2)

# Llama Maria y Juan

posterior_p2a  =infer_2.query(["R"], evidence = {"J": 0, "M": 0})
print(posterior_p2a)

# Llama Juan

posterior_p2b =infer_2.query(["R"], evidence = {"J": 0, "M": 1})
print(posterior_p2b)

# Llama Maria

posterior_p2c =infer_2.query(["R"], evidence = {"J": 1, "M": 0})
print(posterior_p2c)

# Ninguno de los 2 llama

posterior_p2d =infer_2.query(["R"], evidence = {"J": 1, "M": 1})
print(posterior_p2d)