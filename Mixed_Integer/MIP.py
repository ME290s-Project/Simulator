import numpy as np 
import pyomo.environ as pyo 
import matplotlib.pyplot as plt 

Horizon = 48 
T = np.array([t for t in range(0,Horizon)])

# predicted demand  
demand = np.array([100 + 50*np.sin(t*2*np.pi / 24) for t in T])

# problem data:
Nplant = 3  # number of plants 
N = np.array([n for n in range(0,Nplant)]) # list of plant 
pmax = [100, 50, 25]  # max power of the 3 plants: nuclear, fire, wave,, 
pmin = [20, 40, 1]    # min power of the plants 
C = [10,20,20]   # cost of plants 

model = pyo.ConcreteModel() 
model.N = pyo.Set(initialize = N)  # set of plant which can iterate automatically in pyomo
model.T = pyo.Set(initialize = T) 
model.x = pyo.Var(model.N, model.T)  # x[i,t]: production of plant i at time t 
model.u = pyo.Var(model.N, model.T, domain = pyo.Binary)  # u[i,t]: on/off of plant i at time t

# objectve 
model.cost = pyo.Objective(
    expr = sum(model.x[n,t] * C[n] for t in model.T for n in model.N), 
    sense = pyo.minimize
)
# demand constraint:
model.demand = pyo.Constraint(
    model.T, 
    rule = lambda model,t: sum(model.x[n,t] for n in N) >= demand[t]  # the sum of production should satisfy the demand at any time
)

# production constraint:
model.lowbound =  pyo.Constraint(
    model.N, model.T, 
    rule = lambda model, n,t: model.x[n,t] >= pmin[n] * model.u[n,t]  # production is greater than minimal production when it is off 
)

model.upbound = pyo.Constraint(
    model.N, model.T, 
    rule = lambda model, n, t: model.x[n,t] <= pmax[n] * model.u[n,t] # production is less than max production when it is on
)

solver = pyo.SolverFactory('glpk')
result = solver.solve(model) 

u1 = [model.x[0,0]()]
u2 = [model.x[1,0]()]
u3 = [model.x[2,0]()]
for t in T:
    u1.append(model.x[0,t]())
    u2.append(model.x[1,t]())
    u3.append(model.x[2,t]())

plt.figure()
plt.step(u1,'b')

plt.step(u2,'g')
plt.step(u3,'r')
plt.show() 