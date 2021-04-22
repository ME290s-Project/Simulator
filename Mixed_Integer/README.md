# Mixed—Integer Programming Example 
* A classical problem in scheduling and integer programming is the unit commitment problem. In this problem, our task is to turn on and off power generating plants, in order to meet a forecasted future power demand, while minimizing our costs.
* We have three different power plants with different characteristics and running costs, and various constraints on how they can be used.
* The most important thing we learn in this example is that you never multiply binary variables with continuous variables to model on/off behavior. Instead, we derive equivalent linear presentations.
* To begin with, we assume we have power generating plants of different types (nuclear, hydro, gas, oil, coal, …). Each of the plant have a maximum power capacity , and a minimum capacity , when turned on. Our scheduling problem is solved over 48 time units (say hours), and the forecasted power demand is given by a periodic function. The cost of running plant for one time unit is given by a linear function where is the delivered power from plant .

# Idea
* Our project is much similar to this mixed integer problem.