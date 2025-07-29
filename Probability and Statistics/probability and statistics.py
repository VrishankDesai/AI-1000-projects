from itertools import product

#Sample space of a dice roll
sample_space = list(range(1,7))

#Probability of rolling an even number
even_numbers = [2,4,6]
P_even = len(even_numbers)/len(sample_space)
print("P(even): ",P_even)

import numpy as np

#Random variable: dice roll
outcomes = np.array([1,2,3,4,5,6])
probabilties = np.array([1/6]*6)

#Expectation
expectation = np.sum(outcomes*probabilties)
print("Mean: ",expectation)

#Variance and standard deviation
variance = np.sum((outcomes-expectation)**2 * probabilties)
std_dev = np.sqrt(variance)
print("Variance: ",variance)
print("Standard deviation: ",std_dev)

#Stimulating 10,000 dice rolls

rolls = np.random.randint(1,7,size=10000)

#Calculating probabilties

P_even = np.sum(rolls%2==0)/len(rolls)
P_greater_than_4 = np.sum(rolls>4)/len(rolls)

print("P_even: ",P_even)
print("P_greater_than_4: ",P_greater_than_4)

#Creating and analyzing random variables
import matplotlib.pyplot as plt
from scipy.stats import uniform

#Discrete random variables: Dice roll
outcomes = [1,2,3,4,5,6]
probabilties1 = [1/6]*6
plt.bar(outcomes,probabilties1,color="blue",alpha=0.7)
plt.title("PMF of a dice roll")
plt.xlabel("Outcomes")
plt.ylabel("Probability")
plt.show()

#Continous random variable: uniform distribution
x=np.linspace(0,1,100)
pdf=uniform.pdf(x,loc=0,scale=1)
plt.plot(x,pdf,color="red")
plt.title("PDF of uniform(0,1)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()