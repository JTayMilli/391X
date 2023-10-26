# %%

import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import pyplot as plt
import csv
from scipy.stats import norm, binom 
from scipy import linalg
import random
from decimal import Decimal
import pydtmc

# %%
Decimal(2)

# %%
## Base parameters
LC = 1 ## Lobby Cost
T = 10 ## Tax
mF = 50 ## Mean Fee of Getting Caught
nPPL = 9
R = nPPL * .6 # enforcement resources

# %%
LobH = [4] ## Lobby History, arbitrary preset
CatH = [1] ## Catch history, arbitrary preset

# %%
def pSS1(t = len(LobH) - 1): #Gets probability of lobbying successfully, subjective
    return 1 - ((.8+CatH[t]) / (1+LobH[t]))

# %%
def pSS(nL, nC): #Gets probability of lobbying successfully, subjective
    return 1 - ((.8+nC) / (1+nL))

# %%
def pCat(t = len(LobH) - 1):  #Gets objective probability of success 
    if LobH[t] <= R:
        return .8
    else:
        return .8 * (R / LobH[t])

# %%
def pCat2(nL):  #Gets objective probability of success 
    if nL <= R:
        return .8
    else:
        return .8 * (R / nL)

# %%
def pL(): ## Calculates subjective probability of lobbying
    ps = pSS()
    temp = (-LC + (ps*T))/((1-ps)*mF) ## value that the normal distrobution draw need to be less then to choose to lobby
    return norm(loc = 1, scale = 1).cdf(temp) ## probability of lobbying for individual

# %%
def pL(nL, nC): ## Calculates subjective probability of lobbying
    ps = pSS(nL, nC)
    temp = (-LC + (ps*T))/((1-ps)*mF) ## value that the normal distrobution draw need to be less then to choose to lobby
    return norm(loc = 1, scale = 1).cdf(temp) ## probability of lobbying for individual

# %%
lis = []
for i in range(0,nPPL+1):
    for r in range(0,nPPL+1):
        lis.append(i)
        
TM = pd.DataFrame({'L': lis }) #Transition Matrix

c = []
for i in range(0,nPPL+1):
    for r in range(0,nPPL+1):
        c.append(r)

TM["C"] = c
TM = TM.drop(TM[TM.C > TM.L].index) ## drop rows where C > L

for i in range(nPPL+1): ## Clear Matrix
    for r in range(0,i+1):
        TM[f"{i},{r}"] = 0

TM = TM.reset_index(drop = True)
TM.head(10)


# %%
#int(TM.columns[3][0])
#TM.loc[4]
pl = pL(nL = TM.L[2], nC = TM.C[1])
pl

fgh = pd.DataFrame({"G": [.5,.5]})
fgh
g = []

#TM.iloc[4] = 2
#M

# %%
for l2 in range(2,len(TM)+2):
    tempList = []
    for l1 in range(0,len(TM)):
        pl2 = pL(TM.L[l1], TM.C[l1])  #grabs the probability of lobbying based on t-5 lobbying
        pnLob = binom.pmf(int(TM.columns[l2][0]), nPPL, pl2) ## binomial distrobution, gets probability they lobied that many times
        pcat = pCat2(int(TM.columns[l2][0]))
        tempList.append(binom.pmf(int(TM.columns[l2][2]), int(TM.columns[l2][0]), pcat) * pnLob) ###Appends each value to list   ### This need to be fixed for double digits
    TM[TM.columns[l2]] = tempList

TM

# %% [markdown]
# for l1 in range(0,len(TM)):
#     pl2 = pL(TM.L[l1], TM.C[l1])  #grabs the probability of lobbying based on t-5 lobbying
#     for l2 in range(2,len(TM)+2):
#         pnLob = binom.pmf(int(TM.columns[l2][0]), nPPL, pl2) ## binomial distrobution, gets probability they lobied that many times
#         pcat = pCat2(int(TM.columns[l2][0]))
#         TM.loc[l1][l2] = 1#int(binom.pmf(int(TM.columns[l2][2]), int(TM.columns[l2][0]), pcat) * pnLob * 10000000000000000000)### This need to be fixed for double digits
# 
# TM

# %% [markdown]
# for l2 in range(2,len(TM)+2):
#     tempList = []
#     for l1 in range(0,len(TM)):
#         pl2 = pL(TM.L[l1], TM.C[l1])  #grabs the probability of lobbying based on t-5 lobbying
#         pnLob = binom.pmf(int(TM.columns[l2][0]), nPPL, pl2) ## binomial distrobution, gets probability they lobied that many times
#         pcat = pCat2(int(TM.columns[l2][0]))
#     TM.loc[l1][l2] = binom.pmf(int(TM.columns[l2][2]), int(TM.columns[l2][0]), pcat) * pnLob### This need to be fixed for double digits
# 
# TM

# %% [markdown]
# for l1 in range(0,len(TM)):
#     pl2 = pL(TM.L[l1], TM.C[l1])  #grabs the probability of lobbying based on t-5 lobbying
#     for l2 in range(2,len(TM)+2):
#         pnLob = binom.pmf(int(TM.columns[l2][0]), nPPL, pl2) ## binomial distrobution, gets probability they lobied that many times
#         pcat = pCat2(int(TM.columns[l2][0]))
#     TM.loc[l1][l2] = 1.5#int(binom.pmf(int(TM.columns[l2][2]), int(TM.columns[l2][0]), pcat) * pnLob * 10000000000000000000)### This need to be fixed for double digits
# 
# TM

# %%
#TM['s'] = TM.sum(axis = 1)

# %%
pL(2,1)

# %%
#TML = pd.DataFrame({'L': range(0,nPPL+1)}) #Transition Matrix
#for i in range(nPPL+1): ## Clear Matrix
#    TML[i] = 0

#for l1 in range(0,nPPL+1): # row
#    for l2 in range(0,nPPL+1): ## column
#        TML[l2][l1] = TM.loc[TM['L'] == l1, l2].sum() / (l1+1)
#TML




# %%
s = TM.sum(axis = 1) - TM["L"] - TM["C"]
s[50]
#Rows sum to 1


# %%
tempTM = TM.iloc[0:len(TM),2:len(TM)+2]
#tempTM = tempTM.div(tempTM.sum(axis=1), axis=0)
tempTM
s = tempTM.sum(axis = 1)
s[50]

# %%
mat = tempTM.to_numpy()
print(np.matrix(mat))

# %%
#mc = (mat, c)


# %%
st = mat
for i in range(0,100000):
    st = np.matmul(mat,st) #mat!
    #st = st/st.sum(axis=1)
    
st[0]

# %%
st2 = mat
for i in range(0,100):
    st2 = np.matmul(st2,st2)
    st2 = st2/st2.sum(axis=1)
    
st2[0] - st[0]

# %%
sd = pd.DataFrame({'L,C': TM.columns[2:], "prob": st[0]})
print(sd)


# %%
mat2 = mat.transpose()
mat2

# %%
st = mat2
for i in range(0,50):
    st = np.matmul(st,st)
    
st.transpose()[0]

# %%
linalg.eig(mat, left = True)[0]

# %%
linalg.eig(mat2, left = True)[0]

# %% [markdown]
# Dynamic Part

# %%
def it():
    t = len(LobH)
    LobH.append(0)
    CatH.append(0)
    ps = pSS1()
    for i in range(0,nPPL):
        if -T < (ps * -LC) + ((1-ps) * (-T - LC - (mF * np.random.normal(loc = 1, scale = .5)))):
            LobH[t] = LobH[t] + 1
            
    for i in range(0,LobH[t]):
        if random.random() < pCat():
            CatH[t] = CatH[t] + 1

# %%
for i in range(0,10000):
    it()


# %%
plt.plot(LobH)
plt.plot(CatH)

# %%
plt.hist(LobH)

# %%
plt.hist(CatH)

# %%
for i in range(0,9+1):
    print(i)


