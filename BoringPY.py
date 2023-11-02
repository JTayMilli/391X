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
from collections import Counter
import quantecon as qe
from mpl_toolkits.mplot3d import Axes3D

# %%
## Base parameters
LC = 1 ## Lobby Cost
T = 10 ## Tax
mF = 10 ## Mean Fee of Getting Caught
nPPL = 11
R = nPPL * .25 # enforcement resources

LobH = [2] ## Lobby History, arbitrary preset
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
def getTM(numPPL = 10, tR = 3):
    R = tR
    nPPL = numPPL
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
    for l2 in range(2,len(TM)+2):
        tempList = []
        for l1 in range(0,len(TM)):
            pl2 = pL(TM.L[l1], TM.C[l1])  #grabs the probability of lobbying based on t-5 lobbying
            tL, tC = TM.columns[l2].split(",")
            tL = int(tL)
            tC = int(tC)
            pnLob = binom.pmf(tL, nPPL, pl2) ## binomial distrobution, gets probability they lobied that many times
            pcat = pCat2(tL)
            tempList.append(binom.pmf(tC, tL, pcat) * pnLob) ###Appends each value to list   ### This need to be fixed for double digits
        TM[TM.columns[l2]] = tempList
    
    return TM


# %%
def getTMR(tR = nPPL * .3):
    nPPL = 10
    R = tR
    TM = getTM(numPPL = nPPL, tR = R)
    return TM

# %%
getTM(tR = 3) - getTM(tR = 5)

# %%
def CollapsedTM(numPPL = 10):
    nPPL = numPPL
    TM = getTM(nPPL)
    cTM = TM.groupby("L").sum()
    lTM = pd.DataFrame({'L': range(0,nPPL+1)})
    for i in range(0, nPPL + 1):
        lTM[i] = cTM[[col for col in cTM.columns if col.split(",")[0] == str(i)]].sum(axis=1)
    lTM = lTM.div(lTM.L+1, axis=0)
    lTM.L = range(0,nPPL+1)
    return lTM

# %%
def ArrColTM():
    colTM = CollapsedTM()
    colTM = colTM.iloc[0:len(colTM), 1:len(colTM)+1]
    colTM = colTM.to_numpy()
    return colTM

# %%
ArrColTM()

# %%
sns.set_style ("darkgrid")
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)

x = range(nPPL)
y = range(nPPL)

data = ArrColTM()

hf = plt.figure()
ha = hf.add_subplot(projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data)

ha.set_xlabel('Ending #L')
ha.set_ylabel('Initial #L')
ha.set_zlabel('Probability')
ha.set_title("Transition Matrix Projection")

plt.show()

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
#TML = pd.DataFrame({'L': range(0,nPPL+1)}) #Transition Matrix
#for i in range(nPPL+1): ## Clear Matrix
#    TML[i] = 0

#for l1 in range(0,nPPL+1): # row
#    for l2 in range(0,nPPL+1): ## column
#        TML[l2][l1] = TM.loc[TM['L'] == l1, l2].sum() / (l1+1)
#TML




# %%
"""
s = TM.sum(axis = 1) - TM["L"] - TM["C"]
s[0]
#Rows sum to 1
"""

# %%
def StationaryDist(tTM = getTM()):
    TM = tTM
    nPPL = 10
    tempTM = TM.iloc[0:len(TM),2:len(TM)+2]
    #tempTM = tempTM.div(tempTM.sum(axis=1), axis=0)
    tempTM
    mat = tempTM.to_numpy()
    mc = qe.MarkovChain(mat)     #### test if markov chain
    statdist = mc.stationary_distributions
    if mc.is_irreducible:  ## transition matrix is irreducible
        return statdist[0]
    else: return "bruh, ur bad"

# %%
def StationaryDistR(R):
    return StationaryDist(tTM = getTMR(R))

# %%
getTMR(3) - getTMR(8)

# %%
StationaryDistR(3)

# %%
"""
st = mat
for i in range(0,10000):
    st = np.matmul(mat,st) #mat!
    #st = st/st.sum(axis=1)
    
st[0]
"""

# %%
"""
st2 = mat
for i in range(0,100):
    st2 = np.matmul(st2,st2)
    st2 = st2/st2.sum(axis=1)
    
st2[0] - st[0]
"""

# %%
sd = pd.DataFrame({'L': TM.L, 'C': TM.C, "prob": StationaryDist()})
print(sd)

# %%
def CollapsedSD(statDist = StationaryDist()):
    nPPL = 10
    sd = pd.DataFrame({'L': TM.L, 'C': TM.C, "prob": statDist})
    csd = sd.groupby("L").sum()
    csd = csd.drop(['C'], axis = 1) #prob still sums to 1
    return csd

# %%
def CollapsedSDR(R):
    return CollapsedSD(statDist = StationaryDistR(R))

# %%
StationaryDistR(3) - StationaryDistR(10)

# %%
csd = CollapsedSDR(3)

# %%
plt.plot(csd)

# %%
"""
templTM = lTM.iloc[0:len(lTM),1:len(lTM)+1]
#tempTM = tempTM.div(tempTM.sum(axis=1), axis=0)
#s = tempTM.sum(axis = 1)
#s[0]
lmat = templTM.to_numpy()
#print(np.matrix(lmat))

lst = lmat
for i in range(0,100000):
    lst = np.matmul(lmat,lst) #mat!
    #st = st/st.sum(axis=1)
    
lst[0]
"""

# %%
#linalg.eig(mat, left = True)

# %%
#linalg.eig(mat2, left = True)[0]

# %% [markdown]
# Dynamic Part

# %%
def it():
    t = len(LobH)
    LobH.append(0)
    CatH.append(0)
    ps = pSS1()
    for i in range(0,nPPL):
        if -T < (ps * -LC) + ((1-ps) * (-T - LC - (mF * np.random.normal(loc = 1, scale = 1)))):
            LobH[t] = LobH[t] + 1
            
    for i in range(0,LobH[t]):
        if random.random() < pCat():
            CatH[t] = CatH[t] + 1

# %%
def run(itr = 10000):
    for i in range(0,itr):
        it()


# %%
def set(LC = 1, T = 10, mF = 10, nPPL = 11, R = nPPL * .25, itr = 10000):
    LC ## Lobby Cost
    T ## Tax
    mF ## Mean Fee of Getting Caught
    nPPL
    R # enforcement resources

    LobH = [2] ## Lobby History, arbitrary preset
    CatH = [1] ## Catch history, arbitrary preset
    run(itr)

# %%
set(LC = 5, T = 10, mF = 10, nPPL = 100, R = nPPL * .25, itr = 10000)

# %%
plt.plot(LobH)
plt.plot(CatH)

# %%
plt.hist(LobH, range = [0,nPPL+1])



# %%
uni = range(0, nPPL+1)
nL = Counter(LobH).keys() # equals to list(set(words))
Lfreq = Counter(LobH).values() # counts the elements' frequency

GraphProbF = sns.lineplot(x = nL, y = Lfreq) ## Graph probFTR
GraphProbF.set(xlim=(0,nPPL+1))

# %%
plt.plot(CollapsedSD())

# %%
plt.hist(CatH)

# %%
def ArrItrR():
    nPPL = 10
    R = 0
    RDis = [CollapsedSDR(R).prob.to_numpy()]
    for tR in range(1,10+1):
        R = tR * .1 ## .1 intervuls for Resources
        RDis = np.vstack([RDis, CollapsedSDR(R).prob.to_numpy()])
    return RDis

# %%
CollapsedSDR(3).prob.to_numpy() - CollapsedSDR(5).prob.to_numpy()

# %%
ArrItrR()

# %%
sns.set_style ("darkgrid")
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)

x = range(nPPL)
y = range(nPPL)

data = ArrItrR()

hf = plt.figure()
ha = hf.add_subplot(projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data)

ha.set_xlabel('nLobby')
ha.set_ylabel('R')
ha.set_zlabel('Probability')
ha.set_title("3D Stationary Disrobutions based on R")

plt.show()


