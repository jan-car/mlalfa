"""Beispieldatensatz Acute InflammationsData Set aus Abschnitt 4.2.2:
1. Temperatur des Patienten in Grad Celsius
2. Auftreten von Übelkeit als Boolean-Wert
3. Lendenschmerzen als Boolean-Wert
4. Urinschub (kontinuierlicher BedarfWasserzulassen) als Boolean-Wert
5. Blasenschmerzen als Boolean-Wert
6. Beschwerden an der Harnröhre wie Juckreiz oder Schwellung des Harnröhrenaustritts
7. Krankheitsbild Harnblasenentzündung als Boolean-Wert
8. Krankheitsbild Nierenentzündung mit Ursprung im Nierenbecken als Boolean-Wert
"""

# %% [markdown]
# # Imports:

# %%
import os
import logging
import numpy as np
from mlalfa.naive_bayes_classification import NaiveBayesNominalEstimator
from IPython import get_ipython
if type(get_ipython()).__name__ == 'ZMQInteractiveShell':  # IPython Notebook!
    get_ipython().run_line_magic('matplotlib', 'inline')  # '%matplotlib inline'
    get_ipython().run_line_magic('load_ext', 'autoreload')  # '%load_ext autoreload'
    get_ipython().run_line_magic('autoreload', '2')  # '%autoreload 2' (reloads everything)
# Change working directory to file location:
os.chdir(R'C:\Users\Jan\Projects\mlalfa\scripts')
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
np.random.seed(45)


# %% [markdown]
# # Data conversion to csv:
# %%
fString = open(os.path.join('data', 'week1', 'diagnosis.data'), 'r')  # Originale Rohdaten
fFloat = open(os.path.join('data', 'week1', 'diagnosis.csv'), 'w')  # Daten als 0 und 1 im csv Format
for line in fString:
    line = line.replace(",", ".")
    line = line.replace("\t", ",")
    line = line.replace("yes", "1")
    line = line.replace("no", "0")
    line = line.replace("\r\n", "\n")
    fFloat.write(line)
fString.close()
fFloat.close()


# %% [markdown]
# # Extract Features and Labels / Aufteilen in Test- und Trainingsdaten:
# %%
dataset = np.loadtxt(os.path.join('data', 'week1', 'diagnosis.csv'), delimiter=",")  # Erneutes Einlesen
X = dataset[:, 1:6]  # erste Spalte (Temperatur) wird weggelassen, letzte Spalte auch (Labels)
Y = dataset[:, 6]  # letzte Spalte sind Labels (gibt eigentlich zwei Label, wir nehmen nur das erste)
allData = np.arange(0, X.shape[0])  # indices of all data records
# Randomly pick indices of 20% of the data records
iTesting = np.random.choice(X.shape[0], int(X.shape[0]*0.2), replace=False)
iTraining = np.delete(allData, iTesting)  # training indices
dataRecords = len(iTraining)
XTrain = X[iTraining, :]
YTrain = Y[iTraining]
XTest = X[iTesting, :]
YTest = Y[iTesting]


# %% [markdown]
# # Count Co-Occurences
# %%
# PXI: Probability of X(symptom/feature) under the assumption of I (diagnosis/label)
# Shape: (diagnosis is True/False, the symptoms, symptom is True/False)
PXI = np.zeros((2, XTrain.shape[1], 2))
for k in range(X.shape[1]):
    # Number of patients with symptom k, who are diagnosed as ill:
    PXI[1, k, 1] = np.sum(np.logical_and(XTrain[:, k], YTrain))
    # Number of patients that don't show symptom k, who are diagnosed as ill:
    PXI[1, k, 0] = np.sum(np.logical_and(np.logical_not(XTrain[:, k]), YTrain))
    # Number of patients that show symptom k, who are not diagnosed as ill:
    PXI[0, k, 1] = np.sum(np.logical_and(XTrain[:, k], np.logical_not(YTrain)))
    # Number of patients that don't show symptom k, who are not diagnosed as ill:
    PXI[0, k, 0] = np.sum(np.logical_not(np.logical_or(XTrain[:, k], YTrain)))  # DeMorgan: not A & not B = not(A or B)
# Turn into probability (+1/2 makes sure we don't have zeros for divisions later!)
PXI = (PXI + 1/2) / (dataRecords+1)
# Probability of diagnosis overall:
PI = np.zeros(2)
PI[1] = np.sum(YTrain)
PI[0] = dataRecords - PI[1]
PI = PI / dataRecords  # now it's a probability!


# %% [markdown]
# # Predict function:
# %%
def predictNaiveBayesNominal(x):
    P = np.zeros_like(PI)
    allofthem = np.arange(XTrain.shape[1])
    for i in range(len(PI)):
        P[i] = np.prod(PXI[i, allofthem, x])*PI[i]
    denominator = np.sum(P)
    P = P/denominator
    choosenClass = np.argmax(P)
    return choosenClass


# %% [markdown]
# # This time not as a function to get some insight into the variables:
# %%
P = np.zeros_like(PI)  # probability that new patient has illness or not
allofthem = np.arange(XTrain.shape[1])  # [0, 1, 2, 3, 4]: indices of the symptoms, : would work as well...
for i in range(len(PI)):
    P[i] = np.prod(PXI[i, allofthem, XTest[1, :].astype(int)])*PI[i]  # Bayes Rule counter!
denominator = np.sum(P)  # Bayes Rule denominator!
P = P/denominator  # Combined! Denotes probability of patient [being ill, being not ill]
choosenClass = np.argmax(P)  # Choose the option that's more likely!

# %% [markdown]
# # Check how many predictions are correct and how many are wrong:
# %%
correct = np.zeros(2)
incorrect = np.zeros(2)

y_predict = []
for i in range(XTest.shape[0]):  # go over rows/records of patients in the Test sample:
    klasse = predictNaiveBayesNominal(XTest[i, :].astype(int))
    y_predict.append(klasse)
    if klasse == YTest[i]:
        correct[klasse] += 1
    else:
        incorrect[klasse] += 1

print(f"Von {XTest.shape[0]:g} Testfaellen wurden {np.sum(correct):g} richtig",
      f"und {np.sum(incorrect):g} falsch klassifiziert")


# %% [markdown]
# # Use self-written class:
# %%


est_non_vec = NaiveBayesNominalEstimator(vectorized=False)
est = NaiveBayesNominalEstimator()
est_non_vec.fit(XTrain, YTrain)
est.fit(XTrain, YTrain)
print(est.predict(XTest))  # Vectorized version
print(est_non_vec.predict(XTest))  # Non-vectorized version
print(YTest.astype(int))  # Test results
















# %%
T = dataset[:,0]
trueIndex = np.flatnonzero(YTrain==1)
falseIndex = np.flatnonzero(YTrain==0)
muApproxTrue = np.sum(T[trueIndex])/trueIndex.shape[0]
sgApproxTrue = np.sum( (T[trueIndex]-muApproxTrue)**2 ) / (trueIndex.shape[0] -1)
muApproxFalse = np.sum(T[falseIndex])/falseIndex.shape[0]
sgApproxFalse = np.sum( (T[falseIndex]-muApproxFalse)**2 ) / (falseIndex.shape[0] -1)

def Gausverteilung(x,mu,sigma):
    y = np.exp(-0.5*( (x-mu)/sigma)**2 )/(sigma*np.sqrt(2*np.pi))
    return(y)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(131)
ax.hist(T[:],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur');
ax.set_ylabel('Wahrscheinlichkeit')
Tplot = np.arange(33,44,0.05)
ax.plot(Tplot,Gausverteilung(Tplot,muApproxTrue,sgApproxTrue),'k:')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxFalse,sgApproxFalse),'k-.')
ax.set_ylim([0,0.8])
ax.set_title('Alle Trainingsdaten')
ax = fig.add_subplot(132)
ax.hist(T[falseIndex],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxFalse,sgApproxFalse),'k-.')
ax.set_ylim([0,0.8])
ax.set_title('Negative Diagnose')
ax = fig.add_subplot(133)
ax.hist(T[trueIndex],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxTrue,sgApproxTrue),'k:')
ax.set_ylim([0,0.8])
ax.set_title('Positive Diagnose')
plt.tight_layout()
plt.show(block=False)

def predictNaiveBayesMixed(x,T,muTrue,sigmaTrue,muFalse,sigmaFalse):
    P = np.zeros_like(PI)
    allofthem = np.arange(XTrain.shape[1])
    P[0] = np.prod(PXI[0,allofthem,x])*PI[0]
    P[1] = np.prod(PXI[1,allofthem,x])*PI[1]
    P[0] = P[0] * Gausverteilung(T, muFalse,sigmaFalse)
    P[1] = P[1] * Gausverteilung(T, muTrue,sigmaTrue)
    choosenClass = np.argmax(P)
    return choosenClass

TTest = T[iTesting]
def TestNaiveBayesMixed(muTrue,sigmaTrue,muFalse,sigmaFalse):
    correct   = np.zeros(2); incorrect = np.zeros(2)
    for i in range(XTest.shape[0]):
        klasse = predictNaiveBayesMixed(XTest[i,:].astype(int),TTest[i],muTrue,sigmaTrue,muFalse,sigmaFalse)
        if klasse == YTest[i]:
            correct[klasse] = correct[klasse] +1
        else:
            incorrect[klasse] = incorrect[klasse] +1
    return(correct, incorrect)

(correct, incorrect) =  TestNaiveBayesMixed(muApproxTrue,sgApproxTrue, muApproxFalse, sgApproxFalse)
print("Von %d Testfaellen wurden %d richtig und %d falsch klassifiziert" % (XTest.shape[0],np.sum(correct),np.sum(incorrect) ))

keineDiagnose = np.logical_not(np.logical_or(dataset[iTraining,7],YTrain))
index = np.flatnonzero(keineDiagnose)
muApprox = np.sum(T[index])/index.shape[0]
sgApprox = np.sum( (T[index]-muApprox)**2 ) / (index.shape[0] -1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(T[index],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur');
ax.set_ylabel('Wahrscheinlichkeit')
ax.plot(Tplot,Gausverteilung(Tplot,muApprox,sgApprox),'k')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxTrue,sgApproxTrue),'k:')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxFalse,sgApproxFalse),'k-.')

# %%

