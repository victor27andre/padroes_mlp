
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tkinter import *
from math import exp

tempoInicial = time.time()

##################### GERAL #####################

learning_rate = 0.1
epochs = 3500
eqm = list()
datasetTest = pd.read_csv('padroesmlpteste.csv')
datasetTrein = pd.read_csv('padroesmlptrein.csv')
###### DATASET TREINO #####
inputs = datasetTrein.iloc[:, 0:4].values
outputs = datasetTrein.iloc[:, 4:7].values
###### DATASET TESTE #####
inputsTest = datasetTest.iloc[:, 0:4].values
outputsTest = datasetTest.iloc[:, 4:7].values


###### PESOS #####
weightA = np.random.rand(15, 5)
weightB = np.random.rand(3, 16)


def factory(x):
    func = lambda x: (1)/(1 + exp(-x))
    vfunc = np.vectorize(func)
    return vfunc(x)

def reFactory(x):
    func = lambda x: factory(x)*(1-factory(x))
    vfunc = np.vectorize(func)
    return vfunc(x)


def rounding(x):
    h = np.zeros(3)
    for j in range(3):
        h[j] = 1 if x[j]>=0.5 else 0
    return h



for run in range(epochs):
    print(f'Run epoch: {run + 1}')
    eqms = list()
    
    for x, d in zip(inputs, outputs):
        x = np.append(-1,x)
        
        ij1 = np.dot(weightA, x)
        y1 = factory(ij1)
        y1 = np.append(-1, y1)
        
        ij2 = np.dot(weightB, y1)
        y2 = factory(ij2)
        eqms.append(sum(0.5 * (d-y2)**2))
        ay2 = rounding(y2)      
        d2 = np.zeros(ij2.shape)

        for j in range(d2.shape[0]):
            d2[j] = (d[j] - y2[j]) * reFactory(ij2[j])
        
        for j in range(weightB.shape[0]):
            for i in range(weightA.shape[1]):
                weightB[j,i] = weightB[j,i] + learning_rate * d2[j] * y1[i]       
        d1 = np.zeros(ij1.shape)

        for j in range(d1.shape[0]):
            for k in range(0, weightB.shape[0]):
                d1[j] += d2[k] * weightB[k, j+1]
            d1[j] *= reFactory(ij1[j])
            
        for j in range(weightA.shape[0]):
            for i in range(weightA.shape[1]):
                weightA[j,i] = weightA[j,i] + learning_rate*d1[j]*x[i]
    
    eqm.append(sum(eqms)/len(inputs))

linha = 0



# Cria formulario
formulario = Tk()
formulario.title = "Desenvolvimento Aberto"
 
# Evento CB on click

 
# Evento do botão
def callback():
    if (r1.get() == 1):
        n1 = float(campo1.get())
        n2 = float(campo2.get())
        total =  n1 + n2
        texto.insert(END, campo1.get() + " + " + campo2.get() + " = " + str(total) + "\n")
 
    if (r1.get() == 2):
        n1 = float(campo1.get())
        n2 = float(campo2.get())
        total =  n1 - n2
        texto.insert(END, campo1.get() + " - " + campo2.get() + " = " + str(total) + "\n")
 
    if (r1.get() == 3):
        n1 = float(campo1.get())
        n2 = float(campo2.get())
        total =  n1 * n2
        texto.insert(END, campo1.get() + " * " + campo2.get() + " = " + str(total) + "\n")
 
    if (r1.get() == 4):
        n1 = float(campo1.get())
        n2 = float(campo2.get())
        total =  n1 / n2
        texto.insert(END, campo1.get() + " / " + campo2.get() + " = " + str(total) + "\n")
 
# Define variavel para status do RadioButton
r1 = IntVar()
 
# Cria um novo label
rotulo = Label(formulario, text = "Classificador de Padroes")
 
# Identa linhas usando o caracter continuacao de linua
 
# Cria os Componentes

 
rotulo1 = Label(formulario, text = "Tempo Total(ms)")
 
campo1 = Entry(formulario)
 
campo2 = Entry(formulario)
 
texto = Text(formulario, height = 10, width = 50)
 
# Adiciona Componentes no Grid
rotulo.grid(sticky=W)
 

 
rotulo1.grid(row=1,sticky=W)
campo1.grid(row=1, columnspan=2)
 
texto.grid(row=5)


###### TIMER #####
tempoFinal = time.time()
tempoTotal = tempoFinal - tempoInicial

#print('TEMPO TOTAL do processamento: {:.3f}s'.format(tempoTotal))   
 

#########################


for zInputTest in zip(inputsTest):
    
    zInputTest = np.append(-1,zInputTest)
        
    ij1 = np.dot(weightA, zInputTest)
    y1 = factory(ij1)
    y1 = np.append(-1, y1)
        
    ij2 = np.dot(weightB, y1)
    y2 = factory(ij2)
    ay2 = rounding(y2) 
    linha+=1
    texto.insert(END,f'LINHA {linha}: {ay2}\n')

    #print(f'LINHA {linha}: {ay2}')
campo1.insert(END,f'{tempoTotal}')
#print('TEMPO TOTAL do processamento: {:.3f}s'.format(tempoTotal))  
#texto.insert(END,f'{tempoTotal}')

###########
 
    
###### GRAFICO ######  
plt.figure(figsize = (10,5))
plt.plot(eqm)
plt.title('Erro do Quadrado Médio - {:.3f}s'.format(tempoTotal))
plt.show()

mainloop()

