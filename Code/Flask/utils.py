# -*- coding: utf-8 -*-

import os
import pickle
import base64
import numpy as np
# import urllib.request


def modelPredict(model, data, threshold):
    '''
        Renvoie la prédiction du modèle : 0 ou 1 selon le seuil
        et la valeur de probabilité donnée par le modèle.
    '''
    
    pP = model.predict_proba(data)[:,0].item()
    pE = int(np.where(pP<threshold,0,1))
    
    return convToB64(
        dict(
            predProba = pP,
            predExact = pE
            )
        )

def loadModelLightGBM(formatFile='b64'):
    '''
    Décoder et charger le modèle de Machine Learning 'model.pkl'
    Selon la valeur de <formatFile> :
        renvoie le modèle dans son format nominal;
        renvoie le modèle converti au format base-64 encodé en UTF-8.
    '''
    model = pickle.load(open(os.getcwd()+'/pickle/model.pkl', 'rb'))
    
    if formatFile == 'pkl':
        return model
    elif formatFile == 'b64':
        return convToB64(model)
    else:
        return model.class_weight

def loadColumnsOfModel():
    '''
    Décode et renvoie les colonnes attendues par
    le modèle d'apprentissage automatique utilisé dans ce projet.
    '''
    return pickle.load(open(os.getcwd()+'/pickle/cols.pkl', 'rb'))    

def convToB64(data):
    '''
    En entrée : les données de n'importe quel type.
    La fonction convertit les données en base-64 puis la chaîne résultante est encodée en UTF-8.
    En Sortie : le résultat obtenu.
    '''
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')

def restoreFromB64Str(data_b64_str):
    '''
    Entrée : les données converties en Base64 puis encodées en UTF-8.
          Idéalement, les données de la fonction convToB64.
    La fonction restaure les données encodées dans leur format originelle.
    Sortie : les données restaurées.
    '''
    return pickle.loads(base64.b64decode(data_b64_str.encode()))
