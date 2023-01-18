# -*- coding: utf-8 -*-

from flask import Flask, request
import utils
import pandas as pd
import sys
import os
import pickle
import shutil

### Initialisation ###
app = Flask(__name__)
mo = utils.loadModelLightGBM(formatFile='pkl') #Nommé mo car une fonction nommée model existe déjà
cols = utils.loadColumnsOfModel()
minScore=0
maxScore=1
th = 0.53 #Nommé th car une fonction nommée threshold (seuil) existe déjà
MYDIR = os.path.dirname(__file__)
# formatOsSlash = '\\'
formatOsSlash = '/'
tmpDirName = 'tmpSplit'
tmpDir = formatOsSlash+tmpDirName+formatOsSlash


### app.route: routeurs nécessaires pour le projet ###
@app.route('/lightgbm/',methods=['POST'])
def lightgbm():
    return utils.modelPredict(mo,utils.restoreFromB64Str(request.args.get('data_b64_str')),th)

@app.route('/model/',methods=['POST'])
def model():
    return utils.loadModelLightGBM(formatFile='b64')

@app.route('/ratingSystem/',methods=['POST'])
def ratingSystem():
    '''
    Renvoie les détails du système de notation.
    Entrée : Rien
    Sortie:
    - Score minimum du système de notation;
    - Score maximum du système de notation;
    - Seuil du système de notation.
    '''
    print(f'minScore={minScore}', file=sys.stderr)
    print(f'maxScore={maxScore}', file=sys.stderr)
    print(f'th={th}', file=sys.stderr)
    return utils.convToB64((minScore,maxScore,th))

@app.route('/')
def helloworld():
    return '''<h1>Bienvenue sur la partie API du P7 DataScientist d'OpenClassrooms'</h1>'''

@app.route('/initSplit/',methods=['POST'])
def initSplit():
    global MYDIR
    global tmpDirName

    print('initSplit', file=sys.stderr)

    #Initialiser le dossier de destination
    #Le Supprimer s'il existe, avec tout ce qu'il contient
    shutil.rmtree(tmpDirName, ignore_errors=True)
    #Création d'un dissier temporaire
    if not os.path.exists(tmpDirName):
        os.makedirs(tmpDirName)
    return utils.convToB64(True)

@app.route('/merge/',methods=['POST'])
def splitN():
    global MYDIR
    global tmpDir
    pathFile = MYDIR+tmpDir+request.values.get("numSplit")+'.pkl'
    strToSave = request.values.get('txtSplit')
    
    print(f'Merge - numSplit={request.values.get("numSplit")}', file=sys.stderr)

    #Sauvegarde du contenu reçu dans un fichier pickle
    #Le fichier pickle est nommé après le numéro de séparation
    pickle.dump(strToSave, open(pathFile, 'wb'))
    return utils.convToB64(True)

@app.route('/endSplit/',methods=['POST'])
def endSplit():
    global MYDIR
    global tmpDir
    global tmpDirName
    txtB64Global = ''

    print('endSplit', file=sys.stderr)
    
    #Restaurer les données
    for i in range(5):
        pathFile = MYDIR+tmpDir+str(i)+'.pkl'
        #Ouverir le fichier pickle et attacher son contenu au text global
        txtB64Global += pickle.load(open(pathFile, 'rb'))
    

    #Restorer les données

    print(f'Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    
    #Decoder les données
    dataOneCustomer = utils.restoreFromB64Str(txtB64Global)
    
    #Création de dfOneCustomer
    dfOneCustomer = pd.DataFrame(data=dataOneCustomer, columns=cols)
    
    #Pour des raisons de sécurité le dossier temporaire est supprimé
    shutil.rmtree(tmpDirName, ignore_errors=True)
    
    #Intérrogation du modèle et retour des résultats
    return utils.modelPredict(mo,dfOneCustomer,th)


### app.route - fin ###

if __name__ == "__main__":
    app.run()