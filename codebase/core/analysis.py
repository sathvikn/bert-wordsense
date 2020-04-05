import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#Firebase functions, getting trial & subject data
def access_db():
    cred = credentials.Certificate('../data/wordsense-pilesort-firebase-adminsdk-3ipny-791a81e575.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://wordsense-pilesort.firebaseio.com'
    })
    ref = db.reference('polysemy_pilesort')
    return ref.get()

def get_trial_data(db):
    trials = db['trials']
    df_rows = []
    for trialID in trials:
        t = trials[trialID]
        for sense_name in t['response']:
            try:
                left, top = segment_position_string(t['response'][sense_name])
            except:
                left, top = -1, -1
            row = {'trialID': trialID, 'userID': t['userID'], 'trialIndex': t['trialIndex'], 'trialType': t['trialType'], 'prevChanged': t['timesPrevTrialsChanged'],
            'lemma': t['inputWord'], 'sense': sense_name, 'x': left, 'y': top,
            }
            df_rows.append(row)
    return pd.DataFrame(df_rows)

def get_participant_data(db):
    node = db['subjectInfo']
    df_rows = []
    for userID in node:
        user_data = node[userID]
        if 'endedAt' in user_data:
            elapsedTimeSec = (user_data['endedAt'] - user_data['startedAt']) / 1000 #Time for trial in seconds
        else:
            elapsedTimeSec = -1
        row = {"userID": userID, "workerID": user_data['qualtricsWorkerID'], "userIP": user_data['ipAddress'], 
        'completedTask': user_data['completed'], 'timeTaken': elapsedTimeSec
        }
        df_rows.append(row)
    return pd.DataFrame(df_rows)

#Functions to calculate and plot confusion matrices from trial results
def segment_position_string(s):
    left, top = s.strip().split(";")
    x = float(left.split(":")[1].strip("px").strip(" "))
    y = float(top.split(":")[1].strip("px").strip(" "))
    return x, y

def calculate_distance(r1, r2):
    s1 = np.array([r1['x'], r1['y']])
    s2 = np.array([r2['x'], r2['y']])
    return np.linalg.norm(s1 - s2)
    
def get_subject_mtx(results, userID, word_type, trial_type):
    word_data = results[(results['userID'] == userID) & (results['lemma'] == word_type) & (results['trialType'] == trial_type)]
    result_mtx = []
    senses = word_data['sense']
    for i in range(len(word_data.index)):
        row = []
        for j in range(len(word_data.index)):
            row.append(calculate_distance(word_data.iloc[i], word_data.iloc[j]))
        result_mtx.append(np.asarray(row))
    result_mtx = np.asarray(result_mtx)
    return result_mtx, senses


def plot_mtx(result_mtx, senses):
    fig, ax = plt.subplots()
    im = ax.imshow(result_mtx)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(senses)))
    ax.set_yticks(np.arange(len(senses)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(senses)
    ax.set_yticklabels(senses)
    
    threshold = im.norm(result_mtx.max())/2.
        

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    textcolors = ['black', 'white']
    # Loop over data dimensions and create text annotations.
    for i in range(len(senses)):
        for j in range(len(senses)):
            square_color = textcolors[int(im.norm(result_mtx[i][j]) < threshold)]
            text = ax.text(j, i, np.round(result_mtx[i][j], 3),
                           ha="center", va="center", color=square_color)

    #fig.tight_layout()
    plt.show()


