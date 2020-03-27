import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pandas as pd

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
            left, top = segment_position_string(t['response'][sense_name])
            row = {'trialID': trialID, 'userID': t['userID'], 'trialIndex': t['trialIndex'], 'trialType': t['trialType'],
            'lemma': t['inputWord'], 'sense': sense_name, 'x': left, 'y': top,
            }
            df_rows.append(row)
    return pd.DataFrame(df_rows)

def get_participant_data(db):
    node = db['subjectInfo']
    df_rows = []
    for userID in node:
        user_data = node[userID]
        elapsedTimeSec = (user_data['endedAt'] - user_data['startedAt']) / 1000 #Time for trial in seconds
        row = {"userID": userID, "workerID": user_data['qualtricsWorkerID'], 'completedTask': user_data['completed'], 'timeTaken': elapsedTimeSec
        }
        df_rows.append(row)
    return pd.DataFrame(df_rows)

def segment_position_string(s):
    left, top = s.split(";")
    x = int(left.split(" ")[1].strip("px"))
    y = int(top.split(" ")[1].strip("px"))
    return x, y

