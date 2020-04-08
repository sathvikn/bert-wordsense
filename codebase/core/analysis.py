import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.set_cmap('Blues')

#Firebase functions, getting trial & subject data
def access_db():
    cred = credentials.Certificate('../data/wordsense-pilesort-firebase-adminsdk-3ipny-791a81e575.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://wordsense-pilesort.firebaseio.com'
    })
    ref = db.reference('polysemy_pilesort')
    return ref.get()

#Returns dataframe where rows = trials
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

#Returns dataframe where rows = participants
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

def get_num_trials(results):
    lst = []
    for l in results['lemma'].unique():
        num_words = len(results[results['lemma'] == l].index) / get_num_senses(l, db)
        lst.append({'type': l, 'num_trials': num_words, 'num_senses': get_num_senses(l, db)})
    return pd.DataFrame(lst)

#Represents subject's data for a word as a matrix
def get_subject_mtx(results, userID, word_type, trial_type):
    word_data = results[(results['userID'] == userID) & (results['lemma'] == word_type) & (results['trialType'] == trial_type)]
    result_mtx = []
    senses = word_data['sense']
    max_value = 0
    for i in range(len(word_data.index)):
        row = []
        for j in range(len(word_data.index)):
            dist = calculate_distance(word_data.iloc[i], word_data.iloc[j])
            if dist > max_value:
                max_value = dist
            row.append(dist)
        result_mtx.append(np.asarray(row))
    result_mtx = np.asarray(result_mtx)
    result_mtx = result_mtx / max_value
    return result_mtx, senses

#Plots matrices for a user's repeat trials (2x2)
def plot_repeat_trials(results, userID, subject_index = None):
    user_trials = results[results['userID'] == userID]
    repeat_types = user_trials[user_trials['trialType'] == 'repeat']['lemma'].unique()
    for l in repeat_types:
        original_result, senses = get_subject_mtx(results, userID, l, 'test')
        repeat_result, _ = get_subject_mtx(results, userID, l, 'repeat')
        my_eyes = plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title("Original")
        orig_img = ax1.imshow(original_result)
        annotate_mtx(original_result, orig_img, ax1, senses)
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title("Repeat")
        rep_img = ax2.imshow(repeat_result)
        annotate_mtx(repeat_result, rep_img, ax2, senses)
        pos = 2
        my_eyes.subplots_adjust(right = pos)
        my_eyes.suptitle("Subject " + subject_index + " Annotations for Repeated Type " + l, x = pos / 1.9)

def plot_all_repeats(results, users):
    for u in users:
        subject_index = str(users[users == u].index[0])
        plot_repeat_trials(results, u, subject_index)

#Plots all matrices for shared trials (words x subjects)
def plot_all_shared(results, users):
    shared_words = results[results['trialType'] == 'shared']['lemma'].unique()
    user_lst = users.tolist()
    grid = plt.GridSpec(len(shared_words), len(users), wspace = 0.2, hspace = 0.7, figure = plt.figure(figsize = (20, 20)))
    for i in range(len(shared_words)):
        for j in range(len(users)):
            user_result, senses = get_subject_mtx(results, user_lst[j], shared_words[i], 'shared')
            ax = plt.subplot(grid[i, j])
            im = plt.imshow(user_result, aspect = 1)
            write_text = False
            if j == 0:
                write_text = True
            annotate_mtx(user_result, im, ax, senses, write_text)

#Annotates a matrix with the senses and perceived distances
def annotate_mtx(result_mtx, im, ax, senses, write_text = True):
    threshold = im.norm(result_mtx.max())/2.

    if write_text:
        ax.set_xticks(np.arange(len(senses)))
        ax.set_yticks(np.arange(len(senses)))

        ax.set_xticklabels(senses)
        ax.set_yticklabels(senses)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    else:
        ax.set_xticklabels([''] * len(senses))
        ax.set_yticklabels([''] * len(senses))
    textcolors = ['black', 'white']
    for i in range(len(senses)):
        for j in range(len(senses)):
            square_color = textcolors[int(im.norm(result_mtx[i][j]) > threshold)]
            text = ax.text(j, i, np.round(result_mtx[i][j], 3),
                           ha="center", va="center", color=square_color)


#Helper functions

#Creates x and y coordinates
def segment_position_string(s):
    left, top = s.strip().split(";")
    x = float(left.split(":")[1].strip("px").strip(" "))
    y = float(top.split(":")[1].strip("px").strip(" "))
    return x, y
    
#Euclidean distance
def calculate_distance(r1, r2):
    s1 = np.array([r1['x'], r1['y']])
    s2 = np.array([r2['x'], r2['y']])
    return np.linalg.norm(s1 - s2)

def get_senses(db, word):
    return [k for k in db['inputs'][word] if k not in ['senses', 'type']]

def get_num_senses(w, db):
    return db['inputs'][w]['senses']

