import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from sklearn.manifold import MDS
from scipy import stats
from nltk.corpus import wordnet
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import json
plt.set_cmap('Blues')


#Firebase functions, getting trial & subject data
def access_db(use_fb = False):
    """
    Returns a JSON of the data on Firebase
    """
    if use_fb:
        cred = credentials.Certificate('../data/wordsense-pilesort-firebase-adminsdk-3ipny-791a81e575.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://wordsense-pilesort.firebaseio.com'
        })
        ref = db.reference('polysemy_pilesort')
        
        return ref.get()
    else:
        with open('../data/expt_data_scrubbed.json', 'r') as f:
            db = json.load(f)
        return db['polysemy_pilesort']

#Returns dataframe where rows = trials
def get_trial_data(db):
    """
    db - JSON of Firebase data

    Output:
    Pandas Dataframe with one row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
    trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
    prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
    lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
    x, y- coordinates of the box participant placed, in pixels

    """
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
    """
    db- JSON of Firebase data

    Output- Pandas Dataframe with one row for each participant. Columns: userID- UID for participant (primary key), workerID- participant's worker ID from RPP,
    userIP- hashed IP address, completedTask- 0 or 1 for if participant completed task, timeTaken- time partipant spent on task 
    """
    node = db['subjectInfo']
    df_rows = []
    for userID in node:
        user_data = node[userID]
        if 'endedAt' in user_data:
            elapsedTimeSec = (user_data['endedAt'] - user_data['startedAt']) / 1000 #Time for trial in seconds
        else:
            elapsedTimeSec = -1
        row = {"userID": userID,
        'completedTask': user_data['completed'], 'timeTaken': elapsedTimeSec
        }
        df_rows.append(row)
    return pd.DataFrame(df_rows)

#Tables with metadata
def get_num_trials(results):
    """
    Input:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels

    Output:
    TODO: what's the diff between this and lemma_counts?
    Pandas dataframe where every row is a type. Columns: type- word_pos, num_trials- number of times this word was shown to a participant,
    num_senses- number of senses for type
    """
    lst = []
    for l in results['lemma'].unique():
        num_words = len(results[results['lemma'] == l].index) / get_num_senses(l, db)
        lst.append({'type': l, 'num_trials': num_words, 'num_senses': get_num_senses(l, db)})
    return pd.DataFrame(lst)

def display_sense_definitions(results, trial_type):
    """
    Input:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
    trial_type- one of ['training', 'shared', 'test', 'repeat']

    Output:
    Pandas dataframe where each row is a word sense. Columns: Sense- word_pos_number, Type- word_pos, Definition- definition of sense in WordNet
    """
    shared_trials = results[results['trialType'] == trial_type]
    pd.set_option('display.max_colwidth', 200)

    sense_defns = pd.DataFrame({"Sense": shared_trials['sense'],
        "Type": shared_trials['lemma'],
        "Definition": shared_trials['sense'].apply(wordnet_defn)}).drop_duplicates()
    sense_defns['Definition'] = sense_defns['Definition']
    return sense_defns


def get_time_and_changes(results, user_df):
    """
    Inputs:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels

    user_df- Pandas dataframe with schema of get_participant_data's output:
        One row for each participant. Columns: userID- UID for participant (primary key), workerID- participant's worker ID from RPP,
    userIP- hashed IP address, completedTask- 0 or 1 for if participant completed task, timeTaken- time partipant spent on task 

    Outputs:
    time- Pandas Dataframe with one row for each participant. Columns: userID- UID for participant (primary key), timeTaken- time partipant spent on task,
    prevChanged- amount of changes to the schema the participant made
    """
    changed = results[['userID', 'lemma', 'prevChanged']].groupby(['userID', 'lemma']).agg(max).reset_index()
    user_changes = changed.groupby('userID').agg(sum).reset_index()
    time = user_df[['userID', 'timeTaken']]
    time['changes'] = user_changes['prevChanged'].values
    return time

#Represents subject's data for a word as a matrix
def get_subject_mtx(results, userID, word_type, trial_type):
    """
    Inputs:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
    userID- string, UID for a participant
    word_type- string, word_pos
    trial_type- one of ['training', 'shared', 'test', 'repeat']

    Outputs:
    result_mtx- Numpy matrix of pairwise distance between tokens participant with user_id placed for the senses of word_type
    senses- list of senses of word_type (word_pos_number)
    """
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
    result_mtx = result_mtx / max_value #normalize between 0 and 1
    return result_mtx, senses

#Plots matrices for a user's repeat trials (2x2)
def repeat_correlations(results, userID, subject_index = None, plot = False):
    """
    Inputs:
        results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
        userID- string, UID for a participant
        subject_index- integer assigned to a subject (used only when plotting), chose this over UID for brevity
        plot- if true, plots a subject's judgement matrices for repeated types (two repeated trials, one test trial) 
    
    Outputs:
    user_orig- list of two distance matrices for the original trials
    user_repeat- list of two distance matrices for the same two words, repeated
    """
    user_trials = results[results['userID'] == userID]
    repeat_types = user_trials[user_trials['trialType'] == 'repeat']['lemma'].unique()
    user_orig = []
    user_repeat = []
    for l in repeat_types:
        original_result, senses = get_subject_mtx(results, userID, l, 'test')
        repeat_result, _ = get_subject_mtx(results, userID, l, 'repeat')
        if plot:
            fig = plt.figure()
            ax1 = plt.subplot(1, 2, 1)
            ax1.set_title("Original")
            orig_img = ax1.imshow(original_result)
            annotate_mtx(original_result, orig_img, ax1, senses)
            ax2 = plt.subplot(1, 2, 2)
            ax2.set_title("Repeat")
            rep_img = ax2.imshow(repeat_result)
            annotate_mtx(repeat_result, rep_img, ax2, senses)
            pos = 2
            fig.subplots_adjust(right = pos)
            r = mtx_correlation([original_result], [repeat_result])[0]
            fig.suptitle("Subject " + subject_index + " Annotations for Repeated Type " + l + " (r = " + str(np.round(r, 2)) + ")", x = pos / 1.9)
        user_orig.append(np.array(original_result))
        user_repeat.append(np.array(repeat_result))
    return user_orig, user_repeat

def group_consistency(results, users, random_baseline = False, exclude = []):
    """
    Input:
        results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
        users- sequence of userID strings
        random_baseline- if true, compares users to a random subject
        exclude- list of word_pos types to be excluded from calculation

        TODO: Exclude the user correlations under random_baseline? (low priority)
    Output:
        list of hold-one out correlations for each user (or for a random token assignment) 
    """
    #Returns a score for each user
    #"for each pairwise relationship, take the average of all participants except one"
    #shared_results = results[results['trialType'] == 'shared']
    hoo_corrs = []
    for u in users:
        subject_index = str(users[users == u].index[0])
        user_corr = hoo_corr(results, u, exclude)
        hoo_corrs.append(user_corr)
    if random_baseline:
        random_corr = random_vs_all(results)
        hoo_corrs.append(random_corr)
    return hoo_corrs

#hold one out correlation
def hoo_corr(results, userID, exclude):
    """
    Input:
        results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
        userID- string that is the participant's UID
        exclude- list of word_pos types to be excluded from calculation
    
    Output:
        Computes the Spearman rank correlation between the participant's judgements and the averaged distance matrix over all other participants for each type
    """
    user_results = [] #array of participant's matrices
    avg_results = [] #array of averaged responses for all types, done over all participants except the one with userID
    for l in results['lemma'].unique():
        if l not in exclude:
            held_out_results = results[results['userID'] != userID]
            user_lst = held_out_results['userID'].unique().tolist()
            avg_with_others = mean_distance_mtx(held_out_results, l, 'shared', user_lst)
            avg_results.append(avg_with_others)
            user_result, _ = get_subject_mtx(results, userID, l, 'shared')
            user_results.append(user_result)
    return mtx_correlation(user_results, avg_results)[0]

def user_vs_user_shared(results, user1, user2):
    """
    Input: 
        results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
        
        user1 and user2- userIDs (Strings that are UIDs for each participant)

    Output:
        Correlation between results for u1 and u2 for the Shared tasks
    """
    u1_results = []
    u2_results = []
    for l in results.lemma.unique():
        u1_word, _ = get_subject_mtx(results, user1, l, 'shared')
        u2_word, _ = get_subject_mtx(results, user2, l, 'shared')
        u1_results.append(u1_word)
        u2_results.append(u2_word)
    return mtx_correlation(u1_results, u2_results)[0]

def my_correlations(participants, trials, results, userids):
    """
    Input: 
        participants- Pandas dataframe with schema of get_participant_data's output:
        One row for each participant. Columns: userID- UID for participant (primary key), workerID- participant's worker ID from RPP,
        userIP- hashed IP address, completedTask- 0 or 1 for if participant completed task, timeTaken- time partipant spent on task 
        
        trials- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels

        results- Same schema as trials, results for everyone in userids on shared trials
        userids- List of strings that correspond to participants' UIDs

    Output:
        List of correlations of each user in userids against my placements for the shared data (a gold standard)
    """
    complete = participants[participants['completedTask'] == 1]
    my_userid = complete.iloc[1]['userID']
    gt_results = trials[trials['userID'] == my_userid]
    shared_plus_gt = pd.concat([results, gt_results])
    shared_plus_gt = shared_plus_gt[shared_plus_gt['trialType'] == 'shared']
    return [user_vs_user_shared(shared_plus_gt, u, my_userid) for u in userids]

def random_vs_all(results):
    """
    Inputs:
        results- Pandas dataframe with schema of get_trial_data's output:
            One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
            trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
            prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
            lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
            x, y- coordinates of the box participant placed, in pixels

    Outputs:
        Correlation between the averaged results for shared trials and a random token arrangement

    """
    user_results = []
    avg_results = []
    for l in results['lemma'].unique():
        user_lst = results['userID'].unique().tolist()
        avg_results.append(mean_distance_mtx(results, l, 'shared', user_lst))
        user_results.append(create_random_symmetric_mtx()) #All of these words have 3 senses
    return mtx_correlation(user_results, avg_results)[0]

def all_repeats(results, users, random_baseline = False, db = None, plot = False):
    """
    Inputs:
        results- Pandas dataframe with schema of get_trial_data's output:
            One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
            trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
            prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
            lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
            x, y- coordinates of the box participant placed, in pixels
        users- list of participant userIDs as strings
        random_baseline- whether the trials should be compared against random placements (boolean). If true, compares to two pairs of random arrangements 
        db- JSON of experiment results
        plot- whether the matrices should be plotted (boolean)

    Output:
    user_corrs- list of correlations for repeat trials (for users or for random placements of tokens )

    """
    all_orig = []
    all_repeat = []
    user_corrs = []
    for u in users:
        subject_index = str(users[users == u].index[0])
        user_orig, user_repeat = repeat_correlations(results, u, subject_index, plot = plot)
        user_corr = mtx_correlation(np.asarray(user_orig), np.asarray(user_repeat))[0]
        user_corrs.append(user_corr)
        #print("User", subject_index, " Correlation ", user_corr)
        for m in user_orig:
            all_orig.append(m)
        for m in user_repeat:
            all_repeat.append(m)
    if random_baseline:
        first_trial_dim = random_num_senses(db)
        second_trial_dim = random_num_senses(db)
        random_orig = np.array([create_random_symmetric_mtx(first_trial_dim), create_random_symmetric_mtx(second_trial_dim)])
        random_repeat = np.array([create_random_symmetric_mtx(first_trial_dim), create_random_symmetric_mtx(second_trial_dim)])
        random_corrs = mtx_correlation(random_orig, random_repeat)[0]
        user_corrs.append(random_corrs)
        #print("Random Baseline", random_corrs)

    #print("Correlation of all original vs. repeat trials", mtx_correlation(all_orig, all_repeat))
    return user_corrs

def plot_consistency_hist(randoms, parts, title, legend = True):
    """
    Input
    randoms- array of self or group consistency scores computed for random placements of data
    parts- participants' self or group consistency score
    title- plot title

    Distplot of random scores, participants' scores are vertical lines
    """
    sns.distplot(randoms)
    fmt_color = lambda num: "C" + str(num)
    colors = [fmt_color(i) for i in range(len(parts))]
    for i in range(len(parts)):
        plt.axvline(parts[i], c = colors[i], label = i)
    if legend:
        plt.legend(title = 'Subject Index')
    plt.title(title)
    plt.xlabel("Spearman Correlation")

def simulate_self_correlation(num_trials, db):
    """
    Inputs: num_trials- number of simulations (large integer like 1000)
    db- JSON from Firebase

    Output: compute self consistency for two pairs of random arrangements
    """
    randoms_self = []
    for i in range(num_trials):
        first_trial_dim = random_num_senses(db)
        second_trial_dim = random_num_senses(db)
        random_orig = np.array([create_random_symmetric_mtx(first_trial_dim), create_random_symmetric_mtx(second_trial_dim)])
        random_repeat = np.array([create_random_symmetric_mtx(first_trial_dim), create_random_symmetric_mtx(second_trial_dim)])
        random_corrs = mtx_correlation(random_orig, random_repeat)[0]
        randoms_self.append(random_corrs)
    return randoms_self


def plot_all_shared(results, users):
    """
    Input:
        results- Pandas dataframe with schema of get_trial_data's output:
            One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
            trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
            prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
            lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
            x, y- coordinates of the box participant placed, in pixels
        users- list of participant userIDs as strings
    
    Output:
    Plots all matrices for shared trials in a grid with dimensions (words x subjects)
    """
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
    """
    Inputs:
    result_mtx: participant's relatedness/distance matrix
    im- result of Matplotlib's plt.imshow
    ax- Matplotlib axis
    senses- list of senses represented in the matrix (word_pos_number)
    write_text- if the matrix should be annotated with its senses

    Adds the numerical values of distances and optionally labels specifying the senses to the image of a relatedness matrix for a type
    """
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

#MDS functions
def mean_distance_mtx(results, lemma, trial_type, user_lst, normalize = False):
    """
    Input:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
    lemma- word_pos string
    trial_type- one of ['training', 'shared', 'test', 'repeat']
    user_lst- list of userID strings
    normalize- if true, divide by the largest value

    Output:
    Numpy matrix that averages participants' responses for lemma

    """
    shared_tensor = [] #Data for all the users
    for j in range(len(user_lst)):
        user_result, senses = get_subject_mtx(results, user_lst[j], lemma, trial_type)
        if len(user_result):
            shared_tensor.append(user_result)
    shared_tensor = np.asarray(shared_tensor)
    avg = np.mean(shared_tensor, axis = 0)
    if normalize:
        return avg / (np.max(avg))
    else:
        return avg

def plot_mds(word_means, word, mds_model, db, src):
    """
    Input:
    word_means- distance matrix for word
    word- wordform_pos string
    mds_model- SKLearn MDS object
    db- JSON of experiment responses from FB
    src- one of ['human', 'BERT'], part of the plot title

    Plots MDS for the senses of one word 
    """
    results = mds_model.fit_transform(word_means)
    x = results[:,0]
    y = results[:,1]
    senses = get_senses(db, word)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(senses):
        ax.annotate(txt, (x[i] + 0.01, y[i] + 0.01), fontsize = 12)
    plt.title("MDS over Averaged " + src + " Distances for " + word, fontsize = 14)

def plot_all_mds(results, users, trial_type, db):
    """
    Input:
        results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels

        users- list of userIDs whose data we are using
        trial_type- one of ['training', 'shared', 'test', 'repeat']
        db- JSON of data from Firebase

    Output:
    Plot of the MDS for all words in results in separate figures
    """
    data = results[results['trialType'] == trial_type]
    mds_model = MDS(n_components = 2, dissimilarity = 'precomputed')
    for l in data['lemma'].unique():
        word_means = mean_distance_mtx(results, l, trial_type, users)
        plot_mds(word_means, l, mds_model, db, "Reported")
        plt.savefig("../../results/figures/mds_" + l + '.png')

def plot_individual_mds(results, word, trial_type, users, db, sense_df):
    """
    Input:
        results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
        word- string of word_pos
        trial_type- one of ['training', 'shared', 'test', 'repeat']
        users- list of userID strings
        db- Firebase JSON
        sense_df- Pandas dataframe where each row is a word sense. Columns: Sense- word_pos_number, Type- word_pos, Definition- definition of sense in WordNet

    Output:
        Plots MDS for word, and displays a table with the sense definitions
    """
    mds_model = MDS(n_components = 2, dissimilarity = 'precomputed')
    user_lst = users.tolist()
    word_means = mean_distance_mtx(results, word, trial_type, user_lst)
    plot_mds(word_means, word, mds_model, db, "Reported")
    return sense_df[sense_df['Type'] == word]


#Helper/utility functions

def fb_to_local(fb_sense):
    """
    fb_sense- word_pos_number
    Returns word.pos.number because FB doesn't allow . characters for keys in JSON, even if that's how they are stored locally
    """
    parts = fb_sense.split('_')
    return '_'.join(parts[:len(parts) - 2]) + '.' + '.'.join(parts[-2:])

def segment_position_string(s):
    """
    s is a string with format "left: [x]px;top: [y]px", this function extracts the x and y coordinates for a token
    """
    left, top = s.strip().split(";")
    x = float(left.split(":")[1].strip("px").strip(" "))
    y = float(top.split(":")[1].strip("px").strip(" "))
    return x, y
    
#Euclidean distance
def calculate_distance(r1, r2):
    """
    r1 and r2 are rows in the dataframe of trials, and have columns with labels 'x' and 'y'
    Returns the Euclidean distance between the two values
    """
    s1 = np.array([r1['x'], r1['y']])
    s2 = np.array([r2['x'], r2['y']])
    return np.linalg.norm(s1 - s2)

def get_senses(db, word):
    """
    Queries FB for the list of senses that were used in the experiment for word (format word_pos) 
    """
    sr_types = {'face_n': ['expression_n_01', 'face_n_04'], 'book_n': ['record_n_05'],
     'glass_n': ['glass_n_03'], 'door_n': ['door_n_03'], 'school_n': ['school_n_03'], 
     'heart_n': ['center_n_01']}

    if word not in sr_types:
        return [k for k in db['inputs'][word] if k not in ['senses', 'type']]
    else:
        insuf_instances = sr_types[word]
        return [k for k in db['inputs'][word] if k not in ['senses', 'type'] + insuf_instances]

def get_num_senses(w, db):
    """
    Queries FB for the number of senses participants received for type w (word_pos) 
    """
    return db['inputs'][w]['senses']

def wordnet_defn(fb_sense):
    """
    Gets the WordNet definition for a sense in format word_pos_number
    """
    parts = fb_sense.split('_')
    synset_str = '_'.join(parts[:len(parts) - 2]) + '.' + '.'.join(parts[-2:])
    return wordnet.synset(synset_str).definition()

def mtx_correlation(m1, m2, method = 'spearman', randomize_m1_labels = False, confusion = False, return_ut = False, output_ci = False): 
    """
    Input:
        m1 and m2- lists of square Numpy matrices where each element in m1 has the same dimensions as the corresponding element in m2
        method- either 'spearman' or 'pearson'
        randomize_m1_labels- shuffle the labels of one of the matrices if true
        confusion- if true, computes the correlation between the matrices themselves, else computes the correlation of the upper triangular portions only
        return_ut- gives upper triangular sections of the matrices of m1 and m2 (1 dimensional arrays)
        output_ci- boolean that specifies whether or not a 95% confidence interval should be outputted for the correlation

    Output:
        Correlation between flattened versions of m1 and m2
        if return_ut, returns correlation, and upper triangular versions of m1 and m2
        if output_ci, returns (correlation, p-value), (ci_lower_bound, ci_upper_bound)
    """
    #m1 and m2 are lists of distance matrices, spearman or pearson correlation
    assert len(m1) == len(m2)
    if confusion:
        return stats.spearmanr(m1, m2)
    else:
        flat_m1 = []
        for i in range(len(m1)):
            #OpTimiZAtIoNS
            ut_m1 = m1[i][np.triu_indices(m1[i].shape[0], k = 1)]
            if randomize_m1_labels:
                np.random.shuffle(ut_m1)
            flat_m1 += ut_m1.tolist()
        flat_m2 = []
        for i in range(len(m2)):
            ut_m2 = m2[i][np.triu_indices(m2[i].shape[0], k = 1)]
            flat_m2 += ut_m2.tolist()
        if return_ut:
            if method == 'spearman':
                return stats.spearmanr(flat_m1, flat_m2), flat_m1, flat_m2
            if method == 'pearson':
                return stats.pearsonr(flat_m1, flat_m2)
        if output_ci:
            if method == 'spearman':
                r, p, lb, ub = corr_ci(flat_m1, flat_m2)
            if method == 'pearson':
                r, p, lb, ub = corr_ci(flat_m1, flat_m2, method = 'pearson')
            return (r, p), (lb, ub)
        else:
            if method == 'spearman':
                return stats.spearmanr(flat_m1, flat_m2)
            if method == 'pearson':
                return stats.pearsonr(flat_m1, flat_m2)

def rank_transform(a):
    """
    Inputs:
    a- numpy array
    Outputs:
    Rank transform of a, 
    """
    temp = a.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(a))
    return ranks

def corr_ci(x, y, alpha=0.05, method = 'spearman'):
    ''' calculate Spearman or Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    From https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    method = 'spearman' by default (if this is set, then we apply a rank transform)
    Returns
    -------
    r : float
      Spearman's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''
    x, y = np.asarray(x), np.asarray(y)
    if method == 'spearman':
        x, y = rank_transform(x), rank_transform(y)
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(len(x)-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

#Functions for random sampling
def random_num_senses(db):
    """
    Generate a random number of senses based on frequency in the database
    """
    vals, probs = get_num_to_sense_dist(db)
    return np.random.choice(vals, p = probs)

def get_num_to_sense_dist(db):
    """
    Returns all the senses and their probabilities according to frequency (used for random sampling)
    """
    val_counts = pd.Series([db['inputs'][i]['senses'] for i in db['inputs']]).value_counts()
    probs = val_counts / sum(val_counts)
    return probs.index, probs.values

def create_random_symmetric_mtx(dims = 3):
    """
    Generates dims x dims distance matrix of random coordinate assignments 
    """
    min_x = 89
    max_x = 873
    min_y = 282
    max_y = 564
    xs = np.random.uniform(min_x, max_x, size = dims)
    ys = np.random.uniform(min_y, max_y, size = dims)
    max_value = -1
    mtx = []
    for i in range(dims):
        row = []
        for j in range(dims):
            coords1, coords2 = np.array([xs[i], ys[i]]), np.array([xs[j], ys[j]])
            dist = np.linalg.norm(coords1 - coords2)
            row.append(dist)
            if dist > max_value:
                max_value = dist
        mtx.append(row)
    mtx = np.asarray(mtx)
    return mtx / max_value

def get_results_elig_users(db, metric, value):
    """
    TODO: Modify this function based on exclusion criteria/call it
    """
    #gets participants with stats that are higher than value
    results, corrs = get_results_users(db)
    incl_users = corrs[corrs[metric] > value]['userID'].tolist()
    return results, incl_users

def get_results_users(db):
    """
    Returns results- all participants' token assignments
    corrs- Dataframe where each row corresponds to a user, with the following columns:
    userID, Group Consistency, and Self Consistency, time, and prevChanged (number of times participant changed arrangement)
    """
    #Simpler version of the above function, so we can apply more complicated exclusion criteria
    trials = get_trial_data(db)
    participants = get_participant_data(db)
    user_data = participants[(participants['completedTask'] == 1) & (participants.index > 4)] #Excluding data from experimenters
    results = trials[trials['userID'].isin(user_data['userID'])]
    users = user_data['userID']
    repeat_corr = all_repeats(results, users, plot = False) #self correlation
    shared_results = results[results['trialType'] == 'shared']
    shared_corrs = group_consistency(shared_results, users) #shared correlation
    user_time_word_changes = get_time_and_changes(results, user_data) #metadata
    consistency = pd.DataFrame({'Group Consistency': shared_corrs, 'Self Consistency': repeat_corr})
    corrs = user_time_word_changes.merge(consistency, on = user_time_word_changes.index).drop('key_0', axis = 1)
    return results, corrs

#TODO: Updates these two from notebook version
def containing_query(df, value, selection_criteria, dist_mtx_dict, bert_key = 'bert'):
    words_with_crit = df[df[value].isin(selection_criteria)]['lemma'].unique()
    data_for_words = {w : dist_mtx_dict[w] for w in words_with_crit}
    return mtx_correlation([data_for_words[w]['expt'] for w in data_for_words],
                          [data_for_words[w][bert_key] for w in data_for_words], method = 'pearson')

def range_query(df, value, low, high, dist_mtx_dict, bert_key = 'bert'):
    #Inclusive of low and high
    words_with_crit = df[(df[value] >= low) & (df[value] <= high)]['lemma'].unique()
    data_for_words = {w : dist_mtx_dict[w] for w in words_with_crit}
    if bert_key == 'confusion':
        expt_data = [(1 - data_for_words[w]['expt']).tolist() for w in data_for_words]
        conf_matrices = [data_for_words[w][bert_key] for w in data_for_words]

        return stats.spearmanr(flatten(flatten(expt_data)), flatten(flatten(conf_matrices)))[0]

    return mtx_correlation([data_for_words[w]['expt'] for w in data_for_words],
                          [data_for_words[w][bert_key] for w in data_for_words])[0]

def sample_from_shared(results, users, matrices, sample_size = 10):
    """
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
    users- list of userID strings
    matrices is a dict with format {word -> {expt: nxn distance matrix for senses, 
                                    bert: nxn cosine distance matrix of the senses from SEMCOR examples}}
    sample_size- random sample of users

    Return Correlation between [sample_size] responses for the shared trials and BERT matrices for shared words
    """
    shared_words = ['foot_n', 'table_n', 'plane_n', 'degree_n', 'right_n', 'model_n']
    sel_users = np.random.choice(users, sample_size)
    bert_matrices = []
    sample_matrices = []
    for w in shared_words:
        sample_means = mean_distance_mtx(results, w, 'shared', sel_users, normalize = True)
        bert_matrices.append(matrices[w]['bert'])
        sample_matrices.append(sample_means)
    return mtx_correlation(sample_matrices, bert_matrices, method = 'spearman')[0]

def get_lemma_counts(results, incl_users, db):
    """
    Input:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels

    incl_users- list of userID strings
    db- Firebase JSON of experimental data

    Output:
    Dataframe that shows the number of times a type was shown to subjects in incl_users
    """
    test_repeat = results[(results['userID'].isin(incl_users)) & (results['trialType'].isin(['test', 'repeat']))]
    lemma_counts = test_repeat['lemma'].value_counts()
    lemma_counts = pd.DataFrame(lemma_counts / [get_num_senses(l, db) for l in lemma_counts.index]).sort_values('lemma',
                                                                                                 ascending = False)
    lemma_counts['num_trials'] = lemma_counts['lemma']
    lemma_counts.drop('lemma', axis = 1, inplace = True)
    lemma_counts = lemma_counts.reset_index()
    lemma_counts.rename({'index': 'lemma'}, axis = 1, inplace = True)
    return lemma_counts

def mtx_to_df(mtx, senses, reorder = []):
    """
    Input:
    mtx- square Numpy matrix
    senses- senses names that correspond to each entry
    reorder- order the senses should be in

    Output: Pandas Dataframe of matrix information, used for Seaborn plotting
    """
    mtx = np.round(mtx, 3)
    df = pd.DataFrame(mtx, columns = senses, index = senses)
    if len(reorder):
        df.columns = df.columns[reorder]
        df.index = df.index[reorder]
    return df

def get_test_result_data(results, w, incl_users):
    """
    Input:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
    w- word_pos, string
    incl_users- string of userIDs

    Output: Numpy matrix of the mean distances for type w across both test and repeat trials
    """

    test_means = mean_distance_mtx(results, w, 'test', incl_users, normalize = True)
    repeat_means = mean_distance_mtx(results, w, 'repeat', incl_users, normalize = True)
    expt_means = test_means
    if repeat_means.shape:
        expt_means = np.mean([test_means, repeat_means], axis = 0)
        expt_means /= np.max(expt_means)
    return expt_means

def get_tagged_distances(results, incl_users, db):
    """
    Inputs:
    results- Pandas dataframe with schema of get_trial_data's output:
        One row per trial, columns are the following: trialID- UID for row (primary key), userID- UID for participant,
        trialIndex- which trial it was out of 18 [1, 18], trialType- one of ['training', 'shared', 'test', 'repeat'] in that order,
        prevChanged- amount of times participant changed the arrangement for the word after seeing the sense [0, number of senses - 1],
        lemma- word_pos (FB doesn't allow periods in fields so we had to change it from NLTK), sense- word_pos_number,
        x, y- coordinates of the box participant placed, in pixels
    incl_users- list of userID strings
    db- Firebase JSON of responses

    Output:
    Pandas Dataframe where each row represents a pair of senses. Columns: item- tuple of (sense1, sense2), listed as strings of word_pos_number, 
    word_type- word_pos, relation_type- either 'homonymous' or 'polysemous', dist- distance between pairs, normalized over participant's responses for word_type,
    user- participant's userID

    """
    hp_tags = []
    homonyms = [('foot_n_01', 'foot_n_02'), ('foot_n_02', 'foot_n_03'), ('table_n_01', 'table_n_02'), ('table_n_02', 'table_n_03'),
                ('academic_degree_n_01', 'degree_n_01'), ('academic_degree_n_01', 'degree_n_02'), ('right_n_01', 'right_n_02'),
                ('right_n_01', 'right_n_04'), ('model_n_02', 'model_n_03'), ('model_n_01', 'model_n_03'), ('airplane_n_01', 'plane_n_02'),
                ('airplane_n_01', 'plane_n_03')]
    for s in results[results['trialType'] != 'training']['lemma'].unique():
        sense_combos = list(itertools.combinations(get_senses(db, s), 2))
        for t in sense_combos:
            if t in homonyms:
                hp_tags.append({'item': t, 'word_type': s, 'relation_type': 'homonymous'})
            else:
                hp_tags.append({'item': t, 'word_type': s, 'relation_type': 'polysemous'})
    df = []
    for w in results[results['trialType'] != 'training']['lemma'].unique():
        pair_results = results[(results['userID'].isin(incl_users)) & (results['lemma'] == w)]
        for u in pair_results['userID'].unique():
            max_user_dist = 0
            user_reported_distances = []
            word_pairs = [t for t in hp_tags if t['word_type'] == w]
            for t in word_pairs:
                user_report = pair_results[(pair_results['trialType'] != 'training') & (pair_results['userID'] == u) & \
                        (pair_results['lemma'] == t['word_type']) & (pair_results['sense'].isin(t['item']))]
                user_report = user_report.reset_index()
                dist = calculate_distance(user_report.iloc[0], user_report.iloc[1])
                if dist > max_user_dist:
                    max_user_dist = dist
                row = t.copy()
                row['dist'] = dist
                row['user'] = u
                user_reported_distances.append(row)
            for d in user_reported_distances:
                d['dist'] = d['dist'] / max_user_dist
            df += user_reported_distances
    return pd.DataFrame(df)

def exclusion_criteria(corrs):
    """
    corrs- Dataframe where each row corresponds to a user, with the following columns:
    userID, Group Consistency, Self Consistency, and Correlation with SN, time, and prevChanged (number of times participant changed arrangement)

    Applies exclusion criteria to corrs: group consistency > 0.4, or self consistency above 0.2, and English > 50% of the time
    """
    incl_users = corrs[(corrs['Group Consistency'] > 0.4) | (corrs['Self Consistency'] > 0.2)]['userID']
    incl_users = incl_users.tolist()
    incl_users.remove('-M6Cl_rmTwH43zEQtJcK') #user ID for the worker ID reporting English as < 50% of language use (private)
    return incl_users

def sense_pair_as_tuple(s):
    """
    s- String with '(word_pos_number, word_pos_number)', converts to tuple
    """
    s1, s2 = s.split(",")
    remove_chars = "\'() "
    s1, s2 = s1.strip(remove_chars), s2.strip(remove_chars)
    return (fb_to_local(s1), fb_to_local(s2))
"""
Work in progress: seeing if one half can predict the other half
lemmas = lemma_counts.index.tolist()
all_matrices = {l: [] for l in lemmas}
for l in lemmas:
    word_data = test_data[test_data['lemma'] == l]
    test_users = word_data[word_data['trialType'] == 'test']['userID']
    repeat_users = word_data[word_data['trialType'] == 'repeat']['userID']
    for u in test_users:
        u_mtx, _ = get_subject_mtx(word_data, u, l, 'test')
        all_matrices[l].append(u_mtx)
    for u in repeat_users:
        u_mtx, get_subject_mtx(word_data, u, l, 'repeat')
        all_matrices[l].append(u_mtx)
test_corrs = []
flatten = lambda l: [item for sublist in l for item in sublist]

for _ in range(1000):
    half_1 = []
    half_2 = []
    for l in all_matrices:
        word_responses = np.array(all_matrices[l])
        n = len(word_responses)
        if n % 2 == 1:
            drop_one = np.random.choice(np.arange(n), n - 1, replace=False)
            word_responses = word_responses[drop_one]
        n = len(word_responses)
        half = int(n / 2)
        shuffled_indices = np.random.choice(np.arange(n), n, replace = False)
        half_1 += list(np.mean(word_responses[shuffled_indices[:half]], axis = 0))
        half_2 += list(np.mean(word_responses[shuffled_indices[half:]], axis = 0))
    test_corrs.append(mtx_correlation(half_1, half_2, method = 'spearman'))
    print(test_corrs)
shared_matrices = {l: [] for l in ['degree_n', 'plane_n', 'table_n', 'foot_n', 'right_n', 'model_n']}
for l in shared_matrices.keys():
    word_data = results[results['lemma'] == l]
    users = word_data['userID']
    for u in users:
        u_mtx, _ = get_subject_mtx(word_data, u, l, 'shared')
        shared_matrices[l].append(u_mtx)
sns.distplot(test_corrs)
plt.title("Correlation between two random halves of subjects (Test Trials)")
plt.xlabel("Spearman Correlation")

"""