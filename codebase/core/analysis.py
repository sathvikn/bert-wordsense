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

def display_sense_definitions(results, trial_type):
    shared_trials = results[results['trialType'] == trial_type]
    pd.set_option('display.max_colwidth', 200)

    sense_defns = pd.DataFrame({"Sense": shared_trials['sense'],
        "Type": shared_trials['lemma'],
        "Definition": shared_trials['sense'].apply(wordnet_defn)}).drop_duplicates()
    sense_defns['Definition'] = sense_defns['Definition']
    return sense_defns

def fb_to_local(fb_sense):
    parts = fb_sense.split('_')
    return '_'.join(parts[:len(parts) - 2]) + '.' + '.'.join(parts[-2:])


def get_time_and_changes(results, user_df):
    changed = results[['userID', 'lemma', 'prevChanged']].groupby(['userID', 'lemma']).agg(max).reset_index()
    user_changes = changed.groupby('userID').agg(sum).reset_index()
    time = user_df[['userID', 'timeTaken']]
    time['changes'] = user_changes['prevChanged'].values
    return time

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
def repeat_correlations(results, userID, subject_index = None, plot = False):
    user_trials = results[results['userID'] == userID]
    repeat_types = user_trials[user_trials['trialType'] == 'repeat']['lemma'].unique()
    user_orig = []
    user_repeat = []
    for l in repeat_types:
        original_result, senses = get_subject_mtx(results, userID, l, 'test')
        repeat_result, _ = get_subject_mtx(results, userID, l, 'repeat')
        if plot:
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
            r = mtx_correlation([original_result], [repeat_result])
            my_eyes.suptitle("Subject " + subject_index + " Annotations for Repeated Type " + l + " (r = " + str(np.round(r, 2)) + ")", x = pos / 1.9)
        user_orig.append(np.array(original_result))
        user_repeat.append(np.array(repeat_result))
    return user_orig, user_repeat

def group_consistency(results, users, random_baseline = False, exclude = []):
    #Returns a score for each user
    #"for each pairwise relationship, take the average of all participants except one"
    #shared_results = results[results['trialType'] == 'shared']
    hoo_corrs = []
    for u in users:
        subject_index = str(users[users == u].index[0])
        user_corr = hoo_corr(results, u, exclude)
        hoo_corrs.append(user_corr)
        #print("Hold One Out Correlation for User" , subject_index, user_corr)
    if random_baseline:
        random_corr = random_vs_all(results)
        #print("Random Baseline", random_corr)
        hoo_corrs.append(random_corr)
    return hoo_corrs

#hold one out correlation
def hoo_corr(results, userID, exclude):
    user_results = []
    avg_results = []
    for l in results['lemma'].unique():
        if l not in exclude:

            held_out_results = results[results['userID'] != userID]
            user_lst = held_out_results['userID'].unique().tolist()
            avg_with_others = mean_distance_mtx(held_out_results, l, 'shared', user_lst)
            avg_results.append(avg_with_others)
            user_result, _ = get_subject_mtx(results, userID, l, 'shared')
            user_results.append(user_result)
    return mtx_correlation(user_results, avg_results)

def user_vs_user_shared(results, user1, user2):
    u1_results = []
    u2_results = []
    for l in results.lemma.unique():
        u1_word, _ = get_subject_mtx(results, user1, l, 'shared')
        u2_word, _ = get_subject_mtx(results, user2, l, 'shared')
        u1_results.append(u1_word)
        u2_results.append(u2_word)
    return mtx_correlation(u1_results, u2_results)

def my_correlations(participants, trials, results, userids):
    complete = participants[participants['completedTask'] == 1]
    my_userid = complete.iloc[1]['userID']
    gt_results = trials[trials['userID'] == my_userid]
    shared_plus_gt = pd.concat([results, gt_results])
    shared_plus_gt = shared_plus_gt[shared_plus_gt['trialType'] == 'shared']
    return [user_vs_user_shared(shared_plus_gt, u, my_userid) for u in userids]

#refactoring the above fn to work with random data
def random_vs_all(results):
    user_results = []
    avg_results = []
    for l in results['lemma'].unique():
        user_lst = results['userID'].unique().tolist()
        avg_results.append(mean_distance_mtx(results, l, 'shared', user_lst))
        user_results.append(create_random_symmetric_mtx()) #All of these words have 3 senses
    return mtx_correlation(user_results, avg_results)

def all_repeats(results, users, random_baseline = False, db = None, plot = False):
    all_orig = []
    all_repeat = []
    user_corrs = []
    for u in users:
        subject_index = str(users[users == u].index[0])
        user_orig, user_repeat = repeat_correlations(results, u, subject_index, plot = plot)
        user_corr = mtx_correlation(np.asarray(user_orig), np.asarray(user_repeat))
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
        random_corrs = mtx_correlation(random_orig, random_repeat)
        user_corrs.append(random_corrs)
        #print("Random Baseline", random_corrs)

    #print("Correlation of all original vs. repeat trials", mtx_correlation(all_orig, all_repeat))
    return user_corrs
    

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

#MDS functions
def mean_distance_mtx(results, lemma, trial_type, user_lst):
    shared_tensor = []
    for j in range(len(user_lst)):
        user_result, senses = get_subject_mtx(results, user_lst[j], lemma, trial_type)
        if len(user_result):
            shared_tensor.append(user_result)
    shared_tensor = np.asarray(shared_tensor)
    return np.mean(shared_tensor, axis = 0)

def plot_mds(word_means, word, mds_model, db, src):
    results = mds_model.fit_transform(word_means)
    x = results[:,0]
    y = results[:,1]
    senses = get_senses(db, word)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(senses):
        ax.annotate(txt, (x[i], y[i]))
    plt.title("MDS over Averaged " + src + " Distances for " + word)

def plot_all_mds(results, users, trial_type, db):
    data = results[results['trialType'] == trial_type]
    mds_model = MDS(n_components = 2, dissimilarity = 'precomputed')
    for l in data['lemma'].unique():
        word_means = mean_distance_mtx(results, l, trial_type, users)
        plot_mds(word_means, l, mds_model, db, "Reported")

def plot_individual_mds(results, word, trial_type, users, db, sense_df):
    mds_model = MDS(n_components = 2, dissimilarity = 'precomputed')
    user_lst = users.tolist()
    word_means = mean_distance_mtx(results, word, trial_type, user_lst)
    plot_mds(word_means, word, mds_model, db, "Reported")
    return sense_df[sense_df['Type'] == word]

def plot_consistency_hist(randoms, parts, title, legend = True):
    sns.distplot(randoms)
    fmt_color = lambda num: "C" + str(num)
    colors = [fmt_color(i) for i in range(len(parts))]
    #TODO randomly generate this
    for i in range(len(parts)):
        plt.axvline(parts[i], c = colors[i], label = i)
    if legend:
        plt.legend(title = 'Subject Index')
    plt.title(title)
    plt.xlabel("Spearman Correlation")

def simulate_self_correlation(num_trials, db):
    randoms_self = []
    for i in range(num_trials):
        first_trial_dim = random_num_senses(db)
        second_trial_dim = random_num_senses(db)
        random_orig = np.array([create_random_symmetric_mtx(first_trial_dim), create_random_symmetric_mtx(second_trial_dim)])
        random_repeat = np.array([create_random_symmetric_mtx(first_trial_dim), create_random_symmetric_mtx(second_trial_dim)])
        random_corrs = mtx_correlation(random_orig, random_repeat)
        randoms_self.append(random_corrs)
    return randoms_self

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

def wordnet_defn(fb_sense):
    parts = fb_sense.split('_')
    synset_str = '_'.join(parts[:len(parts) - 2]) + '.' + '.'.join(parts[-2:])
    return wordnet.synset(synset_str).definition()

def mtx_correlation(m1, m2, method = 'spearman'): 
    #m1 and m2 are lists of distance matrices, spearman or pearson correlation
    assert len(m1) == len(m2)
    flat_m1 = []
    for i in range(len(m1)):
         #OpTimiZAtIoNS
         flat_m1 += m1[i][np.triu_indices(m1[i].shape[0], k = 1)].tolist()
    flat_m2 = []
    for i in range(len(m2)):
        flat_m2 += m2[i][np.triu_indices(m2[i].shape[0], k = 1)].tolist()
    if method == 'spearman':
        return stats.spearmanr(flat_m1, flat_m2)[0]
    if method == 'pearson':
        return stats.pearsonr(flat_m1, flat_m2)[0]
        
def random_num_senses(db):
    vals, probs = get_num_to_sense_dist(db)
    return np.random.choice(vals, p = probs)

def get_num_to_sense_dist(db):
    val_counts = pd.Series([db['inputs'][i]['senses'] for i in db['inputs']]).value_counts()
    probs = val_counts / sum(val_counts)
    return probs.index, probs.values

def create_random_symmetric_mtx(dims = 3):
    mtx = np.random.uniform(0,1000,size=(dims, dims))
    mtx = (mtx + mtx.T)/2
    np.fill_diagonal(mtx, 0)
    max_val = max(mtx.flatten())
    mtx = mtx / max_val
    return mtx

def get_results_elig_users(db, metric, value):
    #gets participants with stats that are higher than value
    trials = get_trial_data(db)
    participants = get_participant_data(db)
    user_data = participants[(participants['completedTask'] == 1) & (participants.index > 4) 
                             & ~(participants['workerID'].str.startswith("pilot"))] #excluding my data/Jon/Stephan
    results = trials[trials['userID'].isin(user_data['userID'])]
    users = user_data['userID']
    repeat_corr = all_repeats(results, users, plot = False) #self correlation
    shared_results = results[results['trialType'] == 'shared']
    shared_corrs = group_consistency(shared_results, users) #shared correlation
    user_time_word_changes = get_time_and_changes(results, user_data) #metadata
    consistency = pd.DataFrame({'Group Consistency': shared_corrs, 'Self Consistency': repeat_corr})
    corrs = user_time_word_changes.merge(consistency, on = user_time_word_changes.index).drop('key_0', axis = 1)
    corrs['Correlation with SN'] = my_correlations(participants, trials, results, users) #vs gold standard
    incl_users = corrs[corrs[metric] > value]['userID'].tolist()
    return results, incl_users