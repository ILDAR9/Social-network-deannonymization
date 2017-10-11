import re
import os
import pandas as pd
import cyrtranslit
from tqdm import tqdm_notebook as tqdm
from fuzzywuzzy import fuzz
import networkx as nx
import random
import numpy as np
import pickle
import itertools as it
from multiprocessing import Pool as ThreadPool


folder_data = '../data/'
folder_gen = os.path.join(folder_data, 'generated')
foler_matches = '../matches'

def clean_lineinst(line):
    pat = re.compile("(\d+),(.*),(.*)")
    pat_word = re.compile('[^a-zA-Zа-яА-Я\d\s]+')

    uid, uname, fname = pat.match(line).groups()
    fname = re.sub(pat_word, '', fname).strip().lower()

    fname = cyrtranslit.to_latin(fname, 'ru').replace("'", '')
    return (uid, uname, fname)

def read_from_matches(fname):
    pass #ToDo finish

def clean_linevk(line):
    pat = re.compile("(\d+),(.*),(.*),(.*)")
    pat_word = re.compile('[^a-zA-Zа-яА-Я\d\s]+')

    try:
        uid, uname, name1, name2 = pat.match(line).groups()
        name1 = re.sub(pat_word, '', name1).strip().lower()
        name2 = re.sub(pat_word, '', name2).strip().lower()
        fname = name1 + ' ' + name2
    except AttributeError:
        print(line)
    fname = cyrtranslit.to_latin(fname, 'ru').replace("'", '')
    return (uid, uname, fname)


def read_clean_csv(fname, from_raw = True):

    save_to = fname + '.pickle'
    if not from_raw:
        df = pickle.load(open(os.path.join(folder_data, save_to), "rb"))
        return df
    if 'vk' in fname:
        num_col = 4
        columns = ['uid_vk', 'uname', 'name_vk']
    elif 'inst' in fname:
        num_col = 3
        columns = ['uid_inst', 'uname', 'name_inst']

    df = pd.DataFrame()

    clean_line = clean_lineinst if num_col == 3 else clean_linevk
    with open(os.path.join(folder_data, fname), 'r') as f:
        for line in tqdm(f.readlines()):
            df = pd.concat([df, pd.DataFrame([clean_line(line)])], ignore_index=True)
    df.columns = columns
    uid_col = columns[0]
    df[uid_col] = df[uid_col].astype(int)
    pickle.dump(df, open(os.path.join(folder_data, save_to), "wb"))
    return df

def read_combine_df(from_raw = True, merge_how = 'inner'):

    vk = read_clean_csv(fname='vk_personal.csv', from_raw = from_raw)
    inst = read_clean_csv(fname='inst_personal.csv', from_raw = from_raw)

    df = pd.merge(inst, vk, on='uname', how = merge_how)
    print(df.head())
    return df

def __read_uid_set(fname):
    s = set()
    with open(os.path.join(folder_data, fname), 'r') as f:
        for a, b in (line.split() for line in f.readlines()):
            s.add(int(a))
            s.add(int(b))
    return s

def generate_overlap(df, sim_f):
    ls = []
    uid_lset = __read_uid_set('vk_lid_rid.csv')
    uid_rset = __read_uid_set('inst_lid_rid.csv')

    for uid in df.uid:
        if uid in uid_lset:
            uid_s = sim_f(uid)
            if uid_s and uid_s in uid_rset:
                ls.append((uid, uid_s))
    return ls

def generate_true_mapping(from_raw=False):
    if from_raw:
        inst = read_clean_csv(fname='inst_personal.csv', num_col=3)
        inst.columns = ['uid', 'uname', 'inst_name']
        inst.to_csv(os.path.join(folder_gen, 'inst_personal.csv'), index=False)
    else:
        inst = pd.read_csv(os.path.join(folder_gen, 'inst_personal.csv'))
    print(inst.head())

    if from_raw:
        vk = read_clean_csv(fname='vk_personal.csv', num_col=4)
        vk.columns = ['uid', 'uname', 'vk_name']
        vk.to_csv(os.path.join(folder_gen, 'vk_personal.csv'), index=False)
    else:
        vk = pd.read_csv(os.path.join(folder_gen, 'vk_personal.csv'))
    print(vk.head())

    inst_uname_id = dict(((d[1], d[0]) for d in inst[['uid', 'uname']].values))
    vk_id_name = dict(((d[0], d[1]) for d in vk[['uid', 'uname']].values))

    def get_sim(uid_vk):
        return inst_uname_id[vk_id_name[uid_vk]] if uid_vk in vk_id_name and vk_id_name[uid_vk] in inst_uname_id else None

    fname = 'true_mapping.csv'
    with open(os.path.join(folder_gen, fname), 'w') as rw:
        ls = generate_overlap(vk, get_sim)
        rw.write('uid_vk,uid_inst\n')
        for tup in ls:
            rw.write('%s,%s\n' % tup)
    return fname

def get_df_1step_and_others(mapping_file_name = 'true_mapping.csv', suffix_name = None, count_1step = 3000):
    true_mapping = pd.read_csv(os.path.join(folder_gen, mapping_file_name))
    if not suffix_name:
        return true_mapping[:count_1step], true_mapping[count_1step:]

    uid_set = set(true_mapping['uid_' + suffix_name].values[:count_1step])
    uid_set_othres = set(true_mapping['uid_' + suffix_name].values[count_1step:])
    df = pd.read_csv(os.path.join(folder_gen, suffix_name + '_personal.csv'))
    df_1step = df[df['uid'].isin(uid_set)]
    df_others = df[df['uid'].isin(uid_set_othres)]
    print(df.shape, df_1step.shape, df_others.shape)
    return df_1step, df_others

def deg_dist(G, n, bins, size):
    feature_set = [0 for i in range(2 * bins)]
    _1hop = G[n]
    _1hop = G.degree(_1hop).values()
    for h in _1hop:
        if h < bins * size:
            feature_set[int(h / size)] += 1
    _prev = set(nx.single_source_shortest_path_length(G, n, cutoff=1).keys())
    _2hop = set(nx.single_source_shortest_path_length(G, n, cutoff=2).keys())
    _2hop = _2hop - _prev
    _2hop = G.degree(_2hop).values()
    for h in _2hop:
        if h < bins * size:
            feature_set[bins + int(h / size)] += 1
    return feature_set

f_set1s = dict()
f_set2s = dict()

def feature(G1, n, G2, m, bins=21, size=50):
    if n in f_set1s:
        f_set1 = f_set1s[n]
    else:
        f_set1 = deg_dist(G1, n, bins, size)
        f_set1s[n] = f_set1

    if m in f_set2s:
        f_set2 = f_set2s[m]
    else:
        f_set2 = deg_dist(G2, m, bins, size)
        f_set2s[m] = f_set2

    feature_set = f_set1 + f_set2
    # the Silhouette Coefficient
    n_deg = G1.degree(n)
    m_deg = G2.degree(m)
    feature_set.append(abs(n_deg - m_deg) / max(n_deg, m_deg, 1))
    return feature_set


def gen_features(G1, G2, lid_rid):
    features = []
    labels = []

    rid_othres = np.array(list(set(G2.nodes()) - set((x[1] for x in lid_rid))))

    # set_l = set(G1.nodes())
    # set_r = set(G2.nodes())
    for i, j in tqdm(lid_rid):
        # if i in set_l:
        #     if j in set_r:
        features.append(feature(G1, i, G2, j))
        labels.append(1)
        # Choose randomly
        j_other = random.choice(rid_othres)
        # if j_other in set_r:
        features.append(feature(G1, i, G2, j_other))
        labels.append(0)
        if random.random() > 0.6:
            j_other = random.choice(rid_othres)
            # if j_other in set_r:
            features.append(feature(G1, i, G2, j_other))
            labels.append(0)
    return features, labels

read_g = lambda fname : nx.read_edgelist(os.path.join(folder_data, fname), nodetype = int)

def gen_train_data(lid_rid, G1, G2, save_to, from_raw = True):
    if not from_raw:
        features, labels = pickle.load(open(os.path.join(folder_gen, save_to), "rb"))
        return features, labels

    features, labels = gen_features(G1, G2, lid_rid)
    pickle.dump((features, labels), open( os.path.join(folder_gen, save_to + '.pickle'), "wb" ))

def retrain_model(model, fnames):
    features = []
    labels = []
    for fname in fnames:
        fets, labs = pickle.load(open(os.path.join(folder_gen, fname + '.pickle'), "rb"))
        features += fets
        labels += labs

    if len(labels) == len(features):
        model.fit(features, labels)
    else:
        raise Exception('label len = %d, feature len = %d' % (len(labels), len(features)))

    return model

def filter_same(results):
    ratio_dict = dict([((lnode, rnode), ratio) for lnode, rnode, ratio in it.chain.from_iterable(results)])

    def decide(d, node, new_node, ratio):
        if node in d:
            r = ratio_dict[(node, d[node])]
            if r < 1 and r < ratio:
                d[node] = new_node
        else:
            d[node] = new_node

    ldict = dict()
    for lnode, rnode, ratio in it.chain.from_iterable(results):
        decide(ldict, lnode, rnode, ratio)

    rdict = dict()
    for lnode, rnode in ldict.items():
        if rnode in rdict:
            if ratio_dict[(lnode, rnode)] > ratio_dict[(rdict[rnode], rnode)]:
                rdict[rnode] = lnode
        else:
            rdict[rnode] = lnode
    print('resulting 2 step', len(rdict), 'from', len(ratio_dict), ' | ', len(rdict) / len(ratio_dict))
    filtered_res = [(lnode, rnode) for rnode, lnode in rdict.items()]
    pickle.dump(filtered_res, open(os.path.join(folder_gen, 'test2_predicted_filtered.pickle'), "wb"))
    return filtered_res

def precision_recall(lid_rid):
    true_mapping = pd.read_csv(os.path.join(folder_gen, 'true_mapping.csv'))
    true_mapping = set((x[0], x[1]) for x in true_mapping.values)
    count = 0
    for res in lid_rid:
        if res in true_mapping:
            count += 1
    precision = count / len(lid_rid)
    recall = count / len(true_mapping)
    return (precision, recall)

def find_sim_and_predict(data):
    df_vals_l = data['vals_l']
    df_vals_r = data['vals_r']
    model = data['model']
    G1 = data['G1']
    G2 = data['G2']
    thread_num = data['thread_num']
    name_sim_threshold = data['name_sim_threshold']

    res_list = []
    print('Start', thread_num)
    total_count = 1
    true_count = 0
    for row_l in tqdm(df_vals_l):
        for row_r in df_vals_r:
            ratio = fuzz.token_set_ratio(row_l[1], row_r[1])
            if ratio > name_sim_threshold:
                x = np.array(feature(G1, row_l[0], G2, row_r[0]), ).reshape((1, -1))
                total_count += 1
                if model.predict(x) == 1:
                    true_count += 1
                    res_list.append((row_l[0], row_r[0], ratio))
    print('Thread', thread_num, 'True =', true_count, 'False =', total_count - true_count - 1,
          'Total =', total_count, 'True/Total =', true_count / total_count)
    return res_list


def prepare_data_for_threads(df_l, df_r, model, G1, G2, threads, name_sim_threshold = None):
    data = []

    thr_size = len(df_l) // threads
    for i in range(threads):
        s = i * thr_size
        e = i * thr_size + thr_size
        d = {
            'thread_num': i,
            'vals_l': df_l[s:e] if i + 1 < threads else df_l[s:],
            'vals_r': df_r,
            'model': model,
            'G1': G1,
            'G2': G2,
        }
        if name_sim_threshold:
            d['name_sim_threshold'] =  name_sim_threshold
        data.append(d)
    return data


def predict_parallel(df_l, df_r, model, G1, G2, name_sim_threshold, threads=4):
    pool = ThreadPool(threads)
    df_l = df_l[['uid', 'vk_name']].dropna().values
    df_r = df_r[['uid', 'inst_name']].dropna().values
    data = prepare_data_for_threads(df_l, df_r, model, G1, G2, threads, name_sim_threshold = name_sim_threshold)
    results = pool.map(find_sim_and_predict, data)
    pool.close()
    pool.join()
    for i, res in enumerate(results):
        print(i, 'true count', len(res))
    return results


def dump_features(G, graph_name, bins=21, size=50):
    features = []
    for n in tqdm(G.nodes()):
        feature = deg_dist(G, n, bins, size)
        features.append((n, feature))
    pickle.dump(features, open(os.path.join(folder_gen, 'features_%s.pickle' % graph_name), "wb"))

def ml_iteration(df_l_others, df_r_others, model, G1, G2, name_sim_threshold, threads, save_to):
    results = predict_parallel(df_l_others, df_r_others, model, G1, G2, name_sim_threshold, threads)
    lid_rid = filter_same(results)
    print('lid_rid',len(lid_rid), len(set((x[0] for x in lid_rid))), len(set((x[1] for x in lid_rid))))
    pickle.dump(lid_rid, open(os.path.join(folder_gen, save_to + '.pickle'), "wb"))
    return lid_rid

def filter_others_from_predicted(df_l, df_r, lid_rid):
    def filt(df, index):
        used = set((x[index] for x in lid_rid))
        return df[~df['uid'].isin(used)]
    l, r = filt(df_l, 0), filt(df_r, 1)
    print(len(df_l), len(l), len(lid_rid), len(lid_rid) + len(l))
    return l, r


