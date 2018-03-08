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
from importlib import reload

base_folder = '/home/ildar/projects/pycharm/social_network_revealing/graphmatching/'
folder_data = os.path.join(base_folder, 'data')
folder_gen = os.path.join(folder_data, 'generated')
folder_matches = os.path.join(base_folder, 'matches')
fname_true_mapping = 'true_mapping.csv'

def clean_lineinst(line):
    pat = re.compile("(\d+),(.*),(.*)")
    pat_word = re.compile('[^a-zA-Zа-яА-Я\d\s]+')

    uid, uname, fname = pat.match(line).groups()
    fname = re.sub(pat_word, '', fname).strip().lower()

    fname = cyrtranslit.to_latin(fname, 'ru').replace("'", '')
    return (uid, uname, fname)

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

    vk = read_clean_csv(fname='vk_personal2.csv', from_raw = from_raw)
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
    fname = fname_true_mapping
    df = read_combine_df(from_raw=from_raw)

    with open(os.path.join(folder_gen, fname), 'w') as rw:
        rw.write('uid_vk,uid_inst\n')
        for tup in map(lambda x: (x[0], x[1]), df[['uid_vk', 'uid_inst']].values):
            rw.write('%s,%s\n' % tup)
    return fname

def read_true_mapping(as_set = False):
    true_mapping = pd.read_csv(os.path.join(folder_gen, fname_true_mapping))
    if as_set:
        return set((x[0], x[1]) for x in true_mapping.values)
    else:
        return true_mapping

def get_df_1step_and_others(suffix_name = None, count_1step = 3000):
    true_mapping = read_true_mapping()
    if not suffix_name:
        return true_mapping[:count_1step], true_mapping[count_1step:]
    uid = 'uid_' + suffix_name
    uid_set = set(true_mapping[uid].values[:count_1step])
    uid_set_othres = set(true_mapping[uid].values[count_1step:])
    df = pickle.load(open(os.path.join(folder_data, suffix_name + '_personal.csv.pickle'), "rb"))
    df_1step = df[df[uid].isin(uid_set)]
    df_others = df[df[uid].isin(uid_set_othres)]
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

def get_degs(g, uid):
    degs = []
    for v in g[uid]:
        degs.append(g.degree(v))
    return degs

def double_deg_degs(g, uid):
    degs = []
    for v in g[uid]:
        degs += get_degs(g,v)
    return degs

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

    # n_2hop = len(set(nx.single_source_shortest_path_length(G1, n, cutoff=2).keys())) - n_deg
    # m_2hop = len(set(nx.single_source_shortest_path_length(G2, m, cutoff=2).keys())) - m_deg
    #
    # feature_set.append(abs(n_2hop - m_2hop) / max(n_2hop, m_2hop, 1))

    # name sim coef
    # ratio = fuzz.token_set_ratio(G1.node[n]['fname'], G2.node[m]['fname'])
    # feature_set.append(ratio)

    return feature_set

def read_gs(g1_fname, g2_fname, from_raw):
    if not from_raw:
        G1, G2 = pickle.load(open(os.path.join(folder_gen, 'G1_G2.pickle'), "rb"))
    else:
        read_lid_rid = lambda fname : nx.read_edgelist(os.path.join(folder_data, fname), nodetype = int)
        G1 = read_lid_rid(g1_fname)
        G2 = read_lid_rid(g2_fname)

        df = read_combine_df(from_raw=False, merge_how='outer')
        for node in G1.nodes():
            G1.node[node]['fname'] = df[df['uid_vk'] == node]['name_vk'].values[0]
            G1.node[node]['uname'] = df[df['uid_vk'] == node]['uname'].values[0]

        for node in G2.nodes():
            G2.node[node]['fname'] = df[df['uid_inst'] == node]['name_inst'].values[0]
            G2.node[node]['uname'] = df[df['uid_inst'] == node]['uname'].values[0]

        pickle.dump((G1, G2), open(os.path.join(folder_gen, 'G1_G2.pickle'), "wb"))
    print('G1 has all fname', all((G1.node[x]['fname'] for x in G1.nodes())))
    print('G2 has all fname', all((G2.node[x]['fname'] for x in G2.nodes())))
    return G1, G2


def gen_features(data):
    global f_set1s, f_set2s
    G1 = data['G1']
    G2 = data['G2']
    lid_rid = data['vals_l']
    thread_num = data['thread_num']
    features = []
    labels = []


    rid_othres = np.array(list(set(G2.nodes()) - set((x[1] for x in lid_rid))))
    print('Start thread', thread_num)
    count_key_error = 0
    for i, j in tqdm(lid_rid):
        try:
            features.append(feature(G1, i, G2, j))
            labels.append(1)
        except KeyError:
            count_key_error += 1
        # Choose randomly
        j_other = random.choice(rid_othres)
        features.append(feature(G1, i, G2, j_other))
        labels.append(0)
        if random.random() > 0.1:
            j_other = random.choice(rid_othres)
            features.append(feature(G1, i, G2, j_other))
            labels.append(0)
    print('Count if Key Error', count_key_error)
    return (features, labels)

def gen_train_data(lid_rid, G1, G2, save_to, threads = 1):
    # load_cache()
    data_list = prepare_data_for_threads(lid_rid, None, None, G1 = G1, G2 = G2, threads = threads)
    if threads > 1:
        pool = ThreadPool(threads)

        results = pool.map(gen_features, data_list)
        pool.close()
        pool.join()
        features, labels = zip(*results)
        features = list(it.chain.from_iterable(features))
        labels = list(it.chain.from_iterable(labels))
    else:
        features, labels = gen_features(data_list[0])

    pickle.dump((features, labels), open( os.path.join(folder_gen, save_to + '.pickle'), "wb" ))
    # clear_cache()

def load_cache():
    global f_set1s, f_set2s
    f_set1s = dict(pickle.load(open(os.path.join(folder_gen, 'features_G1.pickle'), "rb")))
    f_set2s = dict(pickle.load(open(os.path.join(folder_gen, 'features_G2.pickle'), "rb")))
    print('Cache loaded', len(f_set1s), len(f_set2s))

def clear_cache():
    global f_set1s, f_set2s
    if f_set1s:
        f_set1s.clear()
    if f_set2s:
        f_set2s.clear()
    f_set1s = dict()
    f_set2s = dict()
    print('Clear cache', len(f_set1s), len(f_set2s))

def load_train_data(train_file):
    features, labels = pickle.load(open(os.path.join(folder_gen, train_file + '.pickle'), "rb"))
    return features, labels

def retrain_model(model, fnames):
    features = []
    labels = []
    for fname in fnames:
        fets, labs = load_train_data(folder_gen, fname + '.pickle')
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

def precision_recall(lid_rid, count_v=0):
    is_synthetic_graph = count_v > 0
    if not is_synthetic_graph:
        true_mapping = read_true_mapping(as_set=True)
    count = 0
    check = (lambda x: x in true_mapping) if not is_synthetic_graph else (lambda x: x[0] == x[1])
    for res in lid_rid:
        if check(res):
            count += 1
    precision = count / len(lid_rid)
    recall = count / (len(true_mapping) if not is_synthetic_graph else count_v)
    print('True', count, 'False', len(lid_rid) - count)
    return (precision, recall, 2 * precision * recall / (precision + recall))

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
                try:
                    x = np.array(feature(G1, row_l[0], G2, row_r[0]), ).reshape((1, -1))
                    total_count += 1
                    if model.predict(x) == 1:
                        true_count += 1
                        res_list.append((row_l[0], row_r[0], ratio))
                except KeyError:
                    pass
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

models = {}

def predict_parallel(df_l, df_r, model, G1, G2, name_sim_threshold, threads=4):
    load_cache()
    pool = ThreadPool(threads)
    df_l = df_l[['uid', 'name_vk']].dropna().values
    df_r = df_r[['uid', 'name_inst']].dropna().values
    data_list = prepare_data_for_threads(df_l, df_r, model, G1, G2, threads, name_sim_threshold = name_sim_threshold)
    results = pool.map(find_sim_and_predict, data_list)
    pool.close()
    pool.join()
    for i, res in enumerate(results):
        print(i, 'true count', len(res))
    clear_cache()
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
    pickle.dump((results, lid_rid), open(os.path.join(folder_gen, save_to + '.pickle'), "wb"))
    return results, lid_rid

def filter_others_from_predicted(df_l, df_r, lid_rid):
    def filt(df, index):
        used = set((x[index] for x in lid_rid))
        return df[~df['uid'].isin(used)]
    l, r = filt(df_l, 0), filt(df_r, 1)
    print(len(df_l), len(l), len(lid_rid), len(lid_rid) + len(l))
    return l, r

def read_matches(matches_file_name, threshold, algo_type, with_train = False):
    if '/' in matches_file_name:
        fname = matches_file_name
    else:
        if algo_type == 0:
            folder_name = 'no_repeat'
        elif algo_type == 1:
            folder_name = 'repeat'
        elif algo_type == 2:
            folder_name = 'seed_matches'
        elif algo_type == 3:
            folder_name = 'repeat_steps'
        fname = os.path.join(folder_matches, folder_name, '%.3d' % threshold, matches_file_name)
    print(fname[3:])
    matches = pickle.load(open(fname, 'rb'))
    print('matches len', len(matches))
    miter = iter(matches)
    for i in range(1):
        print(next(miter))

    if with_train:
        fname = fname[:-7] + '_train_seeds.pickle'
        if os.path.isfile(fname):
            seeds_dict_train = pickle.load(open(fname, 'rb'))
            return matches, seeds_dict_train

    return matches

def assure_folder_exists(path):
    folder = os.path.dirname(path)
    print(os.path.abspath(folder))
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_model(model, folder_name):
    folder = os.path.join(folder_gen, 'models', folder_name)
    weight_fname = os.path.join(folder, 'weights.h5')
    assure_folder_exists(weight_fname)
    # serialize weights to HDF5
    model.save_weights(os.path.join(folder, weight_fname))
    print("Saved model to disk")

def load_model(folder_name, feature_amount, log_path='./logs_nn/'):
    import deep_model as dp
    reload(dp)
    [model, [bm_callback, tb_callback]] = dp.get_model(feature_amount=feature_amount, log_path=log_path)
    weight_fname = os.path.join(folder_gen, 'models', folder_name, 'weights.h5')
    # load weights into new model
    model.load_weights(weight_fname)
    print("Loaded model from disk")
    return [model, [bm_callback, tb_callback]]

def generate_struct_cach(Gx, save_to):
    features_g1 = []
    for n in tqdm(Gx.nodes()):
        feature = deg_dist(Gx, n, bins=21, size=50)
        entry = (n, feature)
        features_g1.append(entry)
    pickle.dump(features_g1, open(os.path.join(folder_gen, save_to), "wb"))

def prepare_profile(fname, graph, matches, th, alg_type, cache_l, cache_r, model, train):
    d = {
        'graph' : graph,
        'matches' : matches,
        'alg_type' : alg_type,
        'th' : th,
        'cache_l' : cache_l,
        'cache_r' : cache_r,
        'model' : model,
        'train' : train
    }
    pickle.dump(d, open(os.path.join(folder_gen, 'profiles', fname), 'wb'))
    return d

def load_profile(fname):
    d = pickle.load(open(os.path.join(folder_gen, 'profiles', fname), 'rb'))
    return d

def igraph_to_nx(ig):
    G = nx.from_edgelist([(int(names[x[0]]), int(names[x[1]]))
                      for names in [ig.vs['name']] # simply a let
                      for x in ig.get_edgelist()]) # nx.Graph()
    return G