#!/usr/bin/env python
# -*- coding: utf-8 -*-

import igraph as ig
import sys, time, re, os
from random import randint
import cyrtranslit
import pickle
import gc
sys.path.append('./scripts')

import ml_utils as utils


profile = utils.load_profile(fname='profile_random_t_90_s_80')


def read_edges(f_name):
    print(f_name)
    g = ig.Graph.Read_Ncol(f_name, names=True, directed=False)
    ig.summary(g)
    return g

def enrich_vk_graph(g):
    data_dict = dict()
    pat = re.compile("(\d+),(.*),(.*),(.*)")
    pat_word = re.compile('[^a-zA-Zа-яА-Я\d\s]+')

    g.vs['fname'] = ''
    g.vs['uid'] = None

    with open(os.path.join(utils.folder_data, 'vk_personal2.csv'), 'r') as f:
        for line in f:
            try:
                uid, uname, name1, name2 = pat.match(line).groups()
                name1 = re.sub(pat_word, '', name1).strip().lower()
                name2 = re.sub(pat_word, '', name2).strip().lower()
                data_dict[uid] = (uname, name1 + ' ' + name2)
            except AttributeError:
                print(line)
    for v in g.vs:
        uid = v['name']
        uname, fname = data_dict[uid]
        v['name'] = uname
        v['uid'] = int(uid)
        v['fname'] = cyrtranslit.to_latin(fname, 'ru').replace("'", '')


def enrich_insta_graph(g):
    data_dict = dict()
    pat = re.compile("(\d+),(.*),(.*)")
    pat_word = re.compile('[^a-zA-Zа-яА-Я\d\s]+')

    g.vs['fname'] = ''
    g.vs['uid'] = None

    with open(os.path.join(utils.folder_data, 'inst_personal.csv'), 'r') as f:
        for line in f:
            uid, uname, fname = pat.match(line).groups()
            fname = re.sub(pat_word, '', fname).strip().lower()
            data_dict[uid] = (uname, fname)

    for v in g.vs:
        uid = v['name']
        uname, fname = data_dict[uid]
        v['name'] = uname
        v['uid'] = int(uid)
        v['fname'] = cyrtranslit.to_latin(fname, 'ru').replace("'", '')

def gen_seeds(a_c, lg, rg, fname = None):
    s= time.time()
    if a_c == 0 or fname: ### FILE NAME TO LOAD INITIAL SEEDS FROM MATCHES
        matches_file_name = profile['matches'] if not fname else fname
        lid_rid = utils.read_matches(matches_file_name, threshold = profile['th'], algo_type=profile['alg_type'])
        res = []
        ldict, rdict = {}, {}
        for lv in lg.vs:
            ldict[int(lv['uid'])] = lv.index
        for rv in rg.vs:
            rdict[int(rv['uid'])] = rv.index

        for lid, rid in lid_rid:
            ind_l = ldict[lid]
            ind_r = rdict[rid]
            res.append((ind_l, ind_r))
        print('Len lid_rid and res is same', len(lid_rid) == len(res))
    else:
        res = set()
        while len(res) < a_c:
            try:
                inx = randint(0, lg.vcount())
                inx2 = rg.vs.find(name=lg.vs[inx]['name']).index
                res.add((inx, inx2))
            except ValueError:
                pass
    print('Seed generation time', time.time() - s)
    return res

def load_model():
    model = utils.load_model(profile['model'], feature_amount=85)[0]
    return model
    # return pickle.load(open(os.path.join(utils.folder_gen, 'forest.pickle'), 'rb')) # forest

def load_rand_graph(is_random = False):
    if is_random:
        vk_g, inst_g = pickle.load(open(os.path.join(utils.folder_data, 'random_experiment', profile['graph']), "rb"))
    else:
        vk_g = read_edges(os.path.join(utils.folder_data, 'vk_lid_rid.csv'))
        enrich_vk_graph(vk_g)

        inst_g = read_edges(os.path.join(utils.folder_data, 'inst_lid_rid.csv'))
        enrich_insta_graph(inst_g)
    print('vk_g', vk_g.vcount())
    print('inst_g', inst_g.vcount())
    return vk_g, inst_g

def proceed(alg_GM, a_c, name_sim_threshold, is_repeat, is_model, matches_fname = None):
    if a_c < 0:
        raise Exception('a_c is less than 0.', a_c)

    vk_g, inst_g = load_rand_graph(is_random=True)

    seeds_0 = gen_seeds(a_c, vk_g, inst_g, fname = matches_fname)

    print('Seeds count', len(seeds_0))
    s_time = time.time()

    model = load_model() if is_model else None

    gm = alg_GM(vk_g, inst_g, seeds_0, name_sim_threshold = name_sim_threshold,
        is_repeat = is_repeat, model = model, cache_files = (profile['cache_l'], profile['cache_r']))
    print("Read Graphs time:", time.time() - s_time)
    gm.execute()
    print("Execution time:", gm.time_elapsed)
    return gm



def main():
    """
    argv[1] - algo type : 0-old version. 1-repeat algo with reducing threshold. 2-collect train data with seeds
        if algtype == 1 then name_threshold_step is a delta step
    argv[2] - a_c
    argv[3] - name_threshold
    argv[4] - is_repeat
    argv[5] - is_model
    """
    from expand_UIL2 import ExpandUserIdentity
    alg_type = int(sys.argv[1])
    a_c = int(sys.argv[2])
    name_thres = int(sys.argv[3])
    is_repeat = bool(int(sys.argv[4]))
    is_model = bool(int(sys.argv[5]))
    print('is_repeat', is_repeat)
    print('is_model', is_model)

    threshold_list = [name_thres]
    if alg_type == 1:
        threshold_list = list(range(99, 58, -1 * name_thres))

    last_matches_fname = None
    for i,threshold in enumerate(threshold_list,1):
        print("####################################")
        print("############STEP %.2d (%.2d)############" % (i, threshold))
        print("####################################")
        gm = proceed(ExpandUserIdentity, a_c, name_sim_threshold = threshold,
                     is_repeat = is_repeat, is_model = is_model, matches_fname=last_matches_fname)
        gm.check_result()
        last_matches_fname = gm.save_result()
        a_c = 0
        is_repeat = False
        gc.collect()

# ExpandWhenStuck
def main2():
    def enrich(g):
        for v in g.vs:
            v['fname'] = v['name']

    a_c = int(sys.argv[1])
    from expand_when_stuck import ExpandWhenStuck
    lg = read_edges(os.path.join(utils.folder_data, 'rg_erdos_renyi_100.csv'))
    rg = read_edges(os.path.join(utils.folder_data, 'rg_erdos_renyi_100.csv'))
    enrich(lg)
    enrich(rg)

    seeds_0 = [(randint(0, lg.vcount()),)*2 for i in range(a_c)]
    print(seeds_0)
    s_time = time.time()

    gm = ExpandWhenStuck(lg, rg, seeds_0)
    print("Read Graphs time:", time.time() - s_time)
    gm.execute()
    print("Execution time:", gm.time_elapsed)
    gm.check_result()
    gm.save_result()


# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

if __name__ == "__main__":
    # with PyCallGraph(output=GraphvizOutput()):
    main()