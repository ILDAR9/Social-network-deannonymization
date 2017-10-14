#!/usr/bin/env python
# -*- coding: utf-8 -*-

import igraph as ig
import sys, time, re, os
from random import randint
import cyrtranslit
import pickle
sys.path.append('./scripts')

import ml_utils as utils



f_prefix = 'data/'


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

    with open(f_prefix + 'vk_personal2.csv', 'r') as f:
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

    with open(f_prefix + 'inst_personal.csv', 'r') as f:
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

def gen_seeds(a_c, lg, rg):
    s= time.time()
    if a_c == 0: ### FILE NAME TO LOAD INITIAL SEEDS FROM MATCHES
        matches_file_name = 'matches_s_01_th_091_t_10-12_13:19.pickle'
        lid_rid = utils.read_matches(matches_file_name, threshold = 91, is_repeat=True)
        res = []
        ldict, rdict = {}, {}
        for lv in lg.vs:
            ldict[lv['uid']] = lv.index
        for rv in rg.vs:
            rdict[rv['uid']] = rv.index

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
    return pickle.load(open(os.path.join(utils.folder_gen, 'forest.pickle'), 'rb'))

def proceed(alg_GM, a_c, name_sim_threshold, is_repeat, is_model):
    if a_c < 0:
        raise Exception('a_c is less than 0.', a_c)

    inst_g = read_edges(f_prefix + 'inst_lid_rid.csv')
    enrich_insta_graph(inst_g)

    vk_g = read_edges(f_prefix + 'vk_lid_rid.csv')
    enrich_vk_graph(vk_g)

    seeds_0 = gen_seeds(a_c, vk_g, inst_g)

    print('Seeds count', len(seeds_0))
    s_time = time.time()

    model = load_model() if is_model else None

    gm = alg_GM(vk_g, inst_g, seeds_0, name_sim_threshold = name_sim_threshold, is_repeat = is_repeat, model = model)
    print("Read Graphs time:", time.time() - s_time)
    gm.execute()
    print("Execution time:", gm.time_elapsed)
    return gm

def main():
    from expand_UID import ExpandWhenStuck
    a_c = int(sys.argv[1])
    name_thres = float(sys.argv[2])
    is_repeat = bool(int(sys.argv[3]))
    is_model = bool(int(sys.argv[4]))
    print('is_repeat', is_repeat)
    print('is_model', is_model)

    gm = proceed(ExpandWhenStuck, a_c, name_sim_threshold = name_thres, is_repeat = is_repeat, is_model = is_model)

    gm.check_result()
    gm.save_result()

def main2():
    def enrich(g):
        for v in g.vs:
            v['fname'] = v['name']

    a_c = int(sys.argv[1])
    from expand_when_stuck import ExpandWhenStuck
    lg = read_edges(f_prefix + 'rg_erdos_renyi_100.csv')
    rg = read_edges(f_prefix + 'rg_erdos_renyi_100.csv')
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