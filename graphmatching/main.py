#!/usr/bin/env python
# -*- coding: utf-8 -*-

import igraph as ig
import sys, time, re
from random import randint
import cyrtranslit

f_prefix = 'data/'


def enrich_vk_graph(g):
    inst_dict = dict()
    pat = re.compile("(\d+),(.*),(.*),(.*)")
    pat_word = re.compile('[^a-zA-Zа-яА-Я\d\s]+')

    with open(f_prefix + 'vk_personal2.csv', 'r') as f:
        for line in f:
            try:
                uid, uname, name1, name2 = pat.match(line).groups()
                name1 = re.sub(pat_word, '', name1).strip().lower()
                name2 = re.sub(pat_word, '', name2).strip().lower()
                inst_dict[uid] = (uname, name1 + ' ' + name2)
            except AttributeError:
                print(line)
    for v in g.vs:
        uname, fname = inst_dict[v['name']]
        v['name'] = uname
        v['fname'] = cyrtranslit.to_latin(fname, 'ru').replace("'", '')

def enrich_insta_graph(g):
    inst_dict = dict()
    pat = re.compile("(\d+),(.*),(.*)")
    pat_word = re.compile('[^a-zA-Zа-яА-Я\d\s]+')

    with open(f_prefix + 'inst_personal.csv', 'r') as f:
        for line in f:
            uid, uname, fname = pat.match(line).groups()
            fname = re.sub(pat_word, '', fname).strip().lower()
            inst_dict[uid] = (uname, fname)

    for v in g.vs:
        uname, fname = inst_dict[v['name']]
        v['name'] = uname
        v['fname'] = cyrtranslit.to_latin(fname, 'ru').replace("'", '')

def read_edges(f_name):
    print(f_name)
    g = ig.Graph.Read_Ncol(f_name, names=True, directed=False)
    ig.summary(g)
    return g

def gen_seeds(seed_c, lg, rg):
    res = set()
    i = 0
    while len(res) < seed_c:
        try:
            inx = randint(0, lg.vcount())
            inx2 = rg.vs.find(name=lg.vs[inx]['name']).index
            res.add((inx, inx2))
        except ValueError:
            pass
    return res

def proceed(alg_GM, a_c, name_sim_threshold):
    inst_g = read_edges(f_prefix + 'inst_lid_rid.csv')
    enrich_insta_graph(inst_g)

    vk_g = read_edges(f_prefix + 'vk_lid_rid.csv')
    enrich_vk_graph(vk_g)

    seeds_0 = gen_seeds(a_c, vk_g, inst_g)
    print(seeds_0)
    s_time = time.time()

    gm = alg_GM(vk_g, inst_g, seeds_0, name_sim_threshold = name_sim_threshold)
    print("Read Graphs time:", time.time() - s_time)
    gm.execute()
    print("Execution time:", gm.time_elapsed)
    return gm

def main():
    from expand_UID import ExpandWhenStuck
    a_c = int(sys.argv[1])
    name_thres = float(sys.argv[2])

    gm = proceed(ExpandWhenStuck, a_c, name_sim_threshold = name_thres)

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