#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, time, os
import pickle
import gc
from importlib import reload
sys.path.append('./scripts')
import ml_utils as utils
reload(utils)
from expand_model_prediction import ExpandModelPrediction

f_prefix = 'data/'

def gen_seeds(fname, threshold, algo_type=0):
    matches_file_name = fname
    lid_rid = utils.read_matches(matches_file_name, threshold=threshold, algo_type=algo_type)
    return lid_rid

def load_model():
    # model = utils.load_model('matches_f85_1hop', feature_amount=85)[0]
    model = utils.load_model('matches_f85_th81', feature_amount=85)[0]
    return model
    # return pickle.load(open(os.path.join(utils.folder_gen, 'forest.pickle'), 'rb')) # forest

def read_graphs():
    reload(utils)
    g1_fname = 'vk_lid_rid.csv'
    g2_fname = 'inst_lid_rid.csv'
    G1, G2 = utils.read_gs(g1_fname, g2_fname, from_raw=False)
    return G1, G2

def proceed():
    lg,rg = read_graphs()

    # seeds_0 = gen_seeds(fname = 'matches_s_10_th_070_t_10-17_17:04:48.pickle')
    # seeds_0 = gen_seeds(fname = 'matches_s_03_th_091_t_10-19_23:31:15.pickle', threshold=91)
    # seeds_0 = gen_seeds(fname='matches_s_03_th_081_t_10-20_00:10:59.pickle', threshold=81)
    seeds_0 = gen_seeds('matches_s_5233_th_099_t_10-20_08:53:15.pickle', threshold=99, algo_type=2)

    print('Seeds count', len(seeds_0))
    s_time = time.time()
    model = load_model()
    gm = ExpandModelPrediction(lg, rg, seeds_0 = seeds_0, model = model)
    print("Read Graphs time:", time.time() - s_time)
    gm.execute()
    print("Execution time:", gm.time_elapsed)
    return gm



def main():
    gm = proceed()
    gm.check_result()
    last_matches_fname = gm.save_result()
    print(last_matches_fname)
    gc.collect()


# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

if __name__ == "__main__":
    # with PyCallGraph(output=GraphvizOutput()):
    main()