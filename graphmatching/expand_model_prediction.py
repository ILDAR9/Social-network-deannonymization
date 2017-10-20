#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pickle
import os
from fuzzywuzzy import fuzz
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool

class ExpandModelPrediction:

    def __init__(self, lg, rg, seeds_0, model):
        self.lg = lg
        self.rg = rg
        self.seed_0_count = len(seeds_0)
        # M < - A_0 ps: A = A_0
        self.lNodeM = set()
        self.rNodeM = set()
        self.matches = set()

        for s in seeds_0:
            lnode, rnode = s
            self.matches.add((lnode, rnode))
            self.lNodeM.add(lnode)
            self.rNodeM.add(rnode)

        self.seeds = list(seeds_0)
        self.used = set()
        self.checked = dict()
        self.__load_cache()
        self.model = model
        self.percolation_bound = 2

    def __load_cache(self):
        base_folder = '/home/ildar/projects/pycharm/social_network_revealing/graphmatching/'
        folder_data = os.path.join(base_folder, 'data')
        folder_gen = os.path.join(folder_data, 'generated')
        self.f_set1s = dict(pickle.load(open(os.path.join(folder_gen, 'features_G1.pickle'), "rb")))
        self.f_set2s = dict(pickle.load(open(os.path.join(folder_gen, 'features_G2.pickle'), "rb")))
        print('Cache loaded', len(self.f_set1s), len(self.f_set2s))

    def __to_str(self, ln ,rn): return '%d|%d' % (ln, rn)

    def __untokenize(self, s):
        i = s.index('|')
        return int(s[:i]), int(s[i+1:])

    def f_deg_diff(self, s): return abs(self.lg.degree(s[0]) - self.rg.degree(s[1]))

    def __in_matched(self, lnode, rnode):
        return lnode in self.lNodeM or rnode in self.rNodeM

    def __add_match(self, lnode, rnode):
        self.matches.add((lnode, rnode))
        self.lNodeM.add(lnode)
        self.rNodeM.add(rnode)

    def __name_similar(self, li, ri):
        return fuzz.token_set_ratio(self.lg.node[li]['fname'], self.rg.node[ri]['fname'])

    def get_degree_level(self, lnode):
        count = len(self.lg[lnode])
        if count < 100:
            return (0,100)
        if count < 300:
            return (101, 300)
        if count < 450:
            return (300, 450)
        return (450, 9999999)

    def __spread_mark(self, lnode, rnode):
        self.used.add(self.__to_str(lnode, rnode))
        checked = self.checked
        # Choose best match

        for l_neighbor in self.lg[lnode]:
            best_rigtht = [None, 0, 0]
            for r_neighbor in self.rg[rnode]:
                if self.__in_matched(l_neighbor, r_neighbor):
                    continue
                ID_str = self.__to_str(l_neighbor, r_neighbor)
                checked[ID_str] = checked.get(ID_str, 0) + 1
                if checked[ID_str] < self.percolation_bound:
                    continue
                pred_proba = self.__decide_seed(l_neighbor, r_neighbor)
                cur_marks = checked[ID_str]
                if pred_proba > best_rigtht[1]: # cur_marks >= best_rigtht[2]
                    best_rigtht[0] = r_neighbor
                    best_rigtht[1] = pred_proba
                    best_rigtht[2] = cur_marks
            if best_rigtht[0]:
                self.__add_match(l_neighbor, best_rigtht[0])


    def __spread_marks(self, data = None):
        seeds = data['seeds'] if data else self.seeds
        next_show_boundary = len(self.matches)
        show_boundary_delta = 50
        print('start __spread_marks')
        for seed in tqdm(seeds):
            self.__spread_mark(*seed)
            if len(self.matches) >= next_show_boundary:
                next_show_boundary += show_boundary_delta
                print("Correct = %d, Wrong = %d" % self.__inter_result())

        # A <- None
        self.seeds.clear()
        print("Seed are expanded")

    def __decide_seed(self, lnode, rnode):
        feature_l = self.f_set1s[lnode]
        feature_r = self.f_set2s[rnode]
        feature_set = feature_l + feature_r
        n_deg = self.lg.degree(lnode)
        m_deg = self.rg.degree(rnode)
        feature_set.append(abs(n_deg - m_deg) / max(n_deg, m_deg, 1))
        x = np.array(feature_set).reshape((1, -1))

        pred = self.model.predict(x)
        ratio = self.__name_similar(lnode, rnode) / 100
        return ratio + pred if ratio > 0 else pred


    def __extend_seeds_by_matches(self):
        # A <- all neighbors of M [i,j] not in Z, i,j not in V_1,V_2(M)
        print('__extend_seeds_by_matches')
        for m in tqdm(self.matches):
            lnode, rnode = m
            # all neighbors of M
            for l_neighbor in self.lg[lnode]:
                for r_neighbor in self.rg[rnode]:
                    # i,j not in V_1,V_2(M) and [i,j] not in Z
                    ID_str = self.__to_str(l_neighbor, r_neighbor)
                    if not (self.__in_matched(l_neighbor, r_neighbor) or ID_str in self.used):
                        self.seeds.append((l_neighbor, r_neighbor))
        print("Extended seed size: ", len(self.seeds))

    def __inter_result(self):
        correct, wrong = 0, 0
        for ln, rn in self.matches:
            ln = self.lg.node[ln]['uname']
            rn = self.rg.node[rn]['uname']
            if ln == rn:
                correct += 1
            else:
                wrong += 1
        return correct, wrong

    def execute(self):
        self.s_time = time.time()
        iter_num = 0
        # while |A| > 0 do
        while(len(self.seeds) > 0):
            iter_num += 1
            print("Iter num: %d\tseed size = %d" % (iter_num, len(self.seeds)))
            # for all pairs[i, j] of A do
            self.__spread_marks() #ToDo repair
            print("Correct = %d, Wrong = %d" % self.__inter_result())
            self.__extend_seeds_by_matches()
            print('Round:', iter_num)
        self.time_elapsed = time.time() - self.s_time

    def assure_folder_exists(self, path):
        folder = os.path.dirname(path)
        print(os.path.abspath(folder))
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save_result(self):
        fname = 'matches_s_%.2d_t_%s.pickle' % (self.seed_0_count,time.strftime("%m-%d_%H:%M:%S"))
        fname = os.path.join('matches', 'model_predict', fname)
        self.assure_folder_exists(fname)

        pickle.dump(list(self.matches), open(fname, 'wb'))
        return fname

    def check_result(self):
        correct, wrong = self.__inter_result()
        msize = len(self.matches)
        recall = float(correct) / min(len(self.lg), len(self.rg))
        precision = float(correct) / msize
        f1_score = 2 * (precision*recall / (precision + recall))

        print("------RESULT-------")
        print("\tfor lN = %d, rN = %d, |seed_0| = %d" % (len(self.lg), len(self.rg) ,self.seed_0_count))
        print("\tmatched =", msize)
        print("\t\tcorrect = %d; wrong = %d" % (correct, wrong))
        print("\tRecall = %f" % recall)
        print("\tPrecision = %f" % precision)
        print("\tF1-score = %f" % f1_score)