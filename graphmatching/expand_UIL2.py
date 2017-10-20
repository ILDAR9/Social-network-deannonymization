#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from sortedcontainers import SortedSet
from scipy.stats import ks_2samp
import pickle
import os
from fuzzywuzzy import fuzz
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool

class ExpandUserIdentity:

    def __init__(self, lg, rg, seeds_0, name_sim_threshold = 61, is_repeat=False, model = None):
        if is_repeat:
            print("With repeated seeds algorithm is selected")
        else:
            print('NO seeds repeat algorithm is selected ')
        self.lg = lg
        self.rg = rg
        self.seed_0_count = len(seeds_0)
        if is_repeat:
            print('WITH REPEAT!!!')
        self.with_repeat = is_repeat
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
        # marks for every pair mark count > r
        self.score_map = dict()
        self.bad_name = set()
        self.train_data = {}
        if model:
            print('WITH MODEL!!!')
            self.model = model
            self.has_model = True
            self.load_cache()
            self.__get_top = self.__get_top_with_model
            self.inactive_pairs = SortedSet(key=lambda x: (x[2],
                                self.__name_similar(x[1],x[2])/100 + self.__top_with_model(x)))
            self.__decide_seed = self.__decide_seed_with_model
        else:
            self.__get_top = self.__get_top_no_model
            self.inactive_pairs = SortedSet(key=lambda x: (x[2], -1 * self.f_deg_diff(x)))
            self.__decide_seed = self.__decide_seed_no_model
            self.has_model = False

        self.name_sim_threshold = name_sim_threshold
        print('name_sim_threshold', name_sim_threshold)

    def load_cache(self):
        base_folder = '/home/ildar/projects/pycharm/social_network_revealing/graphmatching/'
        folder_data = os.path.join(base_folder, 'data')
        folder_gen = os.path.join(folder_data, 'generated')
        self.f_set1s = dict(pickle.load(open(os.path.join(folder_gen, 'features_G1.pickle'), "rb")))
        self.f_set2s = dict(pickle.load(open(os.path.join(folder_gen, 'features_G2.pickle'), "rb")))
        print('Cache loaded', len(self.f_set1s), len(self.f_set2s))

    def to_str(self, ln ,rn): return '%d|%d' % (ln, rn)

    def untokenize(self, s):
        i = s.index('|')
        return int(s[:i]), int(s[i+1:])

    def f_deg_diff(self, s): return abs(self.lg.degree(s[0]) - self.rg.degree(s[1]))

    def __in_matched(self, lnode, rnode):
        return lnode in self.lNodeM or rnode in self.rNodeM

    def __add_match(self, lnode, rnode, seed_count):
        if not self.has_model :
            self.train_data[(self.lg.vs[lnode]['uid'], self.rg.vs[rnode]['uid'])] = seed_count
        self.matches.add((lnode, rnode))
        self.lNodeM.add(lnode)
        self.rNodeM.add(rnode)

    def __name_similar(self, li, ri):
        return fuzz.token_set_ratio(self.lg.vs[li]['fname'], self.rg.vs[ri]['fname'])


    # def __part_spread_marks(self, data):
    #     seeds = data['seeds']
    #     seeds_collect = {}
    #     old_marks = {}
    #     for seed in tqdm(seeds):
    #         self.used.add(self.to_str(*seed))
    #         self.__spread_mark(*seed, seeds_collect=seeds_collect, old_marks=old_marks)


    # def __spread_marks_parallel(self):
    #     # for all pairs[i, j] of A do
    #     threads = 4
    #     print('start __spread_marks')
    #     self.seeds_collect = {}
    #     self.old_marks = {}
    #     data_list = []
    #     thr_size = len(self.seeds) // threads
    #     for i in range(threads):
    #         s = i * thr_size
    #         e = i * thr_size + thr_size
    #         data = {
    #             'seeds' : self.seeds[s:e] if (i + 1) < threads else self.seeds[s:]
    #         }
    #         data_list.append(data)
    #
    #     pool = ThreadPool(threads)
    #     pool.map(self.__part_spread_marks, data_list)
    #     pool.close()
    #     pool.join()
    #
    #     for seed, marks_count in tqdm(self.seeds_collect.items()):
    #         ID_str = self.to_str(*seed)
    #         m = (seed[0], seed[1], self.old_marks[ID_str])
    #         self.inactive_pairs.discard(m)
    #         self.inactive_pairs.add((seed[0], seed[1], marks_count))
    #
    #     # A <- None
    #     self.seeds.clear()
    #     print("Seed are expanded")

    def __spread_mark(self, lnode, rnode, seeds_collect=None, old_marks = None):
        # add one mark to all neighboring pairs of [i,j]
        if not seeds_collect:
            seeds_collect = {}
            old_marks = {}
            is_from_spread_marks=False
        else:
            is_from_spread_marks=True
        for l_neighbor in self.lg.neighbors(lnode):
            for r_neighbor in self.rg.neighbors(rnode):
                ID_str = self.__decide_seed(l_neighbor, r_neighbor)
                if not ID_str:
                    continue

                val = self.score_map.get(ID_str)
                if not val:
                    self.score_map[ID_str] = 1
                    continue
                if ID_str not in old_marks:
                    old_marks[ID_str] = val
                self.score_map[ID_str] += 1
                seeds_collect[(l_neighbor, r_neighbor)] = val+1
        if is_from_spread_marks:
            return
        for seed, marks_count in seeds_collect.items():
            ID_str = self.to_str(*seed)
            m = (seed[0], seed[1], old_marks[ID_str])
            self.inactive_pairs.discard(m)
            self.inactive_pairs.add((seed[0], seed[1], marks_count))

    def __spread_marks(self):
        # for all pairs[i, j] of A do
        print('start __spread_marks')
        seeds_collect = {}
        old_marks = {}
        for seed in tqdm(self.seeds):
            self.used.add(self.to_str(*seed))
            self.__spread_mark(*seed, seeds_collect = seeds_collect, old_marks = old_marks)

        for seed, marks_count in tqdm(seeds_collect.items()):
            ID_str = self.to_str(*seed)
            m = (seed[0], seed[1], old_marks[ID_str])
            self.inactive_pairs.discard(m)
            self.inactive_pairs.add((seed[0], seed[1], marks_count))

        # A <- None
        self.seeds.clear()
        print("Seed are expanded")

    def __get_top_no_model(self):
        # remove from start matched pairs
        while self.inactive_pairs:
            s = self.inactive_pairs.pop()
            if not self.__in_matched(s[0], s[1]):
                return s
            else:
                self.train_data[(self.lg.vs[s[0]]['uid'], self.rg.vs[s[1]]['uid'])] = s[2]
        return None

    def __get_top_with_model(self):
        # remove from start matched pairs
        while self.inactive_pairs:
            s = self.inactive_pairs.pop()
            if not self.__in_matched(s[0], s[1]):
                return s
        return None

    def __decide_seed_no_model(self, lnode, rnode):
        # i,j not in V_1,V_2(M) and [i,j] not in Z
        if self.__in_matched(lnode, rnode):
            return False
        ID_str = self.to_str(lnode, rnode)
        if ID_str in self.used or ID_str in self.bad_name:
            return False
        if self.__name_similar(lnode, rnode) < self.name_sim_threshold:
            self.bad_name.add(self.to_str(lnode, rnode))
            return False
        return ID_str

    def __decide_seed_with_model(self, lnode, rnode):
        # i,j not in V_1,V_2(M) and [i,j] not in Z
        if self.__in_matched(lnode, rnode):
            return False
        ID_str = self.to_str(lnode, rnode)
        if ID_str in self.used:
            return False
        return ID_str

    def degs(self, node):
        s = []
        for v in node.neighbors():
            s.append(v.degree())
        return s

    def __top_with_model(self, s):
        lv = self.lg.vs[s[0]]
        rv = self.rg.vs[s[1]]

        feature_l = self.f_set1s[lv['uid']]
        feature_r = self.f_set2s[rv['uid']]

        feature_set = feature_l + feature_r
        n_deg = lv.degree()
        m_deg = rv.degree()
        feature_set.append(abs(n_deg - m_deg) / max(n_deg, m_deg, 1))

        # ratio = self.__name_similar(s[0], s[1])
        # feature_set.append(ratio)
        x = np.array(feature_set).reshape((1, -1))
        return self.model.predict(x) == 1  # ratio >= self.name_sim_threshold and


    def __extend_seeds_by_matches(self):
        # A <- all neighbors of M [i,j] not in Z, i,j not in V_1,V_2(M)
        print('__extend_seeds_by_matches')
        for m in tqdm(self.matches):
            lnode, rnode = m
            # all neighbors of M
            for l_neighbor in self.lg.neighbors(lnode):
                for r_neighbor in self.rg.neighbors(rnode):
                    if not self.__decide_seed(l_neighbor, r_neighbor):
                        continue
                    self.seeds.append((l_neighbor,r_neighbor))
        print("Extended seed size: ", len(self.seeds))

    def __garbage_collect(self):
        #####################
        # Garbage collector #
        #####################
        if len(self.score_map) < 20000000 or len(self.inactive_pairs) < 10000000:
            return
        print("Garbage collector:")
        print("\talgorithm: time elapsed: %s" % (time.time() - self.s_time))
        print("\tSize score map: %d" % len(self.score_map))
        for s in tqdm(self.score_map):
            ln, rn = self.untokenize(s)
            if self.__in_matched(ln, rn):
                del self.score_map[s]

        print("\tSize score map (cleared): %d" % len(self.score_map))

        print("\tSize inactive pairs: %d" % len(self.inactive_pairs))
        for p in self.inactive_pairs:
            if self.__in_matched(p[0], p[1]):
                self.inactive_pairs.remove(p)
        print("\tSize inactive pairs (cleared): %d" % len(self.inactive_pairs))

    def __inter_result(self):
        correct, wrong = 0, 0
        for ln, rn in self.matches:
            ln = self.lg.vs[ln]['name']
            rn = self.rg.vs[rn]['name']
            if ln == rn:
                correct += 1
            else:
                wrong += 1
        return correct, wrong

    def __dist_sim(self, vl, vr):
        sl = [v.degree() for v in self.lg.vs[vl].neighbors()]
        sr = [v.degree() for v in self.rg.vs[vr].neighbors()]
        return ks_2samp(sl, sr).pvalue # statistic

    def execute(self):
        self.s_time = time.time()

        iter_num = 0
        show_counter = 0
        show_bound = 50
        show_bound_match = 10
        used_used = set()
        round = 1
        # while |A| > 0 do
        while (len(self.seeds) > 0):
            # while |A| > 0 do
            while(len(self.seeds) > 0):
                iter_num += 1
                print("Iter num: %d\tseed size = %d" % (iter_num, len(self.seeds)))
                # for all pairs[i, j] of A do
                self.__spread_marks()
                print('Done')
                # while there exists an unmatched pair with score at least r+1
                while self.inactive_pairs:
                    show_counter += 1
                    if show_counter % show_bound == 0:
                        print("In progress... (%d)" % len(self.inactive_pairs))
                    # remove from start matched pairs
                    s = self.__get_top()
                    if not s: break
                    elif (show_counter % show_bound == 0):
                        print("[%d] select the unmatched pair [%d,%d]" % (show_counter, s[0], s[1]))
                        print("score map size = %d" % len(self.score_map))
                    lnode, rnode, seed_count = s
                    # add [i,j] to M
                    self.__add_match(lnode, rnode, seed_count)

                    ID_not_active = self.to_str(lnode, rnode)
                    # if [i,j] not in Z
                    if not ID_not_active in self.used:
                        # add [i,j] to Z
                        self.used.add(ID_not_active)
                        # add one marks to all of its neighbouring pairs
                        self.__spread_mark(lnode,rnode)
                    # self.__garbage_collect()
                    if len(self.matches) % show_bound_match == 0:
                        print("Correct = %d, Wrong = %d" % self.__inter_result())


                print("Finish with inactive_pairs")
                # if len(self.bad_name) > 40000000:
                #     self.bad_name.clear()
                #     print("Cleared bad names storage")
                # A <- all neighbors of M [i,j] not in Z, i,j not in V_1,V_2(M)
                self.__extend_seeds_by_matches()
            if self.with_repeat:
                for s in self.used:
                    l_neighbor, r_neighbor = self.untokenize(s)
                    if not self.__in_matched(l_neighbor, r_neighbor) and s not in used_used and \
                                    self.__name_similar(l_neighbor, r_neighbor) >= 99:
                        self.seeds.append((l_neighbor, r_neighbor))
                        used_used.add(s)
                        print('Updated round %d, seed count = %d. used_used = %d' % (round, len(self.seeds),len(used_used)))
                        round += 1
        self.time_elapsed = time.time() - self.s_time

    def assure_folder_exists(self, path):
        folder = os.path.dirname(path)
        print(os.path.abspath(folder))
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save_result(self):
        if self.seed_0_count > 100:
            repeat_name = 'seed_matches'
        else:
            repeat_name = 'repeat' if self.with_repeat else 'no_repeat'
        fname = '%.3d/matches_s_%.2d_th_%.3d_t_%s.pickle' % (self.name_sim_threshold, self.seed_0_count, self.name_sim_threshold, time.strftime("%m-%d_%H:%M:%S"))
        fname = os.path.join('matches', repeat_name, fname)
        self.assure_folder_exists(fname)

        lid_rid = []
        for lnode, rnode in self.matches:
            lid = self.lg.vs[lnode]['uid']
            rid = self.rg.vs[rnode]['uid']
            lid_rid.append((lid,  rid))
        assert len(lid_rid) == len(self.matches)
        pickle.dump(lid_rid, open(fname, 'wb'))
        self.__save_train(fname)
        return fname

    def __save_train(self, f_matches):
        pickle.dump(self.train_data, open(f_matches[:-7] + '_train_seeds.pickle', 'wb'))

    def check_result(self):
        correct, wrong = self.__inter_result()
        msize = len(self.matches)

        recall = float(correct) / min(self.lg.vcount(), self.lg.vcount())
        precision = float(correct) / msize
        f1_score = 2 * (precision*recall / (precision + recall))

        print("------RESULT-------")
        print("\tfor lN = %d, rN = %d, |seed_0| = %d" % (self.lg.vcount(), self.rg.vcount() ,self.seed_0_count))
        print("\tmatched =", msize)
        print("\t\tcorrect = %d; wrong = %d" % (correct, wrong))
        print("\tRecall = %f" % recall)
        print("\tPrecision = %f" % precision)
        print("\tF1-score = %f" % f1_score)