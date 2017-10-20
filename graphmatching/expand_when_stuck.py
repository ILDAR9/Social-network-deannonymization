#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from sortedcontainers import SortedSet
from difflib import SequenceMatcher

class ExpandWhenStuck:

    def __init__(self, lg, rg, seeds_0):
        print("With name extension algorithm is selected")
        self.lg = lg
        self.rg = rg
        self.seed_0_count = len(seeds_0)

        # M < - A_0 ps: A = A_0
        self.lNodeM = set()
        self.rNodeM = set()
        self.matches = set()
        for s in seeds_0: self.__add_match(*s)
        self.seeds = list(seeds_0)
        self.used = set()
        # marks for every pair mark count > r
        self.inactive_pairs = SortedSet(key = lambda x : x[2])
        self.score_map = dict()
        self.bad_name = set()
        self.seq = SequenceMatcher()

    def to_str(self, ln ,rn): return '%d|%d' % (ln, rn)

    def untokenize(self, s):
        i = s.index('|')
        return s[:i], s[i+1:]

    def f_deg_diff(self, s): return abs(self.lg.degree(s[0]) - self.rg.degree(s[1]))

    def __in_matched(self, lnode, rnode):
        return lnode in self.lNodeM or rnode in self.rNodeM

    def __add_match(self, lnode, rnode):
        self.matches.add((lnode, rnode))
        self.lNodeM.add(lnode)
        self.rNodeM.add(rnode)

    def __name_similar(self, li, ri):
        self.seq.set_seqs(self.lg.vs[li]['fname'], self.rg.vs[ri]['fname'])
        return self.seq.ratio()

    def __spread_mark(self, lnode, rnode):
        # add one mark to all neighboring pairs of [i,j]
        for l_neighbor in self.lg.neighbors(lnode):
            for r_neighbor in self.rg.neighbors(rnode):
                if self.__in_matched(l_neighbor, r_neighbor):
                    continue
                ID_str = self.to_str(l_neighbor, r_neighbor)
                if ID_str in self.bad_name:
                    continue
                if self.__name_similar(l_neighbor, r_neighbor) < 0.61:
                    self.bad_name.add(ID_str)
                    continue

                ID_neighbor = self.to_str(l_neighbor, r_neighbor)
                val = self.score_map.get(ID_neighbor)
                if not val:
                    self.score_map[ID_neighbor] = 1
                    continue
                self.score_map[ID_neighbor] += 1
                m = (l_neighbor, r_neighbor, val)
                try:
                    self.inactive_pairs.remove(val)
                except KeyError:
                    pass
                self.inactive_pairs.add((m[0], m[1], val + 1))


    def __spread_marks(self):
        # for all pairs[i, j] of A do
        for seed in self.seeds:
            self.used.add(self.to_str(*seed))
            self.__spread_mark(*seed)
        # A <- None
        self.seeds.clear()
        print("Seed are expanded")

    def __get_top(self):
        # remove from start matched pairs
        while self.inactive_pairs:
            s = self.inactive_pairs.pop()
            if not self.__in_matched(s[0], s[1]):
                # self.inactive_pairs.add(s)
                return s

    def __get_top_min_deg(self):
        s = self.__get_top()
        if not s: return None
        min_deg_diff = self.f_deg_diff(s)
        marks_cnt = s[2]
        set_back = []
        while self.inactive_pairs:
            st = self.inactive_pairs.pop()
            if self.__in_matched(st[0], st[1]):
                continue
            if st[2] == marks_cnt & min_deg_diff > 0:
                t_deg_diff = self.f_deg_diff(st)
                if t_deg_diff < min_deg_diff:
                    set_back.append(s)
                    s = st
                    min_deg_diff = t_deg_diff
                else:
                    set_back.append(st)
            else:
                set_back.append(st)
                break
        self.inactive_pairs.update(set_back)
        print("selected TOP(%f): %s : %s\t%s : %s" % (self.__name_similar(s[0], s[1]),
        self.lg.vs[s[0]]['name'], self.rg.vs[s[1]]['name'], self.lg.vs[s[0]]['fname'],
        self.rg.vs[s[1]]['fname']))
        return s

    def __extend_seeds_by_matches(self):
        # A <- all neighbors of M [i,j] not in Z, i,j not in V_1,V_2(M)
        for m in self.matches:
            lnode, rnode = m
            # all neighbors of M
            for l_neighbor in self.lg.neighbors(lnode):
                for r_neighbor in self.rg.neighbors(rnode):
                    # i,j not in V_1,V_2(M) and [i,j] not in Z
                    ID_str = self.to_str(l_neighbor, r_neighbor)
                    if self.__in_matched(l_neighbor, r_neighbor) or ID_str in self.used or ID_str in self.bad_name:
                        continue

                    if self.__name_similar(l_neighbor, r_neighbor) < 0.4:
                        self.bad_name.add(ID_str)
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
        for s in self.score_map:
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


    def execute(self):
        self.s_time = time.time()

        iter_num = 0
        show_counter = 0
        show_bound = 150
        # while |A| > 0 do
        while(len(self.seeds) > 0):
            iter_num += 1
            print("Iter num: %d\tseed size = %d" % (iter_num, len(self.seeds)))
            # for all pairs[i, j] of A do
            self.__spread_marks()
            # while there exists an unmatched pair with score at least r+1
            while self.inactive_pairs:
                show_counter += 1
                if show_counter % show_bound == 0:
                    print("In progress... (%d)" % len(self.inactive_pairs))
                # remove from start matched pairs
                s = self.__get_top_min_deg()
                if (show_counter % show_bound == 0):
                    print("[%d] select the unmatched pair [%d,%d]" % (show_counter, s[0], s[1]))
                    print("score map size = %d" % len(self.score_map))
                if not s: break
                lnode, rnode = s[:2]
                # add [i,j] to M
                self.__add_match(lnode, rnode)

                ID_not_active = self.to_str(lnode, rnode)
                # if [i,j] not in Z
                if not ID_not_active in self.used:
                    # add [i,j] to Z
                    self.used.add(ID_not_active)
                    # add one marks to all of its neighbouring pairs
                    self.__spread_mark(lnode,rnode)
                self.__garbage_collect()
                if len(self.matches) % 100 == 0:
                    print("Correct = %d, Wrong = %d" % self.__inter_result())


            print("Finish with inactive_pairs")
            if len(self.bad_name) > 40000000:
                self.bad_name.clear()
                print("Cleared bad names storage")
            # A <- all neighbors of M [i,j] not in Z, i,j not in V_1,V_2(M)
            self.__extend_seeds_by_matches()
        self.time_elapsed = time.time() - self.s_time

    def save_result(self):
        with open('matches/matches_t(%s).csv' % time.time(), 'w') as f:
            f.write('\n'.join('%d %d' for x, y in self.matches))

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