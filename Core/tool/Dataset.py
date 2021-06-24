#!/usr/bin/env python
# coding:utf-8
'''
Data: 2019-06-02 22:35
Author: Zhou Xin
Abstract:   Loading datas.
'''
import scipy.sparse as sp
import numpy as np
from collections import defaultdict

class Dataset(object):
    '''
    Loading the data file
        trainMatrix: mat.
                     if u-i has target action, mat(u,i)=1; else, mat(u,i)=0
        trianList: list[[]].
                     items that u has target action.
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path,num=0):
        if num == 0:
            self.num_users, self.num_items = 0, 0
            self.trainAuxiliaryPosActionMap = self.load_file_as_map(path + ".train.pos.action")
            self.trainAuxiliaryNegActionMap = self.load_file_as_map(path + ".train.neg.action")
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)
        else:
            self.num_users, self.num_items = 0, 0
            self.trainLevel1, targetuset = self.load_file_as_toplevel(path + ".train.pos.action") #int
            self.trainLevel2 = None
            self.trainLevel3 = None
            self.trainLevel4 = None
            self.trainLevel5 = None
            self.trainLevel6 = None
            self.trainLevel7 = None
            self.levels = defaultdict(list)
            if "sobazaar" in path:#1234567,8
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel4,self.trainLevel5, self.trainLevel6, self.trainLevel7 = self.load_file_as_lowlevel_sobazaar(path + ".train.neg.action")
            elif "xing" in path:#1235,4
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel5 = self.load_file_as_lowlevel_xing(path + ".train.neg.action")
            elif "taobao" or "taobao2014" in path:#1234,5
                levelss, self.trainLevel2, self.trainLevel3, self.trainLevel4 = self.load_file_as_lowlevel_taobao(path + ".train.neg.action")
            for user, level in levelss.items():
                if user in targetuset:
                    level.add(1)
                    level = list(level)
                    level.sort()
                    self.levels[user] = level
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)

    def load_rating_file_as_list(self, filename):
        #[u i]
        ratingList = []
        #[seq1 seq2]
        actionList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\n")[0].split("|")
                ui, actions = arr[0], arr[1]
                u, i = ui.split("\t")
                # if u == "10":
                #     print("a")
                u, i = int(u), int(i)
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                actionList.append(actions)
                # action = actions.split(",")
                # for a in action:
                #     actionList.append(a)
                ratingList.append([u, i])
                line = f.readline()
        return ratingList, actionList
    # def load_rating_file_as_list(self, filename):
    #     #[u i]
    #     ratingList = []
    #     #[a1 a2 a3 a4 a5]
    #     actionList = []
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split("\n")[0].split("|")
    #             ui, actions = arr[0], arr[1]
    #             u, i = ui.split("\t")
    #             u, i = int(u), int(i)
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             ratingList.append([u, i])
    #             actionList.append(aa)
    #             line = f.readline()
    #     return ratingList, actionList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        f = open(filename, "r")
        line = f.readline()
        while line != None and line != "":
            array1 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array1[0]), int(array1[1])
            self.num_users = max(self.num_users, u)
            self.num_items = max(self.num_items, i)
            line = f.readline()
        self.num_users = self.num_users + 1
        self.num_items = self.num_items + 1
        # Construct matrix, list
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)

        u_ = 0
        lists, items = [], []
        f.seek(0,0)
        line = f.readline()
        while line != None and line != "":
            array2 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array2[0]), int(array2[1])
            # mat
            mat[u, i] = 1.0
            #list
            if u_ < u:
                # index = 0
                lists.append(items)
                items = []
                u_ += 1
            items.append(i)
            #next line
            line = f.readline()
        lists.append(items)
        return mat, lists

    def load_file_as_map(self, filename):
        # key=u-i; value=auxiliary action list 1\t2\t3\t4
        map = defaultdict(list)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")#str
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                arr = actions.split(",")
                key = str(u)+"-"+str(i)
                for aa in arr:
                    map[key].append(aa)
                line = f.readline()
        return map
    # def load_file_as_map(self, filename):
    #     # key=u-i; value=auxiliary action list
    #     map = defaultdict(list)
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             array3 = line.split("\n")[0].split("|")
    #             ui, actions = array3[0], array3[1]
    #             u, i = ui.split("\t")#str
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")#str
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             key = str(u)+"-"+str(i)
    #             map[key].append(aa)
    #             line = f.readline()
    #     return map

    def load_file_as_toplevel(self, filename):
        # target items
        uset = set()
        u_ = 0
        toplevel, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                u, i = line.split("|")[0].split("\t")
                u = int(u)
                i = int(i)
                uset.add(u)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                if u_ < u:
                    toplevel.append(items)
                    items=[]
                    u_ += 1
                items.append(i)
                line = f.readline()
            toplevel.append(items)
        return toplevel, uset

    def load_file_as_lowlevel_sobazaar(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        level5 = defaultdict(list)
        level6 = defaultdict(list)
        level7 = defaultdict(list)
        levels = defaultdict(set) # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "sobazaar")
                if top_action == 4:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 1:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level4[u].append(i)
                    levels[u].add(4)
                elif top_action == 5:
                    level5[u].append(i)
                    levels[u].add(5)
                elif top_action == 2:
                    level6[u].append(i)
                    levels[u].add(6)
                elif top_action == 6:
                    level7[u].append(i)
                    levels[u].add(7)
                levels[u].add(8)
                line = f.readline()
        return levels, level2,level3,level4,level5,level6,level7

    def find_top_action(self, a, data):
        b = [int(x) for x in a]
        if data == "sobazaar":
            if 4 in b:
                return 4
            if 1 in b:
                return 1
            if 3 in b:
                return 3
            if 5 in b:
                return 5
            if 2 in b:
                return 2
            if 6 in b:
                return 6
            else:
                return 0
        elif data == "xing":
            if 1 in b:
                return 1
            if 2 in b:
                return 2
            if 3 in b:
                return 3
        elif data == "taobao":
            if 2 in b:
                return 2
            if 3 in b:
                return 3
            if 1 in b:
                return 1

    def load_file_as_lowlevel_xing(self, filename):
        # level2346 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level5 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "xing")
                if top_action == 1:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 2:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level5[u].append(i)
                    levels[u].add(5)
                levels[u].add(4)
                line = f.readline()
        return levels, level2, level3, level5

    def load_file_as_lowlevel_taobao(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "taobao")
                if top_action == 3:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 2:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 1:
                    level4[u].append(i)
                    levels[u].add(4)
                levels[u].add(5)
                line = f.readline()
        return levels, level2, level3, level4

class DatasetXing(object):
    '''
    Loading the data file
        trainMatrix: mat.
                     if u-i has target action, mat(u,i)=1; else, mat(u,i)=0
        trianList: list[[]].
                     items that u has target action.
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path,num=0):
        if num == 0:
            self.num_users, self.num_items = 0, 0
            self.trainAuxiliaryPosActionMap = self.load_file_as_map(path + ".train.pos.action")
            self.trainAuxiliaryNegActionMap = self.load_file_as_map(path + ".train.neg.action")
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)
        else:
            self.num_users, self.num_items = 0, 0
            self.trainLevel1, targetuset = self.load_file_as_toplevel(path + ".train.pos.action") #int
            self.trainLevel2 = None
            self.trainLevel3 = None
            self.trainLevel4 = None
            self.trainLevel5 = None
            self.trainLevel6 = None
            self.trainLevel7 = None
            self.levels = defaultdict(list)
            if "sobazaar" in path:#1234567,8
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel4,self.trainLevel5, self.trainLevel6, self.trainLevel7 = self.load_file_as_lowlevel_sobazaar(path + ".train.neg.action")
            elif "xing" in path:#1235,4
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel5 = self.load_file_as_lowlevel_xing(path + ".train.neg.action")
            elif "taobao" in path:#1234,5
                levelss, self.trainLevel2, self.trainLevel3, self.trainLevel4 = self.load_file_as_lowlevel_taobao(path + ".train.neg.action")
            for user, level in levelss.items():
                if user in targetuset:
                    level.add(1)
                    level = list(level)
                    level.sort()
                    self.levels[user] = level
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)

    def load_rating_file_as_list(self, filename):
        #[u i]
        ratingList = []
        #[seq1 seq2]
        actionList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\n")[0].split("|")
                ui, actions = arr[0], arr[1]
                u, i = ui.split("\t")
                # if u == "10":
                #     print("a")
                u, i = int(u), int(i)
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                actionList.append(actions)
                # action = actions.split(",")
                # for a in action:
                #     actionList.append(a)
                ratingList.append([u, i])
                line = f.readline()
        return ratingList, actionList
    # def load_rating_file_as_list(self, filename):
    #     #[u i]
    #     ratingList = []
    #     #[a1 a2 a3 a4 a5]
    #     actionList = []
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split("\n")[0].split("|")
    #             ui, actions = arr[0], arr[1]
    #             u, i = ui.split("\t")
    #             u, i = int(u), int(i)
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             ratingList.append([u, i])
    #             actionList.append(aa)
    #             line = f.readline()
    #     return ratingList, actionList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        f = open(filename, "r")
        line = f.readline()
        while line != None and line != "":
            array1 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array1[0]), int(array1[1])
            self.num_users = max(self.num_users, u)
            self.num_items = max(self.num_items, i)
            line = f.readline()
        self.num_users = self.num_users + 1
        self.num_items = self.num_items + 2
        # Construct matrix, list
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)

        u_ = 0
        lists, items = [], []
        f.seek(0,0)
        line = f.readline()
        while line != None and line != "":
            array2 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array2[0]), int(array2[1])
            # mat
            mat[u, i] = 1.0
            #list
            if u_ < u:
                # index = 0
                lists.append(items)
                items = []
                u_ += 1
            items.append(i)
            #next line
            line = f.readline()
        lists.append(items)
        return mat, lists

    def load_file_as_map(self, filename):
        # key=u-i; value=auxiliary action list 1\t2\t3\t4
        map = defaultdict(list)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")#str
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                arr = actions.split(",")
                key = str(u)+"-"+str(i)
                for aa in arr:
                    map[key].append(aa)
                line = f.readline()
        return map
    # def load_file_as_map(self, filename):
    #     # key=u-i; value=auxiliary action list
    #     map = defaultdict(list)
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             array3 = line.split("\n")[0].split("|")
    #             ui, actions = array3[0], array3[1]
    #             u, i = ui.split("\t")#str
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")#str
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             key = str(u)+"-"+str(i)
    #             map[key].append(aa)
    #             line = f.readline()
    #     return map

    def load_file_as_toplevel(self, filename):
        # target items
        uset = set()
        u_ = 0
        toplevel, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                u, i = line.split("|")[0].split("\t")
                u = int(u)
                i = int(i)
                uset.add(u)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                if u_ < u:
                    toplevel.append(items)
                    items=[]
                    u_ += 1
                items.append(i)
                line = f.readline()
            toplevel.append(items)
        return toplevel, uset

    def load_file_as_lowlevel_sobazaar(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        level5 = defaultdict(list)
        level6 = defaultdict(list)
        level7 = defaultdict(list)
        levels = defaultdict(set) # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "sobazaar")
                if top_action == 4:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 1:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level4[u].append(i)
                    levels[u].add(4)
                elif top_action == 5:
                    level5[u].append(i)
                    levels[u].add(5)
                elif top_action == 2:
                    level6[u].append(i)
                    levels[u].add(6)
                elif top_action == 6:
                    level7[u].append(i)
                    levels[u].add(7)
                levels[u].add(8)
                line = f.readline()
        return levels, level2,level3,level4,level5,level6,level7

    def find_top_action(self, a, data):
        b = [int(x) for x in a]
        if data == "sobazaar":
            if 4 in b:
                return 4
            if 1 in b:
                return 1
            if 3 in b:
                return 3
            if 5 in b:
                return 5
            if 2 in b:
                return 2
            if 6 in b:
                return 6
            else:
                return 0
        elif data == "xing":
            if 1 in b:
                return 1
            if 2 in b:
                return 2
            if 3 in b:
                return 3
        elif data == "taobao":
            if 2 in b:
                return 2
            if 3 in b:
                return 3
            if 1 in b:
                return 1

    def load_file_as_lowlevel_xing(self, filename):
        # level2346 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level5 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "xing")
                if top_action == 1:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 2:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level5[u].append(i)
                    levels[u].add(5)
                levels[u].add(4)
                line = f.readline()
        return levels, level2, level3, level5

    def load_file_as_lowlevel_taobao(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "taobao")
                if top_action == 2:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 3:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 1:
                    level4[u].append(i)
                    levels[u].add(4)
                levels[u].add(5)
                line = f.readline()
        return levels, level2, level3, level4

class DatasetTaobao(object):
    '''
    Loading the data file
        trainMatrix: mat.
                     if u-i has target action, mat(u,i)=1; else, mat(u,i)=0
        trianList: list[[]].
                     items that u has target action.
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path,num=0):
        if num == 0:
            self.num_users, self.num_items = 0, 0
            self.trainAuxiliaryPosActionMap = self.load_file_as_map(path + ".train.pos.action")
            self.trainAuxiliaryNegActionMap = self.load_file_as_map(path + ".train.neg.action")
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)
        else:
            self.num_users, self.num_items = 0, 0
            self.trainLevel1, targetuset = self.load_file_as_toplevel(path + ".train.pos.action") #int
            self.trainLevel2 = None
            self.trainLevel3 = None
            self.trainLevel4 = None
            self.trainLevel5 = None
            self.trainLevel6 = None
            self.trainLevel7 = None
            self.levels = defaultdict(list)
            if "sobazaar" in path:#1234567,8
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel4,self.trainLevel5, self.trainLevel6, self.trainLevel7 = self.load_file_as_lowlevel_sobazaar(path + ".train.neg.action")
            elif "xing" in path:#1235,4
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel5 = self.load_file_as_lowlevel_xing(path + ".train.neg.action")
            elif "taobao" in path:#1234,5
                levelss, self.trainLevel2, self.trainLevel3, self.trainLevel4 = self.load_file_as_lowlevel_taobao(path + ".train.neg.action")
            for user, level in levelss.items():
                if user in targetuset:
                    level.add(1)
                    level = list(level)
                    level.sort()
                    self.levels[user] = level
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)

    def load_rating_file_as_list(self, filename):
        #[u i]
        ratingList = []
        #[seq1 seq2]
        actionList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\n")[0].split("|")
                ui, actions = arr[0], arr[1]
                u, i = ui.split("\t")
                # if u == "10":
                #     print("a")
                u, i = int(u), int(i)
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                actionList.append(actions)
                # action = actions.split(",")
                # for a in action:
                #     actionList.append(a)
                ratingList.append([u, i])
                line = f.readline()
        return ratingList, actionList
    # def load_rating_file_as_list(self, filename):
    #     #[u i]
    #     ratingList = []
    #     #[a1 a2 a3 a4 a5]
    #     actionList = []
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split("\n")[0].split("|")
    #             ui, actions = arr[0], arr[1]
    #             u, i = ui.split("\t")
    #             u, i = int(u), int(i)
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             ratingList.append([u, i])
    #             actionList.append(aa)
    #             line = f.readline()
    #     return ratingList, actionList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        f = open(filename, "r")
        line = f.readline()
        while line != None and line != "":
            array1 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array1[0]), int(array1[1])
            self.num_users = max(self.num_users, u)
            self.num_items = max(self.num_items, i)
            line = f.readline()
        self.num_users = self.num_users + 1
        self.num_items = self.num_items + 37
        # Construct matrix, list
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)

        u_ = 0
        lists, items = [], []
        f.seek(0,0)
        line = f.readline()
        while line != None and line != "":
            array2 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array2[0]), int(array2[1])
            # mat
            mat[u, i] = 1.0
            #list
            if u_ < u:
                # index = 0
                lists.append(items)
                items = []
                u_ += 1
            items.append(i)
            #next line
            line = f.readline()
        lists.append(items)
        return mat, lists

    def load_file_as_map(self, filename):
        # key=u-i; value=auxiliary action list 1\t2\t3\t4
        map = defaultdict(list)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")#str
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                arr = actions.split(",")
                key = str(u)+"-"+str(i)
                for aa in arr:
                    map[key].append(aa)
                line = f.readline()
        return map
    # def load_file_as_map(self, filename):
    #     # key=u-i; value=auxiliary action list
    #     map = defaultdict(list)
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             array3 = line.split("\n")[0].split("|")
    #             ui, actions = array3[0], array3[1]
    #             u, i = ui.split("\t")#str
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")#str
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             key = str(u)+"-"+str(i)
    #             map[key].append(aa)
    #             line = f.readline()
    #     return map

    def load_file_as_toplevel(self, filename):
        # target items
        uset = set()
        u_ = 0
        toplevel, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                u, i = line.split("|")[0].split("\t")
                u = int(u)
                i = int(i)
                uset.add(u)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                if u_ < u:
                    toplevel.append(items)
                    items=[]
                    u_ += 1
                items.append(i)
                line = f.readline()
            toplevel.append(items)
        return toplevel, uset

    def load_file_as_lowlevel_sobazaar(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        level5 = defaultdict(list)
        level6 = defaultdict(list)
        level7 = defaultdict(list)
        levels = defaultdict(set) # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "sobazaar")
                if top_action == 4:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 1:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level4[u].append(i)
                    levels[u].add(4)
                elif top_action == 5:
                    level5[u].append(i)
                    levels[u].add(5)
                elif top_action == 2:
                    level6[u].append(i)
                    levels[u].add(6)
                elif top_action == 6:
                    level7[u].append(i)
                    levels[u].add(7)
                levels[u].add(8)
                line = f.readline()
        return levels, level2,level3,level4,level5,level6,level7

    def find_top_action(self, a, data):
        b = [int(x) for x in a]
        if data == "sobazaar":
            if 4 in b:
                return 4
            if 1 in b:
                return 1
            if 3 in b:
                return 3
            if 5 in b:
                return 5
            if 2 in b:
                return 2
            if 6 in b:
                return 6
            else:
                return 0
        elif data == "xing":
            if 1 in b:
                return 1
            if 2 in b:
                return 2
            if 3 in b:
                return 3
        elif data == "taobao":
            if 2 in b:
                return 2
            if 3 in b:
                return 3
            if 1 in b:
                return 1

    def load_file_as_lowlevel_xing(self, filename):
        # level2346 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level5 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "xing")
                if top_action == 1:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 2:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level5[u].append(i)
                    levels[u].add(5)
                levels[u].add(4)
                line = f.readline()
        return levels, level2, level3, level5

    def load_file_as_lowlevel_taobao(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "taobao")
                if top_action == 3:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 2:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 1:
                    level4[u].append(i)
                    levels[u].add(4)
                levels[u].add(5)
                line = f.readline()
        return levels, level2, level3, level4

class DatasetTaobao14(object):
    '''
    Loading the data file
        trainMatrix: mat.
                     if u-i has target action, mat(u,i)=1; else, mat(u,i)=0
        trianList: list[[]].
                     items that u has target action.
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path,num=0):
        if num == 0:
            self.num_users, self.num_items = 0, 0
            self.trainAuxiliaryPosActionMap = self.load_file_as_map(path + ".train.pos.action")
            self.trainAuxiliaryNegActionMap = self.load_file_as_map(path + ".train.neg.action")
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)
        else:
            self.num_users, self.num_items = 0, 0
            self.trainLevel1, targetuset = self.load_file_as_toplevel(path + ".train.pos.action") #int
            self.trainLevel2 = None
            self.trainLevel3 = None
            self.trainLevel4 = None
            self.trainLevel5 = None
            self.trainLevel6 = None
            self.trainLevel7 = None
            self.levels = defaultdict(list)
            if "sobazaar" in path:#1234567,8
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel4,self.trainLevel5, self.trainLevel6, self.trainLevel7 = self.load_file_as_lowlevel_sobazaar(path + ".train.neg.action")
            elif "xing" in path:#1235,4
                levelss, self.trainLevel2,self.trainLevel3, self.trainLevel5 = self.load_file_as_lowlevel_xing(path + ".train.neg.action")
            elif "taobao" or "taobao2014" or "taobao14" in path:#1234,5
                levelss, self.trainLevel2, self.trainLevel3, self.trainLevel4 = self.load_file_as_lowlevel_taobao(path + ".train.neg.action")
            for user, level in levelss.items():
                if user in targetuset:
                    level.add(1)
                    level = list(level)
                    level.sort()
                    self.levels[user] = level
            self.testRatings, self.testAuxiliaryActionList = self.load_rating_file_as_list(path + ".test.action")
            self.validateRatings, self.validateAuxiliaryActionList = self.load_rating_file_as_list(path + ".validate.action")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
            self.trainMatrix, self.trainList = self.load_training_file(path + ".train.pos.action")
            assert len(self.testRatings) == len(self.testNegatives)

    def load_rating_file_as_list(self, filename):
        #[u i]
        ratingList = []
        #[seq1 seq2]
        actionList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\n")[0].split("|")
                ui, actions = arr[0], arr[1]
                u, i = ui.split("\t")
                # if u == "10":
                #     print("a")
                u, i = int(u), int(i)
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                actionList.append(actions)
                # action = actions.split(",")
                # for a in action:
                #     actionList.append(a)
                ratingList.append([u, i])
                line = f.readline()
        return ratingList, actionList
    # def load_rating_file_as_list(self, filename):
    #     #[u i]
    #     ratingList = []
    #     #[a1 a2 a3 a4 a5]
    #     actionList = []
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split("\n")[0].split("|")
    #             ui, actions = arr[0], arr[1]
    #             u, i = ui.split("\t")
    #             u, i = int(u), int(i)
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             ratingList.append([u, i])
    #             actionList.append(aa)
    #             line = f.readline()
    #     return ratingList, actionList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        f = open(filename, "r")
        line = f.readline()
        while line != None and line != "":
            array1 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array1[0]), int(array1[1])
            self.num_users = max(self.num_users, u)
            self.num_items = max(self.num_items, i)
            line = f.readline()
        self.num_users = self.num_users + 1
        # self.num_items = self.num_items + 99
        self.num_items = 811063
        # Construct matrix, list
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)

        u_ = 0
        lists, items = [], []
        f.seek(0,0)
        line = f.readline()
        while line != None and line != "":
            array2 = line.split("\n")[0].split("|")[0].split("\t")
            u, i = int(array2[0]), int(array2[1])
            # mat
            mat[u, i] = 1.0
            #list
            if u_ < u:
                # index = 0
                lists.append(items)
                items = []
                u_ += 1
            items.append(i)
            #next line
            line = f.readline()
        lists.append(items)
        return mat, lists

    def load_file_as_map(self, filename):
        # key=u-i; value=auxiliary action list 1\t2\t3\t4
        map = defaultdict(list)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")#str
                self.num_users = max(self.num_users, int(u))
                self.num_items = max(self.num_items, int(i))
                arr = actions.split(",")
                key = str(u)+"-"+str(i)
                for aa in arr:
                    map[key].append(aa)
                line = f.readline()
        return map
    # def load_file_as_map(self, filename):
    #     # key=u-i; value=auxiliary action list
    #     map = defaultdict(list)
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             array3 = line.split("\n")[0].split("|")
    #             ui, actions = array3[0], array3[1]
    #             u, i = ui.split("\t")#str
    #             self.num_users = max(self.num_users, int(u))
    #             self.num_items = max(self.num_items, int(i))
    #             action = actions.split("\t")#str
    #             aa = []
    #             for a in action:
    #                 a = int(a)
    #                 aa.append(a)
    #             key = str(u)+"-"+str(i)
    #             map[key].append(aa)
    #             line = f.readline()
    #     return map

    def load_file_as_toplevel(self, filename):
        # target items
        uset = set()
        u_ = 0
        toplevel, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                u, i = line.split("|")[0].split("\t")
                u = int(u)
                i = int(i)
                uset.add(u)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                if u_ < u:
                    toplevel.append(items)
                    items=[]
                    u_ += 1
                items.append(i)
                line = f.readline()
            toplevel.append(items)
        return toplevel, uset

    def load_file_as_lowlevel_sobazaar(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        level5 = defaultdict(list)
        level6 = defaultdict(list)
        level7 = defaultdict(list)
        levels = defaultdict(set) # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "sobazaar")
                if top_action == 4:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 1:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level4[u].append(i)
                    levels[u].add(4)
                elif top_action == 5:
                    level5[u].append(i)
                    levels[u].add(5)
                elif top_action == 2:
                    level6[u].append(i)
                    levels[u].add(6)
                elif top_action == 6:
                    level7[u].append(i)
                    levels[u].add(7)
                levels[u].add(8)
                line = f.readline()
        return levels, level2,level3,level4,level5,level6,level7

    def find_top_action(self, a, data):
        b = [int(x) for x in a]
        if data == "sobazaar":
            if 4 in b:
                return 4
            if 1 in b:
                return 1
            if 3 in b:
                return 3
            if 5 in b:
                return 5
            if 2 in b:
                return 2
            if 6 in b:
                return 6
            else:
                return 0
        elif data == "xing":
            if 1 in b:
                return 1
            if 2 in b:
                return 2
            if 3 in b:
                return 3
        elif data == "taobao":
            if 2 in b:
                return 2
            if 3 in b:
                return 3
            if 1 in b:
                return 1

    def load_file_as_lowlevel_xing(self, filename):
        # level2346 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level5 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "xing")
                if top_action == 1:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 2:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 3:
                    level5[u].append(i)
                    levels[u].add(5)
                levels[u].add(4)
                line = f.readline()
        return levels, level2, level3, level5

    def load_file_as_lowlevel_taobao(self, filename):
        # level2-7 items
        level2 = defaultdict(list)
        level3 = defaultdict(list)
        level4 = defaultdict(list)
        levels = defaultdict(set)  # which levels does the user involves
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                array3 = line.split("\n")[0].split("|")
                ui, actions = array3[0], array3[1]
                u, i = ui.split("\t")  # str
                u = int(u)
                i = int(i)
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                action = actions.split("\t")  # str
                top_action = self.find_top_action(action, "taobao")
                if top_action == 3:
                    level2[u].append(i)
                    levels[u].add(2)
                elif top_action == 2:
                    level3[u].append(i)
                    levels[u].add(3)
                elif top_action == 1:
                    level4[u].append(i)
                    levels[u].add(4)
                levels[u].add(5)
                line = f.readline()
        return levels, level2, level3, level4
