import random
import numpy as np
from scipy import stats
from random import choice


def safe_div(x, y):
    if y == 0:
        return 0
    return round(x / y, 4)


class Diffusion:
    def __init__(self, graph_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the list to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.prob_threshold = 0.001

    def getSeedSetProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        ep = 0.0
        for k in range(self.num_product):
            a_n_set = s_set[k].copy()
            a_n_sequence, a_n_sequence2 = [(s, 1) for s in s_set[k]], []
            benefit = self.product_list[k][0]

            while a_n_sequence:
                i_node, i_acc_prob = a_n_sequence.pop(choice([i for i in range(len(a_n_sequence))]))

                # -- notice: prevent the node from owing no receiver --
                if i_node not in self.graph_dict:
                    continue

                i_dict = self.graph_dict[i_node]
                for ii_node in i_dict:
                    if random.random() > i_dict[ii_node]:
                        continue

                    if ii_node in s_total_set:
                        continue
                    if ii_node in a_n_set:
                        continue
                    a_n_set.add(ii_node)

                    # -- purchasing --
                    ep += benefit

                    ii_acc_prob = round(i_acc_prob * i_dict[ii_node], 4)
                    if ii_acc_prob > self.prob_threshold:
                        a_n_sequence2.append((ii_node, ii_acc_prob))

                if not a_n_sequence:
                    a_n_sequence, a_n_sequence2 = a_n_sequence2, a_n_sequence

        return round(ep, 4)


def getProductWeight(prod_list, wallet_dist_name):
    price_list = [prod[2] for prod in prod_list]
    mu, sigma = 0, 1
    if wallet_dist_name == 'm50e25':
        mu = np.mean(price_list)
        sigma = (max(price_list) - mu) / 0.6745
    elif wallet_dist_name == 'm99e96':
        mu = sum(price_list)
        sigma = abs(min(price_list) - mu) / 3
    X = np.arange(0, 2, 0.001)
    Y = stats.norm.sf(X, mu, sigma)
    pw_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

    return pw_list


class DiffusionPW:
    def __init__(self, graph_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the list to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.prob_threshold = 0.001

    def getSeedSetProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        ep = 0.0
        for k in range(self.num_product):
            a_n_set = s_set[k].copy()
            a_n_sequence, a_n_sequence2 = [(s, 1) for s in s_set[k]], []
            benefit = self.product_list[k][0]
            product_weight = self.product_weight_list[k]

            while a_n_sequence:
                i_node, i_acc_prob = a_n_sequence.pop(choice([i for i in range(len(a_n_sequence))]))

                # -- notice: prevent the node from owing no receiver --
                if i_node not in self.graph_dict:
                    continue

                i_dict = self.graph_dict[i_node]
                for ii_node in i_dict:
                    if random.random() > i_dict[ii_node]:
                        continue

                    if ii_node in s_total_set:
                        continue
                    if ii_node in a_n_set:
                        continue
                    a_n_set.add(ii_node)

                    # -- purchasing --
                    ep += benefit * product_weight

                    ii_acc_prob = round(i_acc_prob * i_dict[ii_node], 4)
                    if ii_acc_prob > self.prob_threshold:
                        a_n_sequence2.append((ii_node, ii_acc_prob))

                if not a_n_sequence:
                    a_n_sequence, a_n_sequence2 = a_n_sequence2, a_n_sequence

        return round(ep, 4)


def getExpectedInf(i_dict):
    ei = 0.0
    for item in i_dict:
        acc_prob = 1.0
        for prob in i_dict[item]:
            acc_prob *= (1 - prob)
        ei += (1 - acc_prob)

    return ei


def insertProbAncIntoDict(i_dict, i_node, i_prob, i_anc_set):
    if i_node not in i_dict:
        i_dict[i_node] = [(i_prob, i_anc_set)]
    else:
        i_dict[i_node].append((i_prob, i_anc_set))


def insertProbIntoDict(i_dict, i_node, i_prob):
    if i_node not in i_dict:
        i_dict[i_node] = [i_prob]
    else:
        i_dict[i_node].append(i_prob)


def combineDict(o_dict, n_dict):
    for item in n_dict:
        if item not in o_dict:
            o_dict[item] = n_dict[item]
        else:
            o_dict[item] += n_dict[item]


class DiffusionAccProb:
    def __init__(self, graph_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.prob_threshold = 0.001

    def buildNodeExpectedInfDict(self, s_set, i_node, i_acc_prob):
        i_dict = {}

        if i_node in self.graph_dict:
            for ii_node in self.graph_dict[i_node]:
                if ii_node in s_set:
                    continue
                ii_prob = round(float(self.graph_dict[i_node][ii_node]) * i_acc_prob, 4)

                if ii_prob >= self.prob_threshold:
                    insertProbIntoDict(i_dict, ii_node, ii_prob)

                    if ii_node in self.graph_dict:
                        for iii_node in self.graph_dict[ii_node]:
                            if iii_node in s_set:
                                continue
                            iii_prob = round(float(self.graph_dict[ii_node][iii_node]) * ii_prob, 4)

                            if iii_prob >= self.prob_threshold:
                                insertProbIntoDict(i_dict, iii_node, iii_prob)

                                if iii_node in self.graph_dict:
                                    for iv_node in self.graph_dict[iii_node]:
                                        if iv_node in s_set:
                                            continue
                                        iv_prob = round(float(self.graph_dict[iii_node][iv_node]) * iii_prob, 4)

                                        if iv_prob >= self.prob_threshold:
                                            insertProbIntoDict(i_dict, iv_node, iv_prob)

                                            if iv_node in self.graph_dict and iv_prob > self.prob_threshold:
                                                diff_d = DiffusionAccProb(self.graph_dict, self.product_list)
                                                iv_dict = diff_d.buildNodeExpectedInfDict(s_set, iv_node, iv_prob)
                                                combineDict(i_dict, iv_dict)

        return i_dict

    def buildSeedSetExpectedInfDictBatch(self, s_set, i_node, mep_item_seq, i_mep_item_id_seq, i_acc_prob):
        s_dict_seq = [{} for _ in range(len(mep_item_seq))]

        if i_node in self.graph_dict and i_mep_item_id_seq:
            for ii_node in self.graph_dict[i_node]:
                if ii_node in s_set:
                    continue
                ii_prob = round(float(self.graph_dict[i_node][ii_node]) * i_acc_prob, 4)

                if ii_prob >= self.prob_threshold:
                    ii_mep_item_id_seq = []
                    for mep_item_id in i_mep_item_id_seq:
                        if mep_item_seq[mep_item_id][1] != ii_node:
                            ii_mep_item_id_seq.append(mep_item_id)
                            insertProbIntoDict(s_dict_seq[mep_item_id], ii_node, ii_prob)

                    if ii_node in self.graph_dict and ii_mep_item_id_seq:
                        for iii_node in self.graph_dict[ii_node]:
                            if iii_node in s_set:
                                continue
                            iii_prob = round(float(self.graph_dict[ii_node][iii_node]) * ii_prob, 4)

                            if iii_prob >= self.prob_threshold:
                                iii_mep_item_id_seq = []
                                for mep_item_id in ii_mep_item_id_seq:
                                    if mep_item_seq[mep_item_id][1] != iii_node:
                                        iii_mep_item_id_seq.append(mep_item_id)
                                        insertProbIntoDict(s_dict_seq[mep_item_id], iii_node, iii_prob)

                                if iii_node in self.graph_dict and iii_mep_item_id_seq:
                                    for iv_node in self.graph_dict[iii_node]:
                                        if iv_node in s_set:
                                            continue
                                        iv_prob = round(float(self.graph_dict[iii_node][iv_node]) * iii_prob, 4)

                                        if iv_prob >= self.prob_threshold:
                                            iv_mep_item_id_seq = []
                                            for mep_item_id in iii_mep_item_id_seq:
                                                if mep_item_seq[mep_item_id][1] != iv_node:
                                                    iv_mep_item_id_seq.append(mep_item_id)
                                                    insertProbIntoDict(s_dict_seq[mep_item_id], iv_node, iv_prob)

                                            if iv_node in self.graph_dict and iv_mep_item_id_seq and iv_prob > self.prob_threshold:
                                                diff_d = DiffusionAccProb(self.graph_dict, self.product_list)
                                                iv_dict = diff_d.buildSeedSetExpectedInfDictBatch(s_set, iv_node, mep_item_seq, iv_mep_item_id_seq, iv_prob)
                                                for dl in range(len(mep_item_seq)):
                                                    combineDict(s_dict_seq[dl], iv_dict[dl])

        return s_dict_seq

    def updateExpectedInfBatch(self, s_set, mep_item_seq):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        mep_item_seq = [(mep_item_l[1], mep_item_l[2]) for mep_item_l in mep_item_seq]
        mep_item_dictionary = [{} for _ in range(len(mep_item_seq))]
        diff_d = DiffusionAccProb(self.graph_dict, self.product_list)

        for k in range(self.num_product):
            mep_item_seq_temp = [mep_item_temp for mep_item_temp in mep_item_seq if mep_item_temp[0] == k]
            if mep_item_seq_temp:
                for s in s_set[k]:
                    s_dict_seq = diff_d.buildSeedSetExpectedInfDictBatch(s_total_set, s, mep_item_seq_temp, [mep_item_id for mep_item_id in range(len(mep_item_seq_temp))], 1)
                    for mep_item_seq_temp_item in mep_item_seq_temp:
                        mep_item_id = mep_item_seq.index(mep_item_seq_temp_item)
                        mep_item_s_dict = s_dict_seq.pop(0)
                        mep_item_dictionary[mep_item_id] = mep_item_s_dict

        for mep_item_seq_item in mep_item_seq:
            s_set_t = s_total_set.copy()
            s_set_t.add(mep_item_seq_item[1])
            node_anc_dict = diff_d.buildNodeExpectedInfDict(s_set_t, mep_item_seq_item[1], 1)
            mep_item_seq_id = mep_item_seq.index(mep_item_seq_item)
            combineDict(mep_item_dictionary[mep_item_seq_id], node_anc_dict)

        return mep_item_dictionary


class DiffusionEdgeAccProb:
    def __init__(self, graph_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.prob_threshold = 0.001

    def buildNodeExpectedInfDict(self, s_set, k_prod, i_node, i_acc_prob):
        product_weight = self.product_weight_list[k_prod]
        i_dict = {}

        if i_node in self.graph_dict:
            for ii_node in self.graph_dict[i_node]:
                if ii_node in s_set:
                    continue
                ii_prob = round(float(self.graph_dict[i_node][ii_node]) * i_acc_prob * product_weight, 4)

                if ii_prob >= self.prob_threshold:
                    insertProbIntoDict(i_dict, ii_node, ii_prob)

                    if ii_node in self.graph_dict:
                        for iii_node in self.graph_dict[ii_node]:
                            if iii_node in s_set:
                                continue
                            iii_prob = round(float(self.graph_dict[ii_node][iii_node]) * ii_prob * product_weight, 4)

                            if iii_prob >= self.prob_threshold:
                                insertProbIntoDict(i_dict, iii_node, iii_prob)

                                if iii_node in self.graph_dict:
                                    for iv_node in self.graph_dict[iii_node]:
                                        if iv_node in s_set:
                                            continue
                                        iv_prob = round(float(self.graph_dict[iii_node][iv_node]) * iii_prob * product_weight, 4)

                                        if iv_prob >= self.prob_threshold:
                                            insertProbIntoDict(i_dict, iv_node, iv_prob)

                                            if iv_node in self.graph_dict and iv_prob > self.prob_threshold:
                                                diff_d = DiffusionEdgeAccProb(self.graph_dict, self.product_list, self.product_weight_list)
                                                iv_dict = diff_d.buildNodeExpectedInfDict(s_set, k_prod, iv_node, iv_prob)
                                                combineDict(i_dict, iv_dict)

        return i_dict

    def buildSeedSetExpectedInfDictBatch(self, s_set, k_prod, i_node, mep_item_seq, i_mep_item_id_seq, i_acc_prob):
        product_weight = self.product_weight_list[k_prod]
        s_dict_seq = [{} for _ in range(len(mep_item_seq))]

        if i_node in self.graph_dict and i_mep_item_id_seq:
            for ii_node in self.graph_dict[i_node]:
                if ii_node in s_set:
                    continue
                ii_prob = round(float(self.graph_dict[i_node][ii_node]) * i_acc_prob * product_weight, 4)

                if ii_prob >= self.prob_threshold:
                    ii_mep_item_id_seq = []
                    for mep_item_id in i_mep_item_id_seq:
                        if mep_item_seq[mep_item_id][1] != ii_node:
                            ii_mep_item_id_seq.append(mep_item_id)
                            insertProbIntoDict(s_dict_seq[mep_item_id], ii_node, ii_prob)

                    if ii_node in self.graph_dict and ii_mep_item_id_seq:
                        for iii_node in self.graph_dict[ii_node]:
                            if iii_node in s_set:
                                continue
                            iii_prob = round(float(self.graph_dict[ii_node][iii_node]) * ii_prob * product_weight, 4)

                            if iii_prob >= self.prob_threshold:
                                iii_mep_item_id_seq = []
                                for mep_item_id in ii_mep_item_id_seq:
                                    if mep_item_seq[mep_item_id][1] != iii_node:
                                        iii_mep_item_id_seq.append(mep_item_id)
                                        insertProbIntoDict(s_dict_seq[mep_item_id], iii_node, iii_prob)

                                if iii_node in self.graph_dict and iii_mep_item_id_seq:
                                    for iv_node in self.graph_dict[iii_node]:
                                        if iv_node in s_set:
                                            continue
                                        iv_prob = round(float(self.graph_dict[iii_node][iv_node]) * iii_prob * product_weight, 4)

                                        if iv_prob >= self.prob_threshold:
                                            iv_mep_item_id_seq = []
                                            for mep_item_id in iii_mep_item_id_seq:
                                                if mep_item_seq[mep_item_id][1] != iv_node:
                                                    iv_mep_item_id_seq.append(mep_item_id)
                                                    insertProbIntoDict(s_dict_seq[mep_item_id], iv_node, iv_prob)

                                            if iv_node in self.graph_dict and iv_mep_item_id_seq and iv_prob > self.prob_threshold:
                                                diff_d = DiffusionEdgeAccProb(self.graph_dict, self.product_list, self.product_weight_list)
                                                iv_dict = diff_d.buildSeedSetExpectedInfDictBatch(s_set, k_prod, iv_node, mep_item_seq, iv_mep_item_id_seq, iv_prob)
                                                for dl in range(len(mep_item_seq)):
                                                    combineDict(s_dict_seq[dl], iv_dict[dl])

        return s_dict_seq

    def updateExpectedInfBatch(self, s_set, mep_item_seq):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        mep_item_seq = [(mep_item_l[1], mep_item_l[2]) for mep_item_l in mep_item_seq]
        mep_item_dictionary = [{} for _ in range(len(mep_item_seq))]
        diff_d = DiffusionEdgeAccProb(self.graph_dict, self.product_list, self.product_weight_list)

        for k in range(self.num_product):
            mep_item_seq_temp = [mep_item_temp for mep_item_temp in mep_item_seq if mep_item_temp[0] == k]
            if mep_item_seq_temp:
                for s in s_set[k]:
                    s_dict_seq = diff_d.buildSeedSetExpectedInfDictBatch(s_total_set, k, s, mep_item_seq_temp, [mep_item_id for mep_item_id in range(len(mep_item_seq_temp))], 1)
                    for mep_item_seq_temp_item in mep_item_seq_temp:
                        mep_item_id = mep_item_seq.index(mep_item_seq_temp_item)
                        mep_item_s_dict = s_dict_seq.pop(0)
                        mep_item_dictionary[mep_item_id] = mep_item_s_dict

        for mep_item_seq_item in mep_item_seq:
            s_set_t = s_total_set.copy()
            s_set_t.add(mep_item_seq_item[1])
            node_anc_dict = diff_d.buildNodeExpectedInfDict(s_set_t, mep_item_seq_item[0], mep_item_seq_item[1], 1)
            mep_item_seq_id = mep_item_seq.index(mep_item_seq_item)
            combineDict(mep_item_dictionary[mep_item_seq_id], node_anc_dict)

        return mep_item_dictionary
