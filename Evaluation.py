from Initialization import *
from random import *
import time
import copy
import os


class Evaluation:
    def __init__(self, graph_dict, prod_list, ppp, wpiwp):
        ### graph_dict: (dict) the graph
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_product: (int) the kinds of products
        ### ppp: (int) the strategy of personal purchasing prob.
        ### wpiwp: (bool) whether passing the information with purchasing
        self.graph_dict = graph_dict
        self.product_list = prod_list
        self.num_product = len(prod_list)
        self.ppp = ppp
        self.wpiwp = wpiwp

    def setPersonalPurchasingProbDict(self, wallet_dict):
        # -- according to ppp, initialize the ppp_dict --
        ### ppp_dict: (list) the list of personal purchasing prob. for all combinations of nodes and products
        ### ppp_dict[k]: (list) the list of personal purchasing prob. for k-product
        ### ppp_dict[k][i]: (float2) the personal purchasing prob. for i-node for k-product
        ppp_dict = [{} for _ in range(self.num_product)]

        for k in range(self.num_product):
            price = self.product_list[k][2]
            # -- if the node i can't purchase the product k, then ppp_dict[k][i] = 0 --
            for i in wallet_dict:
                if wallet_dict[i] >= price:
                    if self.ppp == 'random':
                        # -- after buying a product, the prob. to buy another product will decrease randomly --
                        ppp_dict[k][i] = round(random.uniform(0, 1), 4)
                    elif self.ppp == 'expensive':
                        # -- choose as expensive as possible --
                        ppp_dict[k][i] = round((price / wallet_dict[i]), 4)
                    elif self.ppp == 'cheap':
                        # -- choose as cheap as possible --
                        ppp_dict[k][i] = round(1 - (price / wallet_dict[i]), 4)
                else:
                    ppp_dict[k][i] = 0

        return ppp_dict

    def updatePersonalPurchasingProbDict(self, k_prod, i_node, wallet_dict, ppp_dict):
        # -- after i-node buy k-product, ppp_dict has to be updated --
        price = self.product_list[k_prod][2]
        if self.ppp == 'random':
            # -- after buying a product, the prob. to buy another product will decrease randomly --
            for k in range(self.num_product):
                if k == k_prod or wallet_dict[i_node] == 0.0:
                    ppp_dict[k][i_node] = 0
                else:
                    ppp_dict[k][i_node] = round(random.uniform(0, ppp_dict[k][i_node]), 4)
        elif self.ppp == 'expensive':
            # -- choose as expensive as possible --
            for k in range(self.num_product):
                if k == k_prod or wallet_dict[i_node] == 0.0:
                    ppp_dict[k][i_node] = 0
                else:
                    ppp_dict[k][i_node] *= round((price / wallet_dict[i_node]), 4)
        elif self.ppp == 'cheap':
            # -- choose as cheap as possible --
            for k in range(self.num_product):
                if k == k_prod or wallet_dict[i_node] == 0.0:
                    ppp_dict[k][i_node] = 0
                else:
                    ppp_dict[k][i_node] *= round(1 - (price / wallet_dict[i_node]), 4)

        return ppp_dict

    def getSeedSetProfit(self, s_set, wallet_dict, ppp_dict):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        customer_set = [set() for _ in range(self.num_product)]
        a_n_set = [s_set[k].copy() for k in range(self.num_product)]
        a_n_sequence, a_n_sequence2 = [(k, s, 1) for k in range(self.num_product) for s in s_set[k]], []
        pro_k_list = [0.0 for _ in range(self.num_product)]

        eva = Evaluation(self.graph_dict, self.product_list, self.ppp, self.wpiwp)
        while a_n_sequence:
            k_prod, i_node, i_acc_prob = a_n_sequence.pop(choice([i for i in range(len(a_n_sequence))]))
            benefit, price = self.product_list[k_prod][0], self.product_list[k_prod][2]

            # -- notice: prevent the node from owing no receiver --
            if i_node not in self.graph_dict:
                continue

            i_dict = self.graph_dict[i_node]
            for ii_node in i_dict:
                if random.random() > i_dict[ii_node]:
                    continue

                # -- notice: seed cannot use other product --
                if ii_node in s_total_set:
                    continue
                if ii_node in a_n_set[k_prod]:
                    continue
                if ppp_dict[k_prod][ii_node] == 0:
                    continue
                a_n_set[k_prod].add(ii_node)

                # -- purchasing --
                dp = bool(0)
                if random.random() <= ppp_dict[k_prod][i_node]:
                    customer_set[k_prod].add(i_node)
                    a_n_set[k_prod].add(i_node)
                    wallet_dict[ii_node] -= price
                    ppp_dict = eva.updatePersonalPurchasingProbDict(k_prod, ii_node, wallet_dict, ppp_dict)
                    pro_k_list[k_prod] += benefit
                    dp = bool(1)

                ### -- whether passing the information or not --
                if not self.wpiwp or dp:
                    ii_acc_prob = i_acc_prob * i_dict[ii_node]
                    a_n_sequence2.append((k_prod, ii_node, ii_acc_prob))

                if not a_n_sequence:
                    a_n_sequence, a_n_sequence2 = a_n_sequence2, a_n_sequence

        pro_k_list = [round(pro_k, 4) for pro_k in pro_k_list]
        pnn_k_list = [len(customer) for customer in customer_set]
        ep = round(sum(pro_k_list), 4)

        return ep, pro_k_list, pnn_k_list


class EvaluationM:
    def __init__(self, model_name, dataset_name, product_name, cascade_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.wpiwp = True
        self.eva_monte_carlo = 100

    def evaluate(self, bi, wallet_distribution_type, ppp, seed_set_sequence, ss_time_sequence):
        eva_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        iniW = IniWallet(self.dataset_name, self.product_name, wallet_distribution_type)

        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        wallet_dict = iniW.constructWalletDict()
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))
        total_budget = round(total_cost / 2 ** bi, 4)

        ppp_strategy = 'random' * (ppp == 1) + 'expensive' * (ppp == 2) + 'cheap' * (ppp == 3)
        result = []

        eva = Evaluation(graph_dict, product_list, ppp_strategy, self.wpiwp)
        personal_prob_dict = eva.setPersonalPurchasingProbDict(wallet_dict)
        for sample_count, sample_seed_set in enumerate(seed_set_sequence):
            print('@ ' + self.model_name + ' evaluation @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + wallet_distribution_type + ', ppp = ' + ppp_strategy + ', sample_count = ' + str(sample_count))
            sample_pro_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]

            for _ in range(self.eva_monte_carlo):
                pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(sample_seed_set, copy.deepcopy(wallet_dict), copy.deepcopy(personal_prob_dict))
                sample_pro_k_acc = [(pro_k + sample_pro_k) for pro_k, sample_pro_k in zip(pro_k_list, sample_pro_k_acc)]
                sample_pnn_k_acc = [(pnn_k + sample_pnn_k) for pnn_k, sample_pnn_k in zip(pnn_k_list, sample_pnn_k_acc)]
            sample_pro_k_acc = [round(sample_pro_k / self.eva_monte_carlo, 4) for sample_pro_k in sample_pro_k_acc]
            sample_pnn_k_acc = [round(sample_pnn_k / self.eva_monte_carlo, 4) for sample_pnn_k in sample_pnn_k_acc]
            sample_bud_k_acc = [round(sum(seed_cost_dict[sample_seed_set.index(sample_bud_k)][i] for i in sample_bud_k), 4) for sample_bud_k in sample_seed_set]
            sample_sn_k_acc = [len(sample_sn_k) for sample_sn_k in sample_seed_set]
            sample_pro_acc = round(sum(sample_pro_k_acc), 4)
            sample_bud_acc = round(sum(sample_bud_k_acc), 4)

            result.append([sample_pro_acc, sample_bud_acc, sample_sn_k_acc, sample_pnn_k_acc, sample_pro_k_acc, sample_bud_k_acc, sample_seed_set])

            print('eva_time = ' + str(round(time.time() - eva_start_time, 2)) + 'sec')
            print(result[sample_count])
            print('------------------------------------------')

        avg_pro = round(sum(r[0] for r in result) / len(seed_set_sequence), 4)
        avg_bud = round(sum(r[1] for r in result) / len(seed_set_sequence), 4)
        avg_sn_k = [round(sum(r[2][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]
        avg_pnn_k = [round(sum(r[3][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]
        avg_pro_k = [round(sum(r[4][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]
        avg_bud_k = [round(sum(r[5][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]

        path = 'result/' + \
               self.model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp' * self.wpiwp
        if not os.path.isdir(path):
            os.mkdir(path)
        fw = open(path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(bi) + '.txt', 'w')
        fw.write(self.model_name + ', ' + self.dataset_name + '_' + self.cascade_model + ', ' + self.product_name + '\n' +
                 'ppp = ' + str(ppp) + ', wd = ' + wallet_distribution_type + ', wpiwp = ' + str(self.wpiwp) + '\n' +
                 'total_budget = ' + str(total_budget) + ', sample_number = ' + str(len(seed_set_sequence)) + '\n' +
                 'avg_profit = ' + str(avg_pro) + ', avg_budget = ' + str(avg_bud) + '\n' +
                 'total_time = ' + str(round(sum(ss_time_sequence), 4)) + ', avg_time = ' + str(round(sum(ss_time_sequence) / len(ss_time_sequence), 4)) + '\n')
        fw.write('\nprofit_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_pro_k[kk]))
        fw.write('\nbudget_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_bud_k[kk]))
        fw.write('\nseed_number =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_sn_k[kk]))
        fw.write('\ncustomer_number =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_pnn_k[kk]))
        fw.write('\n')

        for t, r in enumerate(result):
            fw.write('\n' + str(t) + '\t' + str(round(r[0], 4)) + '\t' + str(round(r[1], 4)) + '\t' + str(r[2]) + '\t' + str(r[3]) + '\t' + str(r[4]) + '\t' + str(r[5]) + '\t' + str(r[6]))
        fw.close()