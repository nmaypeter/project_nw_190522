from SeedSelection_NaiveGreedy import *
from SeedSelection_NGAP import *
from SeedSelection_HighDegree import *
from SeedSelection_PMIS import *
from SeedSelection_Random import *
from Evaluation import *
import os
import time


def saveTempSequence(file_name, temp_sequence):
    path = 'temp_log/' + file_name + '.txt'
    fw = open(path, 'w')
    for t in temp_sequence:
        fw.write(str(t) + '\n')
    fw.close()


def getTempSequence(file_name):
    path = 'temp_log/' + file_name + '.txt'
    temp_sequence = []
    with open(path) as f:
        for t_id, t in enumerate(f):
            temp_sequence.append(eval(t))
    f.close()
    os.remove(path)

    return temp_sequence


class Model:
    def __init__(self, model_name, dataset_name, product_name, cascade_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.budget_iteration = [i for i in range(10, 5, -1)]
        self.wd_seq = ['m50e25', 'm99e96']
        self.wpiwp = True
        self.sample_number = 1
        self.ppp_seq = [2, 3]
        self.monte_carlo = 10
        self.batch = 20

    def model_ng(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = round(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        mg_t = round(ep_t - now_profit, 4)
                        flag_t = seed_set_length

                        if mg_t > 0:
                            celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = round(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        mg_t = round(ep_t - now_profit, 4)
                        mg_ratio_t = safe_div(mg_t, sc)
                        flag_t = seed_set_length

                        if mg_ratio_t > 0:
                            celf_item_t = (mg_ratio_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngsr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = round(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        mg_t = round(ep_t - now_profit, 4)
                        mg_seed_ratio_t = safe_div(mg_t, now_budget + sc)
                        flag_t = seed_set_length

                        if mg_seed_ratio_t > 0:
                            celf_item_t = (mg_seed_ratio_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngap(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_mg, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_t = mep_item_sequence[midl][1]
                            i_node_t = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_t = round(expected_inf * product_list[k_prod_t][0], 4)
                            mg_t = round(ep_t - expected_profit_k[k_prod_t], 4)
                            flag_t = seed_set_length

                            if mg_t > 0:
                                celf_item_t = (mg_t, k_prod_t, i_node_t, flag_t)
                                heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngapr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_ratio * sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_ratio * sc, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_t = mep_item_sequence[midl][1]
                            i_node_t = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_t = round(expected_inf * product_list[k_prod_t][0], 4)
                            mg_t = round(ep_t - expected_profit_k[k_prod_t], 4)
                            mg_ratio_t = safe_div(mg_t, seed_cost_dict[k_prod_t][i_node_t])
                            flag_t = seed_set_length

                            if mg_ratio_t > 0:
                                celf_item_t = (mg_ratio_t, k_prod_t, i_node_t, flag_t)
                                heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngapsr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_seed_ratio * now_budget, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_seed_ratio * now_budget, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_t = mep_item_sequence[midl][1]
                            i_node_t = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_t = round(expected_inf * product_list[k_prod_t][0], 4)
                            mg_t = round(ep_t - expected_profit_k[k_prod_t], 4)
                            mg_seed_ratio_t = safe_div(mg_t, now_budget + seed_cost_dict[k_prod_t][i_node_t])
                            flag_t = seed_set_length

                            if mg_seed_ratio_t > 0:
                                celf_item_t = (mg_seed_ratio_t, k_prod_t, i_node_t, flag_t)
                                heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_hd(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshd_model = SeedSelectionHD(self.dataset_name, graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshd_model.generateDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, seed_set, degree_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, seed_set, degree_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(degree_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, seed_set, degree_heap])
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_hed(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshd_model = SeedSelectionHD(self.dataset_name, graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshd_model.generateExpandDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, seed_set, degree_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, seed_set, degree_heap] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(degree_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, seed_set, degree_heap])
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_pmis(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[0.0 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sspmis_model = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            celf_heap = sspmis_model.generateCelfHeap()
            s_matrix_sequence, c_matrix_sequence = [[] for _ in range(num_product)], [[] for _ in range(num_product)]
            for k in range(num_product):
                bud_iter = self.budget_iteration.copy()
                b_iter = bud_iter.pop(0)
                now_budget, now_profit = 0.0, 0.0
                seed_set = [set() for _ in range(num_product)]
                s_matrix, c_matrix = [[set() for _ in range(num_product)]], [0.0]
                ss_acc_time = round(time.time() - ss_start_time, 4)
                temp_sequence_flag = True
                saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                 [ss_acc_time, now_budget, now_profit, seed_set, s_matrix, c_matrix, celf_heap[k]])
                while temp_sequence_flag:
                    ss_start_time = time.time()
                    temp_sequence_flag = False
                    bi_index = self.budget_iteration.index(b_iter)
                    total_budget = round(total_cost / (2 ** b_iter), 4)
                    [ss_acc_time, now_budget, now_profit, seed_set, s_matrix, c_matrix, celf_heap_k] = \
                        getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                    print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                          ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                    mep_item = heap.heappop_max(celf_heap_k)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    while now_budget < total_budget and mep_i_node != '-1':
                        sc = seed_cost_dict[mep_k_prod][mep_i_node]
                        seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                        if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                            temp_sequence_flag = True
                            b_iter = bud_iter.pop(0)
                            heap.heappush_max(celf_heap_k, mep_item)
                            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                             [ss_time, now_budget, now_profit, seed_set, s_matrix, c_matrix, celf_heap_k])
                            mep_item = heap.heappop_max(celf_heap_k)
                            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                        if round(now_budget + sc, 4) > total_budget:
                            mep_item = heap.heappop_max(celf_heap_k)
                            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                            continue

                        if mep_flag == seed_set_length:
                            seed_set[mep_k_prod].add(mep_i_node)
                            now_budget = round(now_budget + sc, 4)
                            now_profit = round(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                            s_matrix.append(copy.deepcopy(seed_set))
                            c_matrix.append(now_budget)
                        else:
                            seed_set_t = copy.deepcopy(seed_set)
                            seed_set_t[mep_k_prod].add(mep_i_node)
                            ep_t = round(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                            mg_t = round(ep_t - now_profit, 4)
                            flag_t = seed_set_length

                            if mg_t > 0:
                                celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                                heap.heappush_max(celf_heap_k, celf_item_t)

                        mep_item = heap.heappop_max(celf_heap_k)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    s_matrix_sequence[k].append(s_matrix)
                    c_matrix_sequence[k].append(c_matrix)
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                    ss_time_sequence[bi_index][sample_count] += ss_time

            ss_start_time = time.time()
            for k in range(num_product):
                while len(s_matrix_sequence[k]) < len(self.budget_iteration):
                    s_matrix_sequence[k].append(s_matrix_sequence[k][-1])
                    c_matrix_sequence[k].append(c_matrix_sequence[k][-1])
            for bi in self.budget_iteration:
                bi_index = self.budget_iteration.index(bi)
                total_budget = round(total_cost / (2 ** bi), 4)
                s_matrix_bi, c_matrix_bi = [], []
                for k in range(num_product):
                    s_matrix_bi.append(s_matrix_sequence[k][bi_index])
                    c_matrix_bi.append(c_matrix_sequence[k][bi_index])
                mep_result = sspmis_model.solveMultipleChoiceKnapsackProblem(total_budget, s_matrix_bi, c_matrix_bi)
                ss_time = round(time.time() - ss_start_time, 4)
                ss_time_sequence[bi_index][sample_count] += ss_time
                seed_set = mep_result[1]
                seed_set_sequence[bi_index].append(seed_set)

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_r(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssr_model = SeedSelectionRandom(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            random_node_set = ssr_model.generateRandomNodeSet()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, seed_set, random_node_set])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, seed_set, random_node_set] = \
                    getTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = ssr_model.selectRandomSeed(random_node_set)
                mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        random_node_set.add(mep_item)
                        saveTempSequence(self.model_name + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, seed_set, random_node_set])
                        random_node_set.remove(mep_item)
                        mep_k_prod, mep_i_node = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = ssr_model.selectRandomSeed(random_node_set)
                        mep_k_prod, mep_i_node = mep_item
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = ssr_model.selectRandomSeed(random_node_set)
                    mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])


class ModelPW:
    def __init__(self, model_name, dataset_name, product_name, cascade_model, wallet_distribution_type):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.budget_iteration = [i for i in range(10, 5, -1)]
        self.wallet_distribution_type = wallet_distribution_type
        self.wpiwp = True
        self.sample_number = 1
        self.ppp_seq = [2, 3]
        self.monte_carlo = 10
        self.batch = 20

    def model_ngpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(sum([diffpw_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = round(sum([diffpw_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        mg_t = round(ep_t - now_profit, 4)
                        flag_t = seed_set_length

                        if mg_t > 0:
                            celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngrpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(sum([diffpw_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = round(sum([diffpw_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        mg_t = round(ep_t - now_profit, 4)
                        mg_ratio_t = safe_div(mg_t, sc)
                        flag_t = seed_set_length

                        if mg_ratio_t > 0:
                            celf_item_t = (mg_ratio_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngsrpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(sum([diffpw_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = round(sum([diffpw_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        mg_t = round(ep_t - now_profit, 4)
                        mg_seed_ratio_t = safe_div(mg_t, now_budget + sc)
                        flag_t = seed_set_length

                        if mg_seed_ratio_t > 0:
                            celf_item_t = (mg_seed_ratio_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngappw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_mg, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_t = mep_item_sequence[midl][1]
                            i_node_t = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_t = round(expected_inf * product_list[k_prod_t][0] * product_weight_list[k_prod_t], 4)
                            mg_t = round(ep_t - expected_profit_k[k_prod_t], 4)
                            flag_t = seed_set_length

                            if mg_t > 0:
                                celf_item_t = (mg_t, k_prod_t, i_node_t, flag_t)
                                heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngaprpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_ratio * sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_ratio * sc, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_t = mep_item_sequence[midl][1]
                            i_node_t = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_t = round(expected_inf * product_list[k_prod_t][0] * product_weight_list[k_prod_t], 4)
                            mg_t = round(ep_t - expected_profit_k[k_prod_t], 4)
                            mg_ratio_t = safe_div(mg_t, seed_cost_dict[k_prod_t][i_node_t])
                            flag_t = seed_set_length

                            if mg_ratio_t > 0:
                                celf_item_t = (mg_ratio_t, k_prod_t, i_node_t, flag_t)
                                heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ngapsrpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(celf_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap])
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_seed_ratio * now_budget, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_seed_ratio * now_budget, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_t = mep_item_sequence[midl][1]
                            i_node_t = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_t = round(expected_inf * product_list[k_prod_t][0] * product_weight_list[k_prod_t], 4)
                            mg_t = round(ep_t - expected_profit_k[k_prod_t], 4)
                            mg_seed_ratio_t = safe_div(mg_t, now_budget + seed_cost_dict[k_prod_t][i_node_t])
                            flag_t = seed_set_length

                            if mg_seed_ratio_t > 0:
                                celf_item_t = (mg_seed_ratio_t, k_prod_t, i_node_t, flag_t)
                                heap.heappush_max(celf_heap, celf_item_t)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_hdpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshdpw_model = SeedSelectionHDPW(self.dataset_name, graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshdpw_model.generateDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, seed_set, degree_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, seed_set, degree_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(degree_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, seed_set, degree_heap])
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_hedpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshdpw_model = SeedSelectionHDPW(self.dataset_name, graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshdpw_model.generateExpandDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence_flag = True
            saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                             self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                             [ss_acc_time, now_budget, seed_set, degree_heap])
            while temp_sequence_flag:
                ss_start_time = time.time()
                temp_sequence_flag = False
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, seed_set, degree_heap] = \
                    getTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                    self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter))
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence_flag:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        temp_sequence_flag = True
                        b_iter = bud_iter.pop(0)
                        heap.heappush_max(degree_heap, mep_item)
                        saveTempSequence(self.model_name + '_' + self.wallet_distribution_type + '_' + self.dataset_name + '_' +
                                         self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter),
                                         [ss_time, now_budget, seed_set, degree_heap])
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index])