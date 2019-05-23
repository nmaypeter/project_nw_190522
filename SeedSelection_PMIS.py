from Diffusion import *
import heap
import operator
import copy


class SeedSelectionPMIS:
    def __init__(self, graph_dict, seed_cost_dict, product_list, monte):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.monte = monte

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [[(0.0, -1, '-1', 0)] for _ in range(self.num_product)]

        diff_ss = Diffusion(self.graph_dict, self.product_list)
        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = round(sum([diff_ss.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0], self.product_list[0][0])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap[k], celf_item)

        return celf_heap

    def solveMultipleChoiceKnapsackProblem(self, bud, s_matrix, c_matrix):
        mep_result = (0.0, [set() for _ in range(self.num_product)])
        ### bud_index: (list) the using budget index for products
        ### bud_bound_index: (list) the bound budget index for products
        bud_index, bud_bound_index = [len(kk) - 1 for kk in c_matrix], [0 for _ in range(self.num_product)]
        ### temp_bound_index: (list) the bound to exclude the impossible budget combination s.t. the k-budget is smaller than the temp bound
        temp_bound_index = [0 for _ in range(self.num_product)]

        diff_ss = Diffusion(self.graph_dict, self.product_list)
        while not operator.eq(bud_index, bud_bound_index):
            ### bud_pmis: (float) the budget in this pmis execution
            bud_pmis = 0.0
            for k in range(self.num_product):
                bud_pmis += copy.deepcopy(c_matrix)[k][bud_index[k]]

            if bud_pmis <= bud:
                temp_bound_flag = 1
                for k in range(self.num_product):
                    if temp_bound_index[k] > bud_index[k]:
                        temp_bound_flag = 0
                        break
                if temp_bound_flag:
                    temp_bound_index = copy.deepcopy(bud_index)

                    s_set = [set() for _ in range(self.num_product)]
                    for k in range(self.num_product):
                        s_set[k] = copy.deepcopy(s_matrix)[k][bud_index[k]][k]

                    pro_acc = 0.0
                    for _ in range(self.monte):
                        pro_acc += diff_ss.getSeedSetProfit(s_set)
                    pro_acc = round(pro_acc / self.monte, 4)

                    if pro_acc > mep_result[0]:
                        mep_result = (pro_acc, s_set)

            pointer = self.num_product - 1
            while bud_index[pointer] == bud_bound_index[pointer]:
                bud_index[pointer] = len(c_matrix[pointer]) - 1
                pointer -= 1
            bud_index[pointer] -= 1

        return mep_result