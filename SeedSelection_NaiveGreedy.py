from Diffusion import *
import heap


class SeedSelectionNG:
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
        celf_heap = [(0.0, -1, '-1', 0)]

        diff_ss = Diffusion(self.graph_dict, self.product_list)
        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = round(sum([diff_ss.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0], self.product_list[0][0])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]

        diff_ss = Diffusion(self.graph_dict, self.product_list)
        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = round(sum([diff_ss.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0], self.product_list[0][0])
                    mg_ratio = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap


class SeedSelectionNGPW:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, monte):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.monte = monte

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]

        diffpw_ss = DiffusionPW(self.graph_dict, self.product_list, self.product_weight_list)

        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diffpw_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0] * self.product_weight_list[k], self.product_list[0][0] * self.product_weight_list[0])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]

        diffpw_ss = DiffusionPW(self.graph_dict, self.product_list, self.product_weight_list)

        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diffpw_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0] * self.product_weight_list[k], self.product_list[0][0] * self.product_weight_list[0])
                    mg_ratio = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap