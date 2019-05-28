from Diffusion import *
import heap


class SeedSelectionNGAP:
    def __init__(self, graph_dict, seed_cost_dict, product_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict: (dict) the set of cost for seeds
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeExpectedInfDict({i}, i, 1)
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0], 4)
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeExpectedInfDict({i}, i, 1)
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0], 4)
                    mg_ratio = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap


class SeedSelectionNGAPPW:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict: (dict) the set of cost for seeds
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeExpectedInfDict({i}, i, 1)
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0] * self.product_weight_list[k], 4)
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeExpectedInfDict({i}, i, 1)
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0] * self.product_weight_list[k], 4)
                    mg_ratio = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap


class SeedSelectionNGAPEPW:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict: (dict) the set of cost for seeds
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]
        diffeap_ss = DiffusionEdgeAccProb(self.graph_dict, self.product_list, self.product_weight_list)

        for k in range(self.num_product):
            for i in self.graph_dict:
                i_dict = diffeap_ss.buildNodeExpectedInfDict({i}, k, i, 1)
                ei = getExpectedInf(i_dict)

                if ei > 0:
                    mg = round(ei * self.product_list[k][0], 4)
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]
        diffeap_ss = DiffusionEdgeAccProb(self.graph_dict, self.product_list, self.product_weight_list)

        for k in range(self.num_product):
            for i in self.graph_dict:
                i_dict = diffeap_ss.buildNodeExpectedInfDict({i}, k, i, 1)
                ei = getExpectedInf(i_dict)

                if ei > 0:
                    mg = round(ei * self.product_list[k][0], 4)
                    mg_ratio = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap