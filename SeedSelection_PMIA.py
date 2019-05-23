from Initialization import *
# from Diffusion import *
import heap

class SeedSelectionPMIA:
    def __init__(self, graph_dict, seed_cost_dict, product_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict: (dict) the set of cost for seeds
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.prob_threshold = 0.001

    def generateMIP(self):
        mioa_dict, miia_dict = {}, {}

        for source_node in self.graph_dict:
            mioa_dict[source_node] = {}
            source_dict = {source_node: (1.0, source_node)}
            source_heap = [(1.0, source_node)]
            for i in self.graph_dict[source_node]:
                source_dict[i] = (self.graph_dict[source_node][i], source_node)
                heap.heappush_max(source_heap, (self.graph_dict[source_node][i], i))

            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = source_dict[i_node][1]
                if i_node != i_prev:
                    i_path = [i_node, i_prev]
                    while i_prev != source_dict[i_prev][1]:
                        i_prev = source_dict[i_prev][1]
                        i_path.append(i_prev)
                    i_path.reverse()

                    mioa_dict[source_node][i_node] = (i_prob, i_path)
                    if i_node not in miia_dict:
                        miia_dict[i_node] = {source_node: (i_prob, i_path)}
                    else:
                        miia_dict[i_node][source_node] = (i_prob, i_path)

                if i_node in self.graph_dict and i_node != source_node:
                    for ii_node in self.graph_dict[i_node]:
                        if ii_node not in mioa_dict[source_node]:
                            ii_prob = round(i_prob * self.graph_dict[i_node][ii_node], 4)

                            if ii_prob >= self.prob_threshold:
                                if ii_node in source_dict:
                                    ii_prob_d = source_dict[ii_node][0]
                                    if ii_prob > ii_prob_d:
                                        source_dict[ii_node] = (ii_prob, i_node)
                                        source_heap.remove((ii_prob_d, ii_node))
                                        heap.heapify_max(source_heap)
                                        heap.heappush_max(source_heap, (ii_prob, ii_node))
                                else:
                                    source_dict[ii_node] = (ii_prob, i_node)
                                    heap.heappush_max(source_heap, (ii_prob, ii_node))

        return mioa_dict, miia_dict

    def calculateAP(self, i_node_u, s_set, miia_v):
        # -- calculate the activation probability which s_set activate i_node_u on miia_v --
        if i_node_u in s_set:
            return 1.0
        else:
            n_in_seq = []
            for i in miia_v:
                if i_node_u in miia_v[i][1] and i_node_u != miia_v[i][1][0]:
                    n_in_seq.append(miia_v[i][1][miia_v[i][1].index(i_node_u) - 1])
            if not n_in_seq:
                return 0.0
            else:
                sspmia_ss = SeedSelectionPMIA(self.graph_dict, self.seed_cost_dict, self.product_list)
                acc_prob = 1.0
                for nis in n_in_seq:
                    nis_w = nis
                    ap_w = sspmia_ss.calculateAP(nis_w, s_set, miia_v)
                    acc_prob *= (1 - ap_w * self.graph_dict[nis_w][i_node_u])
                return round(1 - acc_prob, 4)


if __name__ == '__main__':
    dataset_name = 'toy2'
    product_name = 'item_hplc'
    cascade_model = 'ic'

    ini = Initialization(dataset_name, product_name)
    seed_cost_dict = ini.constructSeedCostDict()
    graph_dict = ini.constructGraphDict(cascade_model)
    product_list = ini.constructProductList()

    sspmia = SeedSelectionPMIA(graph_dict, seed_cost_dict, product_list)
    for i in graph_dict:
        for j in graph_dict[i]:
            print(i, j, graph_dict[i][j])
    mioa_dict, miia_dict = sspmia.generateMIP()
    for item in mioa_dict:
        print(item, mioa_dict[item])
    for item in miia_dict:
        print(item, miia_dict[item])

    # for i in [i for i in seed_cost_dict[0] if i != '4']:
    #     if i in miia_dict:
    #         for j in miia_dict[i]:
    #             print(i, j, sspmia.calculateAP(j, {'4'}, miia_dict[i]))

    for i in [i for i in seed_cost_dict[0] if i != '4']:
        if i in miia_dict:
            print(i, sspmia.calculateAP(i, {'4'}, miia_dict[i]))
