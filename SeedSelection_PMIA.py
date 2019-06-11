import heap


def findNIn_set(i_node, miia):
    n_in_set = set()
    for i in miia:
        if i_node in miia[i][1] and i_node != miia[i][1][0]:
            # -- i_node_w is i_node_u's in-neighbor --
            i_node_w = miia[i][1][miia[i][1].index(i_node) - 1]
            n_in_set.add(i_node_w)

    return n_in_set


def induceGraph(graph, s_node):
    if s_node in graph:
        del graph[s_node]
    for i in graph:
        if s_node in graph[i]:
            del graph[i][s_node]

    del_list = [i for i in graph if not graph[i]]
    while del_list:
        del_node = del_list.pop()
        del graph[del_node]


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
        self.prob_threshold = 0.01

    def generateMIA(self, s_set):
        ### mioa_dict[i_node_u][i_node_v] = (prob, path): MIP from i_node_u to i_node_v
        ### miia_dict[i_node_v][i_node_u] = (prob, path): MIP from i_node_u to i_node_v
        mioa_dict, miia_dict = {}, {}
        sub_graph = self.graph_dict.copy()
        for s_node in s_set:
            induceGraph(sub_graph, s_node)

        for source_node in sub_graph:
            ### source_dict[i_node] = (prob, in-neighbor)
            mioa_dict[source_node] = {}
            source_dict = {source_node: (1.0, source_node)}
            source_heap = []
            for i in sub_graph[source_node]:
                source_dict[i] = (sub_graph[source_node][i], source_node)
                heap.heappush_max(source_heap, (sub_graph[source_node][i], i))

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = source_dict[i_node][1]

                # -- find MIP from source_node to i_node --
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

                if i_node in sub_graph:
                    for ii_node in sub_graph[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_dict[source_node]:
                            ii_prob = round(i_prob * sub_graph[i_node][ii_node], 4)

                            if ii_prob >= self.prob_threshold:
                                # -- if ii_node is in heap --
                                if ii_node in source_dict:
                                    ii_prob_d = source_dict[ii_node][0]
                                    if ii_prob > ii_prob_d:
                                        source_dict[ii_node] = (ii_prob, i_node)
                                        source_heap.remove((ii_prob_d, ii_node))
                                        source_heap.append((ii_prob, ii_node))
                                        heap.heapify_max(source_heap)
                                # -- if ii_node is not in heap --
                                else:
                                    source_dict[ii_node] = (ii_prob, i_node)
                                    heap.heappush_max(source_heap, (ii_prob, ii_node))

        return mioa_dict, miia_dict

    def calculateAP(self, i_node_u, s_set, miia_v):
        # -- calculate the activation probability which s_set activate i_node_u on miia_v --
        if i_node_u in s_set:
            return 1.0
        else:
            n_in_set_u = findNIn_set(i_node_u, miia_v)
            if n_in_set_u:
                sspmia_ss = SeedSelectionPMIA(self.graph_dict, self.seed_cost_dict, self.product_list)
                acc_prob = 1.0
                for nis in n_in_set_u:
                    nis_w = nis
                    ap_w = sspmia_ss.calculateAP(nis_w, s_set, miia_v)
                    acc_prob *= (1 - ap_w * self.graph_dict[nis_w][i_node_u])
                return round(1 - acc_prob, 4)
            else:
                return 0.0

    def calculateAlpha(self, i_node_v, i_node_u, s_set, miia_v, ap_dict_v):
        if i_node_v == i_node_u:
            return 1.0
        else:
            # -- i_node_w is i_node_u's out-neighbor --
            i_node_w = miia_v[i_node_u][1][miia_v[i_node_u][1].index(i_node_u) + 1]
            if i_node_w in s_set:
                return 0.0
            else:
                n_in_set_w = findNIn_set(i_node_w, miia_v)
                while i_node_u in n_in_set_w:
                    n_in_set_w.remove(i_node_u)
                if n_in_set_w:
                    sspmia_ss = SeedSelectionPMIA(self.graph_dict, self.seed_cost_dict, self.product_list)
                    alpha_v_w = sspmia_ss.calculateAlpha(i_node_v, i_node_w, s_set, miia_v, ap_dict_v)
                    pp_u_w = self.graph_dict[i_node_u][i_node_w]
                    acc_prob = 1.0
                    for nis in n_in_set_w:
                        nis_u_prime = nis
                        ap_u_prime = ap_dict_v[nis_u_prime]
                        acc_prob *= (1 - ap_u_prime * self.graph_dict[nis_u_prime][i_node_w])
                    return round(alpha_v_w * pp_u_w * acc_prob, 4)
                else:
                    return 0.0

    def updatePMIIA(self, s_set, pmiia_dict):
        s_list = s_set.copy()

        s_node = ''
        sub_graph = self.graph_dict.copy()
        while s_list:
            if len(s_list) != len(s_set):
                induceGraph(sub_graph, s_node)
            s_node = s_list.pop(0)

            pmiia_dict[s_node] = {}
            mioa_dict, miia_dict = {}, {s_node: {}}
            source_in_dict = {s_node: (1.0, s_node)}
            source_in_heap = []
            s_node_in_neighbor = [i for i in sub_graph if s_node in sub_graph[i]]
            for i in s_node_in_neighbor:
                source_in_dict[i] = (sub_graph[i][s_node], s_node)
                heap.heappush_max(source_in_heap, (sub_graph[i][s_node], i))

            while source_in_heap:
                (i_prob, i_node) = heap.heappop_max(source_in_heap)
                i_subs = source_in_dict[i_node][1]

                # -- find MIP from source_node to i_node --
                i_path = [i_node, i_subs]
                while i_subs != source_in_dict[i_subs][1]:
                    i_subs = source_in_dict[i_subs][1]
                    i_path.append(i_subs)

                if i_node not in mioa_dict:
                    mioa_dict[i_node] = {s_node: (i_prob, i_path)}
                else:
                    mioa_dict[i_node][s_node] = (i_prob, i_path)
                miia_dict[s_node][i_node] = (i_prob, i_path)

                i_node_in_neighbor = [i for i in sub_graph if i_node in sub_graph[i]]
                if i_node_in_neighbor:
                    for ii_node in i_node_in_neighbor:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_dict[i_node]:
                            ii_prob = round(i_prob * sub_graph[ii_node][i_node], 4)

                            if ii_prob >= self.prob_threshold:
                                # -- if ii_node is in heap --
                                if ii_node in source_in_dict:
                                    ii_prob_d = source_in_dict[ii_node][0]
                                    if ii_prob > ii_prob_d:
                                        source_in_dict[ii_node] = (ii_prob, i_node)
                                        source_in_heap.remove((ii_prob_d, ii_node))
                                        source_in_heap.append((ii_prob, ii_node))
                                        heap.heapify_max(source_in_heap)
                                # -- if ii_node is not in heap --
                                else:
                                    source_in_dict[ii_node] = (ii_prob, i_node)
                                    heap.heappush_max(source_in_heap, (ii_prob, ii_node))

            for i_node_v in miia_dict[s_node]:
                is_set = {i for i in miia_dict[s_node][i_node_v][1] if i in s_list}
                if not is_set:
                    pmiia_dict[s_node][i_node_v] = miia_dict[s_node][i_node_v]

        del_list = [i for i in pmiia_dict if not pmiia_dict[i]]
        while del_list:
            del_node = del_list.pop()
            del pmiia_dict[del_node]

        return pmiia_dict