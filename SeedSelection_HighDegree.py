import heap


class SeedSelectionHD:
    def __init__(self, data_name, graph_dict, product_list):
        ###  data_degree_path: (str) tha file path
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.data_degree_path = 'data/' + data_name + '/degree.txt'
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)

    def generateDegreeHeap(self):
        degree_heap = [(-1, -1, '-1')]

        with open(self.data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    degree_item = (int(deg), k, i)
                    heap.heappush_max(degree_heap, degree_item)
        f.close()

        return degree_heap

    def generateExpandDegreeHeap(self):
        degree_dict = {}
        with open(self.data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                degree_dict[i] = int(deg)
        f.close()

        degree_expand_heap = [(1, -1, '-1')]

        for i in self.graph_dict:
            deg = degree_dict[i]
            for ii in self.graph_dict[i]:
                deg += degree_dict[ii]
            for k in range(self.num_product):
                degree_item = (deg, k, i)
                heap.heappush_max(degree_expand_heap, degree_item)

        return degree_expand_heap


class SeedSelectionHDPW:
    def __init__(self, data_name, graph_dict, product_list, product_weight_list):
        ###  data_degree_path: (str) tha file path
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.data_degree_path = 'data/' + data_name + '/degree.txt'
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list

    def generateDegreeHeap(self):
        degree_heap = [(0, -1, '-1')]

        with open(self.data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    degree_item = (float(int(deg) * self.product_weight_list[k]), k, i)
                    heap.heappush_max(degree_heap, degree_item)
        f.close()

        return degree_heap

    def generateExpandDegreeHeap(self):
        degree_dict = {}
        with open(self.data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                degree_dict[i] = int(deg)
        f.close()

        degree_expand_heap = [(0, -1, '-1')]

        for i in self.graph_dict:
            deg = degree_dict[i]
            for ii in self.graph_dict[i]:
                deg += degree_dict[ii]
            for k in range(self.num_product):
                degree_item = (deg * self.product_weight_list[k], k, i)
                heap.heappush_max(degree_expand_heap, degree_item)

        return degree_expand_heap