from random import choice
import numpy as np
from scipy import stats


def getQuantiles(pd, mu, sigma):
    discrimination = -2 * sigma**2 * np.log(pd * sigma * np.sqrt(2 * np.pi))

    # no real roots
    if discrimination < 0:
        return None

    # one root, where x == mu
    elif discrimination == 0:
        return mu

    # two roots
    else:
        return choice([mu - np.sqrt(discrimination), mu + np.sqrt(discrimination)])


class Initialization:
    def __init__(self, data_name, prod_name):
        ### data_data_path, data_ic_weight_path, data_wc_weight_path, data_degree_path, product_path: (str) tha file path
        self.data_data_path = 'data/' + data_name + '/data.txt'
        self.data_ic_weight_path = 'data/' + data_name + '/weight_ic.txt'
        self.data_wc_weight_path = 'data/' + data_name + '/weight_wc.txt'
        self.data_degree_path = 'data/' + data_name + '/degree.txt'
        self.product_path = 'item/' + prod_name + '.txt'

    def setEdgeWeight(self):
        # -- browse dataset --
        with open(self.data_data_path) as f:
            num_node = 0
            out_degree_list, in_degree_list = [], []
            for line in f:
                (node1, node2) = line.split()
                num_node = max(num_node, int(node1), int(node2))
                out_degree_list.append(node1)
                in_degree_list.append(node2)
        f.close()

        # -- count degree for each node --
        fw = open(self.data_degree_path, 'w')
        for i in range(0, num_node + 1):
            fw.write(str(i) + '\t' + str(out_degree_list.count(str(i))) + '\n')
        fw.close()

        # -- set weight on edge for ic model and wc model --
        fw_ic = open(self.data_ic_weight_path, 'w')
        fw_wc = open(self.data_wc_weight_path, 'w')
        with open(self.data_data_path) as f:
            for line in f:
                (node1, node2) = line.split()
                fw_ic.write(node1 + '\t' + node2 + '\t0.1\n')
                fw_wc.write(node1 + '\t' + node2 + '\t' + str(round(1 / in_degree_list.count(node2), 2)) + '\n')
        fw_ic.close()
        fw_wc.close()

    def getNodeOutDegree(self, i_node):
        #  -- get the out-degree --
        deg = -1
        with open(self.data_degree_path) as f:
            for line in f:
                (node, degree) = line.split()
                if node == i_node:
                    deg = int(degree)
                    break
        f.close()

        return deg

    def getTotalNumNode(self):
        # -- get the num_node --
        num_node = 0
        with open(self.data_data_path) as f:
            for line in f:
                (node1, node2) = line.split()
                num_node = max(int(node1), int(node2), num_node)
        f.close()
        print('num_node = ' + str(round(num_node + 1, 2)))

        return num_node

    def getTotalNumEdge(self):
        # -- get the num_edge --
        num_edge = 0
        with open(self.data_data_path) as f:
            for _ in f:
                num_edge += 1
        f.close()
        print('num_edge = ' + str(round(num_edge, 2)))

        return num_edge

    def getMaxDegree(self):
        # -- get the max_deg --
        max_deg = 0
        with open(self.data_degree_path) as f:
            for line in f:
                (node, degree) = line.split()
                max_deg = max(max_deg, int(degree))
        f.close()
        print('max_deg = ' + str(round(max_deg, 2)))

        return max_deg

    def getProductList(self):
        # -- get product list --
        ### prod_list: (list) [profit, cost, price]
        prod_list = []
        with open(self.product_path) as f:
            for line in f:
                (b, c, r, p) = line.split()
                prod_list.append([float(b), float(c), round(float(b) + float(c), 2)])

        return prod_list

    def getTotalPrice(self):
        # -- get total_price from file
        total_price = 0.0
        with open(self.product_path) as f:
            for line in f:
                (p, c, r, pr) = line.split()
                total_price += float(pr)
        print('total_price = ' + str(round(total_price, 2)))

        return round(total_price, 2)


class IniWallet:
    def __init__(self, data_name, prod_name, wallet_dist_type):
        ### dataset_name: (str)
        self.data_name = data_name
        self.prod_name = prod_name
        self.wallet_dist_type = wallet_dist_type
        self.wallet_dist_name = 'wallet_' + prod_name.split('_')[1] + '_' + wallet_dist_type

    def setNodeWallet(self):
        # -- set wallet for each node for each item --
        ini = Initialization(self.data_name, self.prod_name)
        price_list = [prod[2] for prod in ini.getProductList()]
        num_node = ini.getTotalNumNode()

        mu, sigma = 0, 1
        if self.wallet_dist_type == 'm50e25':
            mu = np.mean(price_list)
            sigma = (max(price_list) - mu) / 0.6745
        elif self.wallet_dist_type == 'm99e96':
            mu = sum(price_list)
            sigma = abs(min(price_list) - mu) / 3

        fw = open('data/' + self.data_name + '/' + self.wallet_dist_name + '.txt', 'w')
        for i in range(0, num_node + 1):
            wal = 0
            while wal <= 0:
                q = stats.norm.rvs(mu, sigma)
                pd = stats.norm.pdf(q, mu, sigma)
                wal = getQuantiles(pd, mu, sigma)
            fw.write(str(i) + '\t' + str(round(wal, 2)) + '\n')
        fw.close()

    def getTotalWallet(self):
        # -- get total_wallet from file --
        total_w = 0.0
        with open('data/' + self.data_name + '/' + self.wallet_dist_name + '.txt') as f:
            for line in f:
                (node, wallet) = line.split()
                total_w += float(wallet)
        f.close()
        print('total wallet = ' + self.wallet_dist_name + ' = ' + str(round(total_w, 2)))

        return total_w