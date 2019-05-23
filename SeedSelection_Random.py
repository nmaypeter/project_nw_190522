from random import choice


class SeedSelectionRandom:
    def __init__(self, graph_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)

    def generateRandomNodeSet(self):
        rn_set = set((k, i) for i in self.graph_dict for k in range(self.num_product))

        return rn_set

    @staticmethod
    def selectRandomSeed(rn_set):
        # -- select a seed for a random product randomly --
        mep = (-1, '-1')
        if len(rn_set) != 0:
            mep = choice(list(rn_set))
            rn_set.remove(mep)

        return mep