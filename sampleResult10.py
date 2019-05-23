from Evaluation import *
from openpyxl import *
import numpy as np


def output(cascade_model, dataset_name, product_name, wallet_distribution_type, ppp_strategy, model_name, eva_time, result10_pro_list):
    path = [cascade_model, dataset_name, product_name, wallet_distribution_type, ppp_strategy, model_name + '\t']
    result10_mean = np.mean(result10_pro_list)
    result10_std = np.std(result10_pro_list)
    result10_max = max(result10_pro_list)
    result10_min = min(result10_pro_list)
    result10_list = [eva_time, round(float(result10_mean), 4), round(float(result10_std), 4),
                     round(float(result10_mean) - 3 * float(result10_std), 4), round(float(result10_mean) + 3 * float(result10_std), 4),
                     result10_max, result10_min, '\t']
    result10_pro_list.sort()
    result10 = path + result10_list + result10_pro_list

    wb = load_workbook('sample.xlsx')
    sheet = wb.active
    sheet.append(result10)
    wb.save('sample.xlsx')


if __name__ == '__main__':
    cm_seq = [1, 2]
    dataset_seq = [1, 2, 3, 4]
    prod_seq = [1, 2]
    wallet_distribution_seq = [1, 2]
    ppp_seq = [2, 3]
    model_seq = ['mng', 'mngpw', 'mngr', 'mngrpw', 'mngsr', 'mngsrpw',
                 'mngap', 'mngappw', 'mngapr', 'mngaprpw', 'mngapsr', 'mngapsrpw',
                 'mhd', 'mhdpw', 'mhed', 'mhedpw',
                 'mpmis', 'mr']

    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for data_setting in dataset_seq:
            dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                           'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                for wallet_distribution in wallet_distribution_seq:
                    wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (
                                wallet_distribution == 2)
                    for ppp in [2, 3]:
                        for model_name in model_seq:
                            ppp_strategy = 'random' * (ppp == 1) + 'expensive' * (ppp == 2) + 'cheap' * (ppp == 3)
                            try:
                                path = 'result/' + model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp'
                                seed_set = []
                                with open(path + '/' + dataset_name + '_' + cascade_model + '_' + product_name + '_bi10.txt') as f:
                                    for lnum, line in enumerate(f):
                                        if lnum <= 10:
                                            continue
                                        elif lnum == 11:
                                            (l) = line.split('\t')
                                            seed_set = eval(l[7])
                                        else:
                                            break
                                f.close()

                                eva_start_time = time.time()
                                ini = Initialization(dataset_name, product_name)
                                iniW = IniWallet(dataset_name, product_name, wallet_distribution_type)

                                seed_cost_dict = ini.constructSeedCostDict()
                                graph_dict = ini.constructGraphDict(cascade_model)
                                product_list = ini.constructProductList()
                                num_product = len(product_list)
                                wallet_dict = iniW.constructWalletDict()

                                eva = Evaluation(graph_dict, product_list, ppp_strategy, True)
                                personal_prob_dict = eva.setPersonalPurchasingProbDict(wallet_dict)

                                print('@ ' + model_name + ' evaluation @ dataset_name = ' + dataset_name + '_' + cascade_model + ', product_name = ' + product_name +
                                      ', wd = ' + wallet_distribution_type + ', ppp = ' + ppp_strategy)

                                result10_pro_list = []
                                for _ in range(100):
                                    pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, copy.deepcopy(wallet_dict), copy.deepcopy(personal_prob_dict))
                                    result10_pro_list.append(pro)
                                eva_time = round(time.time() - eva_start_time, 4)
                                output(cascade_model, dataset_name, product_name, wallet_distribution_type, ppp_strategy, model_name, eva_time, result10_pro_list)
                            except FileNotFoundError:
                                output(cascade_model, dataset_name, product_name, wallet_distribution_type, ppp_strategy, model_name, 0.0, [])