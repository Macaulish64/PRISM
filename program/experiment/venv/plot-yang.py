import numpy as np
import time
import math
import os
import dataset_setting
import sys
import datetime
import grid_generate as GridGen

import random
import copy

# from cvxopt import solvers, matrix, spdiag, log
import frequency_oracle as FreOra

import itertools



import algorithm_UniformGrid_HMM
import algorithm_PrivGR_PrivSL
import algorithm_UniformGrid_PrivSL
import algorithm_PrivGR_NaivePrivSL
import algorithm_UniformGrid_PrivSL_vary_granularity
import algorithm_PrivGR_PrivSL
import algorithm_PrivGR_PrivSL_vary_sigma
import algorithm_PrivGR_PrivSL_vary_alpha

import algorithm_PrivGR_Ngram3


import choose_granularity

import pickle


from parameter_setting import args

import multiprocessing as mp

import matplotlib.pyplot as plt

import parameter_setting as para

import seaborn as sns


import utility_metric_query_avre as UMquery_avre
import utility_metric_query_avae as UMquery_avae
import utility_metric_query_FP as UM_FP
import utility_metric_query_length_error as UMlength_error





#***************************************new parallel plot below*********************************************




def parallel_new_plot_result_vary_all_algorithm_vary_attribute_num(tmp_plot_args_key_in_each_fig_dict:dict = None, group_attribute_num = None, task_id = None, metric = 'MNAE', args = None):
    print("parallel_new_plot_result_vary_all_algorithm_vary_attribute_num task %d begin!!" % task_id)

    show_algorithm_name_list = args.show_algorithm_name_list

    x_axis_args_key = 'attribute_num'

    if args.query_dimension_query_volume_flag == 0:
        result_mean_and_std_args_key_list = ['algorithm_name', 'dataset_name', 'user_num', 'attribute_num',
                                             'domain_size',
                                             'fre_oracle_type', 'granularity', 'epsilon', 'group_attribute_num',
                                             'private_flag', 'query_num', 'query_dimension', 'total_query_volume']
    else:
        result_mean_and_std_args_key_list = ['algorithm_name', 'dataset_name', 'user_num', 'attribute_num',
                                             'domain_size',
                                             'fre_oracle_type', 'granularity', 'epsilon', 'group_attribute_num',
                                             'private_flag', 'query_num', 'query_dimension', 'dimension_query_volume']

    # set the args
    for tmp_key in tmp_plot_args_key_in_each_fig_dict:
        setattr(args, tmp_key, tmp_plot_args_key_in_each_fig_dict[tmp_key])
    dataset = dataset_setting.Dataset(args.user_num, args.attribute_num, args.domain_size, args= args)

    def get_y_yerr(tmp_algorithm_name, x_axis_args_key):
        tmp_y =  []
        tmp_yerr = []
        tmp_result_mean_and_std_args_dict = dict()

        for tmp_key in result_mean_and_std_args_key_list: # initialize the dict
            tmp_result_mean_and_std_args_dict[tmp_key] = None

        tmp_result_mean_and_std_args_dict['algorithm_name'] = tmp_algorithm_name
        if tmp_algorithm_name == "Uni":
            tmp_result_mean_and_std_args_dict['granularity'] = args.domain_size
        elif tmp_algorithm_name == "1_way":
            tmp_result_mean_and_std_args_dict['granularity'] = args.domain_size
            tmp_result_mean_and_std_args_dict['group_attribute_num'] = 1
        elif tmp_algorithm_name in ["SW", "SW_float"]:
            tmp_result_mean_and_std_args_dict['granularity'] = args.domain_size
            tmp_result_mean_and_std_args_dict['group_attribute_num'] = 1
        elif "CALM" in tmp_algorithm_name:
            tmp_result_mean_and_std_args_dict['granularity'] = args.domain_size
            tmp_result_mean_and_std_args_dict['group_attribute_num'] = group_attribute_num
        elif 'opt' in tmp_algorithm_name:
            del tmp_result_mean_and_std_args_dict['granularity']
            tmp_result_mean_and_std_args_dict['group_attribute_num'] = group_attribute_num
        # elif "Grid" in tmp_algorithm_name:
        #     tmp_result_mean_and_std_args_dict['granularity'] = grid_granularity
        #     tmp_result_mean_and_std_args_dict['group_attribute_num'] = group_attribute_num
        elif 'GHDR' in tmp_algorithm_name:
            # here fanout_set is set to 4. It can be added to the para list for other values
            tmp_result_mean_and_std_args_dict['fanout_set'] = 4
            del tmp_result_mean_and_std_args_dict['granularity']
            tmp_result_mean_and_std_args_dict['group_attribute_num'] = group_attribute_num
        elif 'PHDR' in tmp_algorithm_name:
            # here fanout_set is set to 4. It can be added to the para list for other values
            tmp_result_mean_and_std_args_dict['fanout_set'] = 4
            del tmp_result_mean_and_std_args_dict['granularity']



        for tmp_key in tmp_plot_args_key_in_each_fig_dict:
            tmp_result_mean_and_std_args_dict[tmp_key] = tmp_plot_args_key_in_each_fig_dict[tmp_key]

        tmp_x_list = getattr(args, x_axis_args_key + '_list')
        for tmp_x in tmp_x_list:
            tmp_result_mean_and_std_args_dict[x_axis_args_key] = tmp_x
            if tmp_algorithm_name in ['Uni', 'PHDR']:
                tmp_result_mean_and_std_args_dict['group_attribute_num'] = tmp_result_mean_and_std_args_dict['attribute_num']
            result_mean_and_std_pickle_file_name = dataset.get_result_file_name(tmp_result_mean_and_std_args_dict, file_category='pickle')
            result_mean_and_std_pickle_file_folder = "pickle_result_mean_and_std"
            result_mean_and_std_pickle_file_path = result_mean_and_std_pickle_file_folder + '/' + result_mean_and_std_pickle_file_name
            with open(result_mean_and_std_pickle_file_path, "rb") as algorithm_pickle_fr:
                MNAE_mean = pickle.load(algorithm_pickle_fr)
                MNAE_std = pickle.load(algorithm_pickle_fr)

                MRE_mean = pickle.load(algorithm_pickle_fr)
                MRE_std = pickle.load(algorithm_pickle_fr)

                MSE_mean = pickle.load(algorithm_pickle_fr)
                MSE_std = pickle.load(algorithm_pickle_fr)

            if metric == 'MNAE':
                tmp_y.append(MNAE_mean)
                tmp_yerr.append(MNAE_std)
            elif metric == 'MRE':
                tmp_y.append(MRE_mean)
                tmp_yerr.append(MRE_std)
            elif metric == 'MSE':
                tmp_y.append(MSE_mean)
                tmp_yerr.append(MSE_std)

        return tmp_y, tmp_yerr

    x_list = getattr(args, x_axis_args_key + '_list')
    x = np.array(x_list)

    y_list = []
    yerr_list = []
    label_list = []

    for tmp_algorithm_name in show_algorithm_name_list:
        tmp_y, tmp_yerr = get_y_yerr(tmp_algorithm_name, x_axis_args_key)
        y_list.append(tmp_y)
        yerr_list.append(tmp_yerr)

    ploter = PlotResult()

    # get label_list
    for tmp_algorithm_name in show_algorithm_name_list:
        tmp_label = ploter.algorithm_name_to_label_dict[tmp_algorithm_name]
        label_list.append(tmp_label)

    # for i in range(len(y_list)):
    #     print(i, 'y_list:', y_list[i])
    # print("label_list:", label_list)

    # added 2020-04-06
    fig = plt.figure(figsize= (ploter.width, ploter.height))
    plt.style.use('seaborn-darkgrid')
    for i in range(len(show_algorithm_name_list)):
        tmp_algorithm_name = show_algorithm_name_list[i]
        tmp_maker = ploter.marker_list[ploter.algorithm_name_to_index_dict[tmp_algorithm_name]]
        tmp_color = ploter.color_list[ploter.algorithm_name_to_index_dict[tmp_algorithm_name]]


        plt.errorbar(x, y_list[i], yerr=yerr_list[i], marker= tmp_maker, \
                     fillstyle='none', label=label_list[i],
                     color= tmp_color, markersize=10 * ploter.size_mul,
                     linewidth=2.0 * ploter.size_mul)
    folder_name = "all_algorithm_vary_attribute_num"

    plt.yscale("log")
    plt.xlabel(ploter.x_label_attribute_num, fontsize=ploter.fontsize)

    # if metric == 'MNAE':
    #     plt.ylabel('MAE', fontsize=ploter.fontsize)
    # else:
    #     plt.ylabel(metric, fontsize=ploter.fontsize)


    plt.xticks(x)
    # plt.xticks(fontsize=ploter.fontsize)
    # plt.yticks(fontsize=ploter.fontsize)

    plt.xticks(fontsize=ploter.ticks_fontsize_normal)
    plt.yticks(fontsize=ploter.ticks_fontsize_normal)




    fig_name = ploter.set_fig_name(tmp_plot_args_key_in_each_fig_dict) + '-' + metric
    ploter.save_fig(folder_name, fig_name)

    print("^^^^^^^^^^^plot done.^^^^^^^^^^")

    # create a second figure for the legend
    plt.figure()
    # produce a legend for the objects in the other figure
    plt.figlegend(*fig.gca().get_legend_handles_labels(), fontsize=ploter.fontsize, ncol= len(show_algorithm_name_list))
    fig_legend_name = 'legend'
    ploter.save_fig(folder_name, fig_legend_name)
    print("^^^^^^^^legend^^^plot done.^^^^^^^^^^")

    return


def parallel_new_plot_result_vary_all_algorithm_vary_epsilon(tmp_plot_args_key_in_each_fig_dict:dict = None, task_id = None, args = None):
    print("parallel_new_plot_result_vary_all_algorithm_vary_epsilon task %d begin!!" % task_id)

    show_algorithm_name_list = args.show_algorithm_name_list

    x_axis_args_key = 'epsilon'

    if 'query' in tmp_plot_args_key_in_each_fig_dict['utility_metric']:
        result_mean_and_std_args_key_list = ['algorithm_name', 'dataset_name', 'user_num', 'trajectory_len', 'epsilon',
                                             'utility_metric', 'query_region_num']
    else:
        result_mean_and_std_args_key_list = ['algorithm_name', 'dataset_name', 'user_num', 'trajectory_len', 'epsilon',
                                             'utility_metric']
    # set the args
    for tmp_key in tmp_plot_args_key_in_each_fig_dict:
        setattr(args, tmp_key, tmp_plot_args_key_in_each_fig_dict[tmp_key])
    dataset = dataset_setting.Dataset(args= args)

    def get_y_yerr(tmp_algorithm_name, x_axis_args_key):
        tmp_y =  []
        tmp_yerr = []
        tmp_result_mean_and_std_args_dict = dict()

        for tmp_key in result_mean_and_std_args_key_list: # initialize the dict
            tmp_result_mean_and_std_args_dict[tmp_key] = None

        tmp_result_mean_and_std_args_dict['algorithm_name'] = tmp_algorithm_name
        # if tmp_algorithm_name == "Uni":
        #     tmp_result_mean_and_std_args_dict['granularity'] = args.domain_size
        #     tmp_result_mean_and_std_args_dict['group_attribute_num'] = args.attribute_num

        # # 2021-02-17 for temperory showing optimal PrivTC
        # if tmp_result_mean_and_std_args_dict['algorithm_name']  == 'PrivTC':
        #     tmp_result_mean_and_std_args_dict['algorithm_name'] = 'PrivTC_sigma'
        #     tmp_result_mean_and_std_args_dict['sigma'] = 0.2



        for tmp_key in tmp_plot_args_key_in_each_fig_dict:
            tmp_result_mean_and_std_args_dict[tmp_key] = tmp_plot_args_key_in_each_fig_dict[tmp_key]

        tmp_x_list = getattr(args, x_axis_args_key + '_list')
        for tmp_x in tmp_x_list:
            tmp_result_mean_and_std_args_dict[x_axis_args_key] = tmp_x
            result_mean_and_std_pickle_file_name = dataset.get_result_file_name(tmp_result_mean_and_std_args_dict, file_category='pickle')
            result_mean_and_std_pickle_file_folder = "pickle_result_mean_and_std"
            result_mean_and_std_pickle_file_path = result_mean_and_std_pickle_file_folder + '/' + result_mean_and_std_pickle_file_name
            with open(result_mean_and_std_pickle_file_path, "rb") as algorithm_pickle_fr:
                ans_mean = pickle.load(algorithm_pickle_fr)
                ans_std = pickle.load(algorithm_pickle_fr)

            tmp_y.append(ans_mean)
            tmp_yerr.append(ans_std)

        return tmp_y, tmp_yerr

    x_list = getattr(args, x_axis_args_key + '_list')
    x = np.array(x_list)

    y_list = []
    yerr_list = []
    label_list = []

    for tmp_algorithm_name in show_algorithm_name_list:
        tmp_y, tmp_yerr = get_y_yerr(tmp_algorithm_name, x_axis_args_key)
        y_list.append(tmp_y)
        yerr_list.append(tmp_yerr)

    ploter = PlotResult()

    # get label_list
    for tmp_algorithm_name in show_algorithm_name_list:
        tmp_label = ploter.algorithm_name_to_label_dict[tmp_algorithm_name]
        label_list.append(tmp_label)

    # for i in range(len(y_list)):
    #     print(i, 'y_list:', y_list[i])
    # print("label_list:", label_list)

    fig = plt.figure(figsize=(ploter.width, ploter.height))
    plt.style.use('seaborn-darkgrid')
    for i in range(len(show_algorithm_name_list)):
        tmp_algorithm_name = show_algorithm_name_list[i]
        tmp_maker = ploter.marker_list[ploter.algorithm_name_to_index_dict[tmp_algorithm_name]]
        tmp_color = ploter.color_list[ploter.algorithm_name_to_index_dict[tmp_algorithm_name]]


        plt.errorbar(x, y_list[i], yerr=yerr_list[i], marker= tmp_maker, \
                     fillstyle='none', label=label_list[i],
                     color= tmp_color, markersize=10 * ploter.size_mul,
                     linewidth=2.0 * ploter.size_mul)
    folder_name = "all_algorithm_vary_epsilon"

    plt.yscale("log")
    plt.xlabel(ploter.x_label_epsilon, fontsize=ploter.fontsize)


    plt.xticks(x)
    # plt.xticks(fontsize=ploter.fontsize)
    # plt.yticks(fontsize=ploter.fontsize)

    plt.xticks(fontsize=ploter.ticks_fontsize_small)
    plt.yticks(fontsize=ploter.ticks_fontsize_normal)


    fig_name = ploter.set_fig_name(tmp_plot_args_key_in_each_fig_dict)
    ploter.save_fig(folder_name, fig_name)

    print("^^^^^^^^^^^plot done.^^^^^^^^^^")

    # create a second figure for the legend
    plt.figure()
    # produce a legend for the objects in the other figure
    plt.figlegend(*fig.gca().get_legend_handles_labels(), fontsize=ploter.fontsize, ncol= len(show_algorithm_name_list))
    fig_legend_name = 'legend'
    ploter.save_fig(folder_name, fig_legend_name)
    print("^^^^^^^^legend^^^plot done.^^^^^^^^^^")

    return







#plot class

class PlotResult():
    def __init__(self):

        self.fontsize = 17
        self.scatter_color = 'b'
        self.scatter_size = 80
        self.size_mul = 1

        self.ticks_fontsize_small = 13
        self.ticks_fontsize_normal = 15

        # self.default_width = 6.4
        # self.default_width = 4.8


        self.width = 4.8
        self.height = 2.7

        self.aspect = self.width / self.height


        self.x_label_epsilon = "$\\epsilon$"
        self.x_label_tralen = "$t$"


        self.x_label_dimension_query_volume = "$\\omega$"
        # self.x_label_user_num = "$n$"
        self.x_label_user_num = "$\lg(n)$"

        self.x_label_covariance = "Cov"

        self.x_label_attribute_num = "$d$"
        self.x_label_domain_size = "$c$"
        self.x_label_query_dimension = "$\\lambda$"
        self.x_label_log_g1 = "${\log_2}({g_1})$"
        self.x_label_log_g2 = "${\log_2}({g_2})$"
        self.x_label_log_domain_size = "${\log_2}({c})$"

        self.x_label_sigma = "$\\sigma$"
        self.x_label_alpha = "$\\alpha$"

        # need to add for new algorithm
        self.algorithm_name_to_label_dict = {
            'PrivTC': 'PrivTC',
            'UG': 'UG',
            'NSL': 'NSL',
            'PrivTC_Pro': 'PrivTC_Pro',
            'PrivTC_Pro_2': 'PrivTC_Pro_2',
            'Ngram3': 'Ngram',
        }


        self.marker_list = [
            "^",
            "*",
            "o",
            "s",
            "<",
            "D",
            "p",
            'H',
            'd',
            "v",
            ">",
            'h',
            "x", # below cannot be used as scatter
            '1',
        ]

        # self.marker_list = [
        #     '.',
        #     ',',
        #     'o',
        #     'v',
        #     '^',
        #     '<',
        #     '>',
        #     '1',
        #     '2',
        #     '3',
        #     '4',
        #     's',
        #     'p',
        #     '*',
        #     'h',
        #     'H',
        #     '+',
        #     'x',
        #     'D',
        #     'd',
        #     '|',
        #     '_',
        # ]


        self.color_list = [
            '#ff0000',
            '#a0522d',
            '#87ceeb',
            '#3d9140',
            '#808a87',
            '#ff00ff',
            '#ff9912',
            '#e3cf57',
            '#BC8F8F',
            '#6a5acd',
            '#000000',
            '#004c99',
            '#E5FFCC'
        ]


        # need to add for new algorithm
        self.algorithm_name_list = [
            'PrivTC',
            'UG',
            'NSL',
            'PrivTC_Pro',
            'PrivTC_Pro_2',
            'Ngram3',
        ]


        # self.label_name_list =[
        #     "Uni",
        #     "1_way",
        #     "CALM",
        #     "Grid(d,d)",
        #     "Grid(2)",
        #     "Grid(4)",
        #     "Grid(8)",
        #     "Grid(16)",
        #     "Grid(32)",
        #     "Grid(2,d)",
        #     'OHG',
        #     'ONG',
        #     "Grid(4,d)",
        #     "Grid(8,d)",
        #     "Grid(16,d)",
        #     "Grid(32,d)",
        #     "Grid(2,4)",
        #     "Grid(2,8)",
        #     "Grid(2,16)",
        #     "Grid(2,32)",
        #     "Grid(4,8)",
        #     "Grid(4,16)",
        #     "Grid(4,32)",
        #     "Grid(8,16)",
        #     "Grid(8,32)",
        # ]

        # self.label_name_list = [
        #     "Uni",
        #     "1_way",
        #     "CALM_2",
        #     "CALM_1+2",
        #     "Grid_2-g_2",
        #     "Grid_2-g_4",
        #     "Grid_2-g_8",
        #     "Grid_2-g_16",
        #     "Grid_2-g_32",
        #     "Grid_1+2-g_2",
        #     "Grid_1+2-g_4",
        #     "Grid_1+2-g_8",
        #     "Grid_1+2-g_16",
        #     "Grid_1+2-g_32",
        #     "Grid_1+2_ad-g_2-g1_4",
        #     "Grid_1+2_ad-g_2-g1_8",
        #     "Grid_1+2_ad-g_2-g1_16",
        #     "Grid_1+2_ad-g_2-g1_32",
        #     "Grid_1+2_ad-g_4-g1_8",
        #     "Grid_1+2_ad-g_4-g1_16",
        #     "Grid_1+2_ad-g_4-g1_32",
        #     "Grid_1+2_ad-g_8-g1_16",
        #     "Grid_1+2_ad-g_8-g1_32",
        # ]


        self.label_name_dict = None
        # np.random.seed(1)
        # np.random.seed(3)
        # np.random.shuffle(self.color_list)
        self.generate_algorithm_name_to_index_dict()

    def generate_algorithm_name_to_index_dict(self):
        self.algorithm_name_to_index_dict = dict()
        for i in range(len(self.algorithm_name_list)):
            self.algorithm_name_to_index_dict[self.algorithm_name_list[i]] = i


    def set_fig_name(self, tmp_plot_args_key_in_each_fig_dict:dict = None):
        fig_name = ""
        dict_len = len(tmp_plot_args_key_in_each_fig_dict)
        i = 0

        for tmp_key in tmp_plot_args_key_in_each_fig_dict:
            tmp_v = tmp_plot_args_key_in_each_fig_dict[tmp_key]
            if tmp_key == "total_query_volume" or tmp_key == 'epsilon' or tmp_key == "dimension_query_volume" :
                # print ("tmp_v", tmp_v, type(tmp_v))
                tmp_v = int(tmp_v * 10)
                if tmp_v < 1:
                    tmp_v = int(tmp_v * 10)
            if tmp_key == 'MRE_gama':
                tmp_v = -1 * int(math.log10(tmp_v))
            fig_name = fig_name + args.args2acronym_dict[tmp_key] + '-' + str(tmp_v)
            if i < dict_len - 1:
                fig_name += '-'
            i += 1
        return fig_name

    def save_fig(self, folder_name = "", fig_name = None):
        fig_path = "figure/" + folder_name
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        # print ("fig_name:", fig_name)
        # os.system("pause")
        # plt.savefig(fig_path + fig_name + ".eps", format='eps',  bbox_inches='tight')
        plt.savefig(fig_path + "/" + fig_name + ".pdf", format='pdf', bbox_inches='tight')
        plt.close()


    def show_color_marker(self):
        color_num = len(self.color_list)
        print ("color_num", color_num)
        print ("marker_num", len(self.marker_list))
        x =[1,2,3,4,5]
        y =[1,1,1,1,1]
        x = np.array(x)
        y_list = np.array(y)
        yerr = 0.2
        plt.figure()
        for i in range(color_num):
            plt.errorbar(x, y_list + i, yerr=yerr, fillstyle='none', color = self.color_list[i], marker= self.marker_list[i],
                         label=str(i), markersize=10 * self.size_mul)
        plt.show()



if __name__ == '__main__':
    # aa = PlotResult()
    # aa.show_color_marker()
    pass