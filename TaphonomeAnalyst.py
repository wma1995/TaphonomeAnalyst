import re
import copy
import argparse
import warnings
import matplotlib
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from math import ceil
from scipy import stats
from matplotlib import cm
from functools import reduce
from operator import itemgetter
import matplotlib.pyplot as plt
import community as community_louvain
from matplotlib_venn import venn2, venn3
from itertools import combinations, permutations
from skbio.diversity.alpha import chao1, chao1_ci
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)


# 水生
def clusterenv(args):
    all_category_set = set()
    for i in data_df_dict.keys():
        all_category_set |= set(data_df_dict[i][args.category])
    all_category_set.remove('unknown')
    all_otu_df_list = list()
    for i in data_df_dict.keys():
        category_list = [m for m, n in data_df_dict[i][args.category].value_counts().items() if m != 'unknown']
        count_list = [n for m, n in data_df_dict[i][args.category].value_counts().items() if m != 'unknown']
        diff_set = all_category_set - set(category_list)
        category_list.extend(list(diff_set))
        count_list.extend([0] * len(diff_set))
        all_otu_df_list.append(pd.DataFrame(count_list, index=category_list))
    otu_category_df = pd.concat([i.T for i in all_otu_df_list], ignore_index=True)
    otu_category_df.index = data_df_dict.keys()
    del_columns_list = list()
    for i in otu_category_df.columns:
        if all([j < 5 for j in otu_category_df[i]]):
            del_columns_list.append(i)
    otu_category_df_filter = otu_category_df.drop(columns=del_columns_list)
    if args.aquatic:
        otu_category_df_filter = otu_category_df_filter.loc[:, otu_category_df_filter.columns.isin(args.aquatic)]
    otu_category_df_filter_standardscale = otu_category_df_filter.apply(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    mergings = linkage(otu_category_df_filter_standardscale, method='average', metric='braycurtis',
                       optimal_ordering=True)
    dendrogram(Z=mergings, labels=otu_category_df_filter_standardscale.index, leaf_rotation=90)
    plt.savefig(fname=f'{args.output}_tree.{args.format}', bbox_inches='tight')
    plt.cla()
    sns.clustermap(otu_category_df_filter_standardscale.T, cmap='Blues', row_cluster=False, method='average',
                   metric='braycurtis')
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')
    print('Finished!')


def divloc(args):
    color_list = ["#BC3C29FF", "#0072B5FF", "#E18727FF"]
    all_category_set = set()
    for i in data_df_dict.keys():
        all_category_set |= set(data_df_dict[i][args.category])
    all_category_set.remove('unknown')
    all_otu_df_list = list()
    for i in data_df_dict.keys():
        category_list = [m for m, n in data_df_dict[i][args.category].value_counts().items() if m != 'unknown']
        count_list = [n for m, n in data_df_dict[i][args.category].value_counts().items() if m != 'unknown']
        diff_set = all_category_set - set(category_list)
        category_list.extend(list(diff_set))
        count_list.extend([0] * len(diff_set))
        all_otu_df_list.append(pd.DataFrame(count_list, index=category_list))
    otu_category_df = pd.concat([i.T for i in all_otu_df_list], ignore_index=True)
    otu_category_df.index = data_df_dict.keys()
    place_set = set([re.sub(r'[0-9]+', '', i) for i in otu_category_df.index])
    place_plot_dict = {i: [j for j in otu_category_df.index if j.startswith(i) and j.replace(i, '').isdigit()] for i in
                       place_set}
    otu_category_df_ = pd.DataFrame(columns=otu_category_df.columns)
    for i in place_plot_dict.keys():
        otu_category_df_.loc['/'.join(place_plot_dict[i])] = np.ravel(
            sum([otu_category_df.loc[[j]].values for j in place_plot_dict[i]]))
    otu_category_df_ = otu_category_df_.T
    subset = []
    for i in otu_category_df_.columns:
        tmp_list = []
        for index, row in otu_category_df_.iterrows():
            tmp_list.extend([index] * row[i])
        subset.append(set(tmp_list))
    plt.figure(figsize=(800 / 150, 800 / 150), dpi=200)
    num_plot = len(place_plot_dict.keys())
    if num_plot == 2:
        venn2(subsets=subset, set_labels=place_plot_dict.keys(), set_colors=tuple(color_list[:num_plot]),
              alpha=0.8, normalize_to=1.0)
    elif num_plot == 3:
        venn3(subsets=subset, set_labels=place_plot_dict.keys(), set_colors=tuple(color_list[:num_plot]),
              alpha=0.8, normalize_to=1.0)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


# 自定义分组
def divenv(args):
    color_list = ["#BC3C29FF", "#0072B5FF", "#E18727FF"]
    all_category_set = set()
    for i in data_df_dict.keys():
        all_category_set |= set(data_df_dict[i][args.category])
    all_category_set.remove('unknown')
    all_otu_df_list = list()
    for i in data_df_dict.keys():
        category_list = [m for m, n in data_df_dict[i][args.category].value_counts().items() if m != 'unknown']
        count_list = [n for m, n in data_df_dict[i][args.category].value_counts().items() if m != 'unknown']
        diff_set = all_category_set - set(category_list)
        category_list.extend(list(diff_set))
        count_list.extend([0] * len(diff_set))
        all_otu_df_list.append(pd.DataFrame(count_list, index=category_list))
    otu_category_df = pd.concat([i.T for i in all_otu_df_list], ignore_index=True)
    otu_category_df.index = data_df_dict.keys()
    otu_category_df_ = pd.DataFrame(columns=otu_category_df.columns)
    groups_list = args.groups
    for i in groups_list:
        values = 0
        for j in i.split('/'):
            values += otu_category_df.loc[[j]].values
        otu_category_df_.loc[i] = np.ravel(values)
    otu_category_df_ = otu_category_df_.T
    subset = []
    for i in otu_category_df_.columns:
        tmp_list = []
        for index, row in otu_category_df_.iterrows():
            tmp_list.extend([index] * row[i])
        subset.append(set(tmp_list))
    plt.figure(figsize=(800 / 150, 800 / 150), dpi=200)
    num_plot = len(otu_category_df_.columns)
    if num_plot == 2:
        venn2(subsets=subset, set_labels=otu_category_df_.columns, set_colors=tuple(color_list[:num_plot]),
              alpha=0.8, normalize_to=1.0)
    elif num_plot == 3:
        venn3(subsets=subset, set_labels=otu_category_df_.columns, set_colors=tuple(color_list[:num_plot]),
              alpha=0.8, normalize_to=1.0)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def TGotus(args):
    plot_df_dict = {
        i: data_df_dict[i][args.category].value_counts().rename_axis(args.category).reset_index(name=i).set_index(
            args.category)
        for i in data_df_dict}
    otu_category_df_T = reduce(
        lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True),
        list(plot_df_dict.values()))
    otu_category_df_T['total'] = otu_category_df_T.apply(lambda x: x.sum(), axis=1)
    category_count_df = otu_category_df_T[['total']].reset_index()

    category_grade_df = pd.DataFrame(columns=grade_list)
    for i in category_count_df[args.category]:
        grade_A2E_count_list = list()
        tmp_grade_tuple_list = list()
        for j in data_df_dict.keys():
            tmp_grade_tuple_list.extend([(m, n) for m, n in data_df_dict[j][data_df_dict[j][args.category] == i][
                'taphonomic grade'].value_counts().items()])
        for k in grade_list:
            try:
                grade_A2E_count_list.append(sum([j[1] for j in tmp_grade_tuple_list if j[0] == k]))
            except Exception as e:
                grade_A2E_count_list.append(0)
        category_grade_df.loc[category_count_df[args.category].tolist().index(i)] = grade_A2E_count_list
    category_grade_count_df = pd.concat([category_count_df, category_grade_df], axis=1).set_index(args.category)
    for i in grade_list:
        category_grade_count_df[i] /= category_grade_count_df['total']
    category_grade_count_df.sort_values(by=['A'], inplace=True)
    category_grade_count_df_ = category_grade_count_df.drop('total', axis=1)
    fig = category_grade_count_df_.plot.barh(stacked=True, color=color_list[:len(grade_list)])
    fig.set_ylabel('')
    ax2 = fig.twinx()
    ax2.set_ylim(fig.get_ylim())
    ax2.set_yticks([i for i in range(category_grade_count_df.shape[0])])
    ax2.set_yticklabels([' n = ' + str(int(i)) for i in category_grade_count_df['total']])
    fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), fontsize=12)
    fig.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def TGplots(args):
    all_plot_grade_count_df = pd.DataFrame(index=data_df_dict.keys(), columns=grade_list)
    for i in data_df_dict.keys():
        grade_and_count_list = [(m, n) for m, n in
                                data_df_dict[i]['taphonomic grade'].value_counts().sort_index().items()]
        tmp_grade_list = list(map(itemgetter(0), grade_and_count_list))
        tmp_grade_count_list = list(map(itemgetter(1), grade_and_count_list))
        if len(tmp_grade_list) != 5:
            tmp_grade_count_list = [0 if j not in tmp_grade_list else tmp_grade_count_list[tmp_grade_list.index(j)]
                                    for j in grade_list]
        all_plot_grade_count_df.loc[i] = tmp_grade_count_list

    all_plot_grade_count_df['total'] = all_plot_grade_count_df.apply(lambda x: x.sum(), axis=1)
    for i in grade_list:
        all_plot_grade_count_df[i] /= all_plot_grade_count_df['total']
    all_plot_grade_count_df_sort = all_plot_grade_count_df.sort_values(by=['A'])
    all_plot_grade_count_df_sort_ = all_plot_grade_count_df_sort.drop('total', axis=1)
    fig = all_plot_grade_count_df_sort_.plot.barh(stacked=True, color=color_list[:len(grade_list)])
    ax2 = fig.twinx()
    ax2.set_ylim(fig.get_ylim())
    ax2.set_yticks([i for i in range(all_plot_grade_count_df_sort.shape[0])])
    ax2.set_yticklabels([' n = ' + str(i) for i in all_plot_grade_count_df_sort['total']])
    fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), fontsize=12)
    fig.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def abundplots(args):
    all_plot_grade_count_df = pd.DataFrame(index=data_df_dict.keys(), columns=grade_list)
    for i in data_df_dict.keys():
        grade_and_count_list = [(m, n) for m, n in
                                data_df_dict[i]['taphonomic grade'].value_counts().sort_index().items()]
        tmp_grade_list = list(map(itemgetter(0), grade_and_count_list))
        tmp_grade_count_list = list(map(itemgetter(1), grade_and_count_list))
        if len(tmp_grade_list) != 5:
            tmp_grade_count_list = [0 if j not in tmp_grade_list else tmp_grade_count_list[tmp_grade_list.index(j)]
                                    for j in grade_list]
        all_plot_grade_count_df.loc[i] = tmp_grade_count_list
    all_plot_grade_count_df['total'] = all_plot_grade_count_df.apply(lambda x: x.sum(), axis=1)
    for i in grade_list:
        all_plot_grade_count_df[i] /= all_plot_grade_count_df['total']
    plot_normalize_df_dict = {
        i: data_df_dict[i][args.category].value_counts(normalize=True).rename_axis(args.category).reset_index(
            name=i).set_index(args.category)
        for i in data_df_dict}
    otu_category_normalize_df = reduce(
        lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True),
        list(plot_normalize_df_dict.values())).T
    fig = otu_category_normalize_df.plot.barh(stacked=True, color=color_list[:len(otu_category_normalize_df.columns)])
    ax2 = fig.twinx()
    ax2.set_ylim(fig.get_ylim())
    ax2.set_yticks([i for i in range(otu_category_normalize_df.shape[0])])
    ax2.set_yticklabels([' n = ' + str(int(i)) for i in all_plot_grade_count_df['total']])
    fig.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), fontsize=12)
    fig.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def cooccurnet(args):
    plot_df_dict = {
        i: (data_df_dict[i][args.category].value_counts(normalize=True) * 3000).rename_axis(args.category).reset_index(
            name=i).set_index(args.category) for i in data_df_dict}
    otu_category_df = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True),
                             list(plot_df_dict.values())).fillna(0).T
    try:
        del otu_category_df['unknown']
    except Exception as e:
        pass
    all_comb = list(combinations(list(otu_category_df), 2))
    limit_list = []
    nodes = []
    edges = []
    if args.corr == 'pearson':
        for comb in all_comb:
            corr_and_p_value = stats.pearsonr(otu_category_df[f'{comb[0]}'], otu_category_df[f'{comb[1]}'])
            if corr_and_p_value[1] < args.p_value and corr_and_p_value[0] > args.corr_coef:
                limit_list.append(comb[0])
                limit_list.append(comb[1])
        limit_list = list(set(limit_list))
        for i in limit_list:
            node = (limit_list.index(i), i)
            nodes.append(node)
        for comb in all_comb:
            corr_and_p_value = stats.pearsonr(otu_category_df[f'{comb[0]}'], otu_category_df[f'{comb[1]}'])
            if corr_and_p_value[1] < args.p_value and corr_and_p_value[0] > args.corr_coef:
                edge = (limit_list.index(comb[0]), limit_list.index(comb[1]), corr_and_p_value[0])
                edges.append(edge)
    elif args.corr == 'spearman':
        for comb in all_comb:
            corr_and_p_value = stats.spearmanr(otu_category_df[f'{comb[0]}'], otu_category_df[f'{comb[1]}'])
            if corr_and_p_value[1] < args.p_value and corr_and_p_value[0] > args.corr_coef:
                limit_list.append(comb[0])
                limit_list.append(comb[1])
        limit_list = list(set(limit_list))
        for i in limit_list:
            node = (limit_list.index(i), i)
            nodes.append(node)
        for comb in all_comb:
            corr_and_p_value = stats.spearmanr(otu_category_df[f'{comb[0]}'], otu_category_df[f'{comb[1]}'])
            if corr_and_p_value[1] < args.p_value and corr_and_p_value[0] > args.corr_coef:
                edge = (limit_list.index(comb[0]), limit_list.index(comb[1]), corr_and_p_value[0])
                edges.append(edge)
    elif args.corr == 'kendall':
        for comb in all_comb:
            corr_and_p_value = stats.kendalltau(otu_category_df[f'{comb[0]}'], otu_category_df[f'{comb[1]}'])
            if corr_and_p_value[1] < args.p_value and corr_and_p_value[0] > args.corr_coef:
                limit_list.append(comb[0])
                limit_list.append(comb[1])
        limit_list = list(set(limit_list))
        for i in limit_list:
            node = (limit_list.index(i), i)
            nodes.append(node)
        for comb in all_comb:
            corr_and_p_value = stats.kendalltau(otu_category_df[f'{comb[0]}'], otu_category_df[f'{comb[1]}'])
            if corr_and_p_value[1] < args.p_value and corr_and_p_value[0] > args.corr_coef:
                edge = (limit_list.index(comb[0]), limit_list.index(comb[1]), corr_and_p_value[0])
                edges.append(edge)
    with open('./cooccurnet.gml', 'w') as f:
        f.write("graph\n")
        f.write("[\n")
        for i in nodes:
            i_1 = i[1].replace('（', '(').replace('）', ')')
            f.write("node\n")
            f.write("[\n")
            f.write("id " + str(i[0]) + "\n")
            f.write("label \"" + i_1 + "\"\n")
            f.write("]\n")
        for i in edges:
            f.write("edge\n")
            f.write("[\n")
            f.write("source " + str(i[0]) + "\n")
            f.write("target " + str(i[1]) + "\n")
            f.write("value " + str(i[2]) + "\n")
            f.write("]\n")
        f.write("]")
    f.close()
    G = nx.read_gml('./cooccurnet.gml')
    partition = community_louvain.best_partition(G)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(40, 24), frameon=False)
    for key, spine in ax.spines.items():
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    cmap = cm.get_cmap(None, max(partition.values()) + 1)
    node_color_list = []
    for i in list(partition.values()):
        node_color_list.append(color_list[i])
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=400,
                           cmap=cmap, node_color=node_color_list, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, alpha=0.8, font_size=20, verticalalignment='bottom')
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def chao(args):
    otu_df_dict = dict()
    for i in data_df_dict.keys():
        result_dict = dict()
        for j in range(50, data_df_dict[i].shape[0] + 1, 50):
            result_list = [k for k in data_df_dict[i][:j][args.category] if not k in [np.nan, 'unknown']]
            result_dict[j] = [(k, result_list.count(k)) for k in set(result_list)]
        otu_df_dict[i] = result_dict
    plot_dict = dict.fromkeys([i[:-1] for i in all_sheet_name_list])
    for i in plot_dict.keys():
        plot_dict[i] = [n for n in all_sheet_name_list if n[:-1] == i]
    plot_otu_df_dict = dict()
    for i in plot_dict.keys():
        plot_df_list = list()
        for j in plot_dict[i]:
            plot_df_list.extend([k[0] for k in otu_df_dict[j][3000]])
        plot_otu_df_dict[i] = set(plot_df_list)
    otu_df_dict_ = copy.deepcopy(otu_df_dict)
    for i in plot_dict.keys():
        for j in otu_df_dict.keys():
            if j.startswith(i) and j.replace(i, '').isdigit():
                for k in range(50, 3001, 50):
                    diff = set(plot_otu_df_dict[i]).difference(set([l[0] for l in otu_df_dict[j][k]]))
                    zip_diff = zip(list(diff), [0] * len(diff))
                    otu_df_dict_[j][k].extend([l for l in zip_diff])
    step_df_list = []
    for i in plot_dict.keys():
        for j in plot_dict[i]:
            for k in range(50, 3001, 50):
                columns = [l[0] for l in otu_df_dict_[j][k]]
                line = [l[1] for l in otu_df_dict_[j][k]]
                df = pd.DataFrame(line, columns)
                step_df_list.append(df)
    step_idx = []
    step_len_list = [i.shape[0] for i in step_df_list]
    step_len_set_list = list(set(step_len_list))
    step_len_set_list.sort(key=step_len_list.index)
    for i in step_len_set_list:
        step_idx.append([k for k, l in enumerate([j.shape[0] for j in step_df_list]) if l == i][-1])
    step_dict = dict()
    count = 0
    for i in plot_dict.keys():
        step_dict[i] = [i for i in zip([0] + step_idx[:-1], step_idx)][count]
        count += 1
    step_dict_ = dict()
    for i in step_dict.keys():
        step_dict_[i] = dict()
        count_ = 0
        for j in plot_dict[i]:
            if step_dict[i][0] == 0:
                if count_ == 0:
                    step_dict_[i][j] = (
                        step_dict[i][0], int((step_dict[i][1] + 1 - step_dict[i][0]) / len(plot_dict[i]) - 1))
                else:
                    step_dict_[i][j] = (count_ * int((step_dict[i][1] - step_dict[i][0] + 1) / len(plot_dict[i])),
                                        (count_ + 1) * int(
                                            (step_dict[i][1] - step_dict[i][0] + 1) / len(plot_dict[i])) - 1)
            else:
                if count_ == 0:
                    step_dict_[i][j] = (
                        step_dict[i][0] + 1,
                        step_dict[i][0] + int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])))
                else:
                    step_dict_[i][j] = (
                        step_dict[i][0] + 1 + count_ * int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])),
                        step_dict[i][0] + (count_ + 1) * int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])))
            count_ += 1
    step_df_concat = pd.concat([i.T for i in step_df_list], ignore_index=True).fillna(0)
    plot_chao_dict = dict()
    plot_chao_ci_dict = dict()
    for i in step_dict_.keys():
        for j in step_dict_[i]:
            plot_chao_dict[j] = list()
            plot_chao_ci_dict[j] = list()
            for k in range(step_dict_[i][j][0], step_dict_[i][j][1] + 1):
                plot_chao_dict[j].append(chao1(step_df_concat.T[k]))
                plot_chao_ci_dict[j].append(chao1_ci(step_df_concat.T[k]))
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(15, 20))
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
    loc_list = [i for i in step_dict_.keys()]
    sns.set_style('ticks')
    sns.set_palette(palette=color_list[:len(loc_list)], n_colors=len(loc_list), desat=None, color_codes=True)
    for i in loc_list:
        plot_list = [n for n in plot_chao_dict.keys() if n.startswith(i) and n.replace(i, '').isdigit()]
        for j in plot_list:
            sns.regplot([n for n in range(50, 3001, 50)], plot_chao_dict[j], logx=True, label=j,
                        ax=ax_list[loc_list.index(i)], scatter_kws={'s': 12})
            ci_0 = np.array([n[0] for n in plot_chao_ci_dict[j]]) - np.array(plot_chao_dict[j])
            ci_0 = [abs(n) for n in ci_0]
            ci_1 = np.array([n[1] for n in plot_chao_ci_dict[j]]) - np.array(plot_chao_dict[j])
            ci_1 = [abs(n) for n in ci_1]
            ci = np.array([ci_0, ci_1])
            ax_list[loc_list.index(i)].errorbar([n for n in range(50, 3001, 50)], plot_chao_dict[j], yerr=ci,
                                                fmt='none')
            ax_list[loc_list.index(i)].tick_params(labelsize=16)
            ax_list[loc_list.index(i)].xaxis.set_tick_params(labelbottom=True)
            # if loc_list.index(i) == 4 or loc_list.index(i) == 5:
            ax_list[loc_list.index(i)].set_xlabel('Number of fossil individuals', fontsize=18)
            # if loc_list.index(i) % 2 == 0:
            ax_list[loc_list.index(i)].set_ylabel(f'Number of {args.category}', fontsize=18)
            ax_list[loc_list.index(i)].legend(markerscale=3, fontsize=18)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def samplecurve(args):
    otu_df_dict = dict()
    for i in data_df_dict.keys():
        result_dict = dict()
        for j in range(50, data_df_dict[i].shape[0] + 1, 50):
            result_list = [k for k in data_df_dict[i][:j][args.category] if k not in [np.nan, 'unknown']]
            result_dict[j] = [(k, result_list.count(k)) for k in set(result_list)]
        otu_df_dict[i] = result_dict
    plot_dict = dict.fromkeys([i[:-1] for i in all_sheet_name_list])
    for i in plot_dict.keys():
        plot_dict[i] = [n for n in all_sheet_name_list if n[:-1] == i]
    plot_otu_df_dict = dict()
    for i in plot_dict.keys():
        plot_df_list = list()
        for j in plot_dict[i]:
            plot_df_list.extend([k[0] for k in otu_df_dict[j][3000]])
        plot_otu_df_dict[i] = set(plot_df_list)
    otu_df_dict_ = copy.deepcopy(otu_df_dict)
    for i in plot_dict.keys():
        for j in otu_df_dict.keys():
            if j.startswith(i) and j.replace(i, '').isdigit():
                for k in range(50, 3001, 50):
                    diff = set(plot_otu_df_dict[i]).difference(set([l[0] for l in otu_df_dict[j][k]]))
                    zip_diff = zip(list(diff), [0] * len(diff))
                    otu_df_dict_[j][k].extend([l for l in zip_diff])
    step_df_list = []
    for i in plot_dict.keys():
        for j in plot_dict[i]:
            for k in range(50, 3001, 50):
                columns = [l[0] for l in otu_df_dict_[j][k]]
                line = [l[1] for l in otu_df_dict_[j][k]]
                df = pd.DataFrame(line, columns)
                step_df_list.append(df)
    step_idx = []
    step_len_list = [i.shape[0] for i in step_df_list]
    step_len_set_list = list(set(step_len_list))
    step_len_set_list.sort(key=step_len_list.index)
    for i in step_len_set_list:
        step_idx.append([k for k, l in enumerate([j.shape[0] for j in step_df_list]) if l == i][-1])
    step_dict = dict()
    count = 0
    for i in plot_dict.keys():
        step_dict[i] = [i for i in zip([0] + step_idx[:-1], step_idx)][count]
        count += 1
    step_dict_ = dict()
    for i in step_dict.keys():
        step_dict_[i] = dict()
        count_ = 0
        for j in plot_dict[i]:
            if step_dict[i][0] == 0:
                if count_ == 0:
                    step_dict_[i][j] = (
                        step_dict[i][0], int((step_dict[i][1] + 1 - step_dict[i][0]) / len(plot_dict[i]) - 1))
                else:
                    step_dict_[i][j] = (count_ * int((step_dict[i][1] - step_dict[i][0] + 1) / len(plot_dict[i])),
                                        (count_ + 1) * int(
                                            (step_dict[i][1] - step_dict[i][0] + 1) / len(plot_dict[i])) - 1)
            else:
                if count_ == 0:
                    step_dict_[i][j] = (
                        step_dict[i][0] + 1,
                        step_dict[i][0] + int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])))
                else:
                    step_dict_[i][j] = (
                        step_dict[i][0] + 1 + count_ * int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])),
                        step_dict[i][0] + (count_ + 1) * int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])))
            count_ += 1
    step_df_concat = pd.concat([i.T for i in step_df_list], ignore_index=True).fillna(0)
    plot_real_dict = dict()
    for i in step_dict_.keys():
        for j in step_dict_[i]:
            plot_real_dict[j] = list()
            for k in range(step_dict_[i][j][0], step_dict_[i][j][1] + 1):
                plot_real_dict[j].append(len([l for l in step_df_concat.T[k] if int(l) != 0]))
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(15, 20))
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
    loc_list = [i for i in step_dict_.keys()]
    sns.set_style('ticks')
    sns.set_palette(palette=color_list[:len(loc_list)], n_colors=len(loc_list), desat=None, color_codes=True)
    for i in loc_list:
        plot_list = [n for n in plot_real_dict.keys() if n.startswith(i) and n.replace(i, '').isdigit()]
        for j in plot_list:
            sns.regplot([n for n in range(50, 3001, 50)], plot_real_dict[j], logx=True, label=j,
                        ax=ax_list[loc_list.index(i)], scatter_kws={'s': 12})
            ax_list[loc_list.index(i)].errorbar([n for n in range(50, 3001, 50)], plot_real_dict[j], fmt='none')
            ax_list[loc_list.index(i)].tick_params(labelsize=16)
            ax_list[loc_list.index(i)].xaxis.set_tick_params(labelbottom=True)
            # if loc_list.index(i) == 4 or loc_list.index(i) == 5:
            ax_list[loc_list.index(i)].set_xlabel('Number of fossil individuals', fontsize=18)
            # if loc_list.index(i) % 2 == 0:
            ax_list[loc_list.index(i)].set_ylabel(f'Number of {args.category}', fontsize=18)
            ax_list[loc_list.index(i)].legend(markerscale=3, fontsize=18)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def _otus_rare(freq_counts, rare_threshold):
    return freq_counts[1:rare_threshold + 1].sum()


def _otus_abundant(freq_counts, rare_threshold):
    return freq_counts[rare_threshold + 1:].sum()


def _number_rare(freq_counts, rare_threshold, gamma=False):
    n_rare = 0
    if gamma:
        for i, j in enumerate(freq_counts[:rare_threshold + 1]):
            n_rare = n_rare + (i * j) * (i - 1)
    else:
        for i, j in enumerate(freq_counts[:rare_threshold + 1]):
            n_rare = n_rare + (i * j)
    return n_rare


def ace_(counts, rare_threshold=10):
    counts = np.asarray(counts)
    freq_counts = np.bincount(counts)
    s_rare = _otus_rare(freq_counts, rare_threshold)
    singles = freq_counts[1]
    s_abun = _otus_abundant(freq_counts, rare_threshold)
    if s_rare == 0:
        return s_abun
    n_rare = _number_rare(freq_counts, rare_threshold)
    c_ace = 1 - singles / n_rare
    top = s_rare * _number_rare(freq_counts, rare_threshold, gamma=True)
    bottom = c_ace * n_rare * (n_rare - 1)
    gamma_ace = (top / bottom) - 1
    if gamma_ace < 0:
        gamma_ace = 0
    return s_abun + (s_rare / c_ace) + ((singles / c_ace) * gamma_ace)


def ace(args):
    otu_df_dict = dict()
    for i in data_df_dict.keys():
        result_dict = dict()
        for j in range(50, data_df_dict[i].shape[0] + 1, 50):
            result_list = [k for k in data_df_dict[i][:j][args.category] if not k in [np.nan, 'unknown']]
            result_dict[j] = [(k, result_list.count(k)) for k in set(result_list)]
        otu_df_dict[i] = result_dict
    plot_dict = dict.fromkeys([i[:-1] for i in all_sheet_name_list])
    for i in plot_dict.keys():
        plot_dict[i] = [n for n in all_sheet_name_list if n[:-1] == i]
    plot_otu_df_dict = dict()
    for i in plot_dict.keys():
        plot_df_list = list()
        for j in plot_dict[i]:
            plot_df_list.extend([k[0] for k in otu_df_dict[j][3000]])
        plot_otu_df_dict[i] = set(plot_df_list)
    otu_df_dict_ = copy.deepcopy(otu_df_dict)
    for i in plot_dict.keys():
        for j in otu_df_dict.keys():
            if j.startswith(i) and j.replace(i, '').isdigit():
                for k in range(50, 3001, 50):
                    diff = set(plot_otu_df_dict[i]).difference(set([l[0] for l in otu_df_dict[j][k]]))
                    zip_diff = zip(list(diff), [0] * len(diff))
                    otu_df_dict_[j][k].extend([l for l in zip_diff])
    step_df_list = []
    for i in plot_dict.keys():
        for j in plot_dict[i]:
            for k in range(50, 3001, 50):
                columns = [l[0] for l in otu_df_dict_[j][k]]
                line = [l[1] for l in otu_df_dict_[j][k]]
                df = pd.DataFrame(line, columns)
                step_df_list.append(df)
    step_idx = []
    step_len_list = [i.shape[0] for i in step_df_list]
    step_len_set_list = list(set(step_len_list))
    step_len_set_list.sort(key=step_len_list.index)
    for i in step_len_set_list:
        step_idx.append([k for k, l in enumerate([j.shape[0] for j in step_df_list]) if l == i][-1])
    step_dict = dict()
    count = 0
    for i in plot_dict.keys():
        step_dict[i] = [i for i in zip([0] + step_idx[:-1], step_idx)][count]
        count += 1
    step_dict_ = dict()
    for i in step_dict.keys():
        step_dict_[i] = dict()
        count_ = 0
        for j in plot_dict[i]:
            if step_dict[i][0] == 0:
                if count_ == 0:
                    step_dict_[i][j] = (
                        step_dict[i][0], int((step_dict[i][1] + 1 - step_dict[i][0]) / len(plot_dict[i]) - 1))
                else:
                    step_dict_[i][j] = (count_ * int((step_dict[i][1] - step_dict[i][0] + 1) / len(plot_dict[i])),
                                        (count_ + 1) * int(
                                            (step_dict[i][1] - step_dict[i][0] + 1) / len(plot_dict[i])) - 1)
            else:
                if count_ == 0:
                    step_dict_[i][j] = (
                        step_dict[i][0] + 1,
                        step_dict[i][0] + int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])))
                else:
                    step_dict_[i][j] = (
                        step_dict[i][0] + 1 + count_ * int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])),
                        step_dict[i][0] + (count_ + 1) * int((step_dict[i][1] - step_dict[i][0]) / len(plot_dict[i])))
            count_ += 1
    step_df_concat = pd.concat([i.T for i in step_df_list], ignore_index=True).fillna(0)
    plot_ace_dict = dict()
    for i in step_dict_.keys():
        for j in step_dict_[i]:
            plot_ace_dict[j] = list()
            for k in range(step_dict_[i][j][0], step_dict_[i][j][1] + 1):
                plot_ace_dict[j].append(ace_(step_df_concat.T[k].astype(np.int64), rare_threshold=args.rare))
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(15, 20))
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
    loc_list = [i for i in step_dict_.keys()]
    sns.set_style('ticks')
    sns.set_palette(palette=color_list[:len(loc_list)], n_colors=len(loc_list), desat=None, color_codes=True)
    for i in loc_list:
        plot_list = [n for n in plot_ace_dict.keys() if n.startswith(i) and n.replace(i, '').isdigit()]
        for j in plot_list:
            sns.regplot([n for n in range(50, 3001, 50)], plot_ace_dict[j], logx=True, label=j,
                        ax=ax_list[loc_list.index(i)], scatter_kws={'s': 12})
            ax_list[loc_list.index(i)].errorbar([n for n in range(50, 3001, 50)], plot_ace_dict[j], fmt='none')
            ax_list[loc_list.index(i)].tick_params(labelsize=16)
            ax_list[loc_list.index(i)].xaxis.set_tick_params(labelbottom=True)
            # if loc_list.index(i) == 4 or loc_list.index(i) == 5:
            ax_list[loc_list.index(i)].set_xlabel('Number of fossil individuals', fontsize=18)
            # if loc_list.index(i) % 2 == 0:
            ax_list[loc_list.index(i)].set_ylabel(f'Number of {args.category}', fontsize=18)
            ax_list[loc_list.index(i)].legend(markerscale=3, fontsize=18)
    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


def corrotus(args):
    plot_df_dict = {
        i: (data_df_dict[i][args.category].value_counts(normalize=True) * 3000).rename_axis(args.category).reset_index(
            name=i).set_index(args.category) for i in data_df_dict}
    otu_category_df = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True),
                             list(plot_df_dict.values())).fillna(0).T
    try:
        del otu_category_df['unknown']
    except Exception as e:
        pass
    all_perm = list(permutations(list(otu_category_df), 2))

    name_list = list(otu_category_df.columns)
    corr = pd.DataFrame(columns=name_list, index=name_list)
    for perm in all_perm:
        corr_and_p_value = stats.pearsonr(otu_category_df[f'{perm[0]}'], otu_category_df[f'{perm[1]}'])
        if corr_and_p_value[1] < 0.1:
            corr.loc[perm[0], perm[1]] = corr_and_p_value[0]
    corr.fillna(np.nan, inplace=True)
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(60, 48))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=-360)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, annot=True, mask=mask, center=0, linewidths=.5, fmt='.2g')

    plt.savefig(fname=f'{args.output}.{args.format}', bbox_inches='tight')


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--input', type=str, required=True,
                               help='Absolute path or relative file.(ex: ./data.xlsx)')
    parent_parser.add_argument('--format', type=str, default='png', choices=['png', 'svg', 'pdf'],
                               help='Output format.(default: %(default)s)')
    parser = argparse.ArgumentParser(description='A comprehensive visual software for study taphonome.')
    parser.add_argument('-v', '--version', action='version', version='TaphonomeAnalyst 1.0')
    subparsers = parser.add_subparsers(help='commands')

    clusterenv_parser = subparsers.add_parser(name='clusterenv',
                                              parents=[parent_parser],
                                              help='Hierarchical clustering-sedimentary environment.\t[clustermap]')
    clusterenv_parser.add_argument('--category', type=str, default='family',
                                   choices=['order', 'family', 'genera', 'species'],
                                   help='Taxonomic units.(default: %(default)s)')
    clusterenv_parser.add_argument('--aquatic', type=str, default='', nargs='+',
                                   help='Aquatic animals.(default: all animals)\t[e.g. animal1 animal2 animal3]')
    clusterenv_parser.add_argument('--output', type=str, default='./clusterenv',
                                   help='Absolute path or relative path and filename.(default: %(default)s)')
    clusterenv_parser.set_defaults(func=clusterenv)

    divloc_parser = subparsers.add_parser(name='divloc', parents=[parent_parser],
                                          help='Venn diagram-sampling locations.\t[venn]')
    divloc_parser.add_argument('--category', type=str, default='family',
                               choices=['order', 'family', 'genera', 'species'],
                               help='Taxonomic units.(default: %(default)s)')
    divloc_parser.add_argument('--output', type=str, default='./divloc',
                               help='Absolute path or relative path and filename.(default: %(default)s)')
    divloc_parser.set_defaults(func=divloc)

    divenv_parser = subparsers.add_parser(name='divenv', parents=[parent_parser],
                                          help='Venn diagram-Taphonomic environments.\t[venn]')
    divenv_parser.add_argument('--category', type=str, default='family',
                               choices=['order', 'family', 'genera', 'species'],
                               help='Taxonomic units.(default: %(default)s)')
    divenv_parser.add_argument('--groups', type=str, required=True, nargs='+',
                               help='Environment groups.\t[e.g. locA1/locB2 locB1/locA2 locC1/locC2]')
    divenv_parser.add_argument('--output', type=str, default='./divenv',
                               help='Absolute path or relative path and filename.(default: %(default)s)')
    divenv_parser.set_defaults(func=divenv)

    TGotus_parser = subparsers.add_parser(name='TGotus', parents=[parent_parser],
                                          help='Taphonomic grades-taxa.\t[barh]')
    TGotus_parser.add_argument('--category', type=str, default='order',
                               choices=['order', 'family', 'genera', 'species'],
                               help='Taxonomic units.(default: %(default)s)')
    TGotus_parser.add_argument('--output', type=str, default='./TGotus',
                               help='Absolute path or relative path and filename.(default: %(default)s)')
    TGotus_parser.set_defaults(func=TGotus)

    TGplots_parser = subparsers.add_parser(name='TGplots', parents=[parent_parser],
                                           help='Taphonomic grades-sampling plots.\t[barh]')
    TGplots_parser.add_argument('--output', type=str, default='./TGplots',
                                help='Absolute path or relative path and filename.(default: %(default)s)')
    TGplots_parser.set_defaults(func=TGplots)

    abundplots_parser = subparsers.add_parser(name='abundplots', parents=[parent_parser],
                                              help='Abundance-sampling plots.\t[barh]')
    abundplots_parser.add_argument('--category', type=str, default='order',
                                   choices=['order', 'family', 'genera', 'species'],
                                   help='Taxonomic units.(default: %(default)s)')
    abundplots_parser.add_argument('--output', type=str, default='./abundplots',
                                   help='Absolute path or relative path and filename.(default: %(default)s)')
    abundplots_parser.set_defaults(func=abundplots)

    cooccurnet_parser = subparsers.add_parser(name='cooccurnet', parents=[parent_parser],
                                              help='Co-occurrence networks.\t[network]')
    cooccurnet_parser.add_argument('--category', type=str, default='family',
                                   choices=['order', 'family', 'genera', 'species'],
                                   help='Taxonomic units.(default: %(default)s)')
    cooccurnet_parser.add_argument('--corr', type=str, default='pearson', choices=['pearson', 'spearman', 'kendall'],
                                   help='Correlation algorithm.(default: %(default)s)')
    cooccurnet_parser.add_argument('--corr_coef', type=float, default=0.0,
                                   help='Minimum threshold of correlation coefficient.(default: %(default)s)')
    cooccurnet_parser.add_argument('--p_value', type=float, default=0.1,
                                   help='Maximum threshold of p-value.(default: %(default)s)')
    cooccurnet_parser.add_argument('--output', type=str, default='./cooccurnet',
                                   help='Absolute path or relative path and filename.(default: %(default)s)')
    cooccurnet_parser.set_defaults(func=cooccurnet)

    samplecurve_parser = subparsers.add_parser(name='samplecurve', parents=[parent_parser],
                                               help='Sampling coverage curve.\t[regplot]')
    samplecurve_parser.add_argument('--category', type=str, default='family',
                                    choices=['order', 'family', 'genera', 'species'],
                                    help='Taxonomic units.(default: %(default)s)')
    samplecurve_parser.add_argument('--output', type=str, default='./samplecurve',
                                    help='Absolute path or relative path and filename.(default: %(default)s)')
    samplecurve_parser.set_defaults(func=samplecurve)

    chao_parser = subparsers.add_parser(name='chao', parents=[parent_parser],
                                        help='Chao1 potential diversity curve.\t[regplot]')
    chao_parser.add_argument('--category', type=str, default='family',
                             choices=['order', 'family', 'genera', 'species'],
                             help='Taxonomic units.(default: %(default)s)')
    chao_parser.add_argument('--output', type=str, default='./chao',
                             help='Absolute path or relative path and filename.(default: %(default)s)')
    chao_parser.set_defaults(func=chao)

    ace_parser = subparsers.add_parser(name='ace', parents=[parent_parser],
                                       help='ACE potential diversity curve.\t[regplot]')
    ace_parser.add_argument('--category', type=str, default='family', choices=['order', 'family', 'genera', 'species'],
                            help='Taxonomic units.(default: %(default)s)')
    ace_parser.add_argument('--output', type=str, default='./ace',
                            help='Absolute path or relative path and filename.(default: %(default)s)')
    ace_parser.add_argument('--rare', type=int, default=10, help='ACE rare threshold.(default: %(default)s)')
    ace_parser.set_defaults(func=ace)

    corrotus_parser = subparsers.add_parser(name='corrotus', parents=[parent_parser],
                                            help='Heatmap-OTUs correlation analysis.\t[heatmap]')
    corrotus_parser.add_argument('--category', type=str, default='family',
                                 choices=['order', 'family', 'genera', 'species'],
                                 help='Taxonomic units.(default: %(default)s)')
    corrotus_parser.add_argument('--output', type=str, default='./corrotus',
                                 help='Absolute path or relative path and filename.(default: %(default)s)')
    corrotus_parser.set_defaults(func=corrotus)

    args = parser.parse_args()

    data_file = args.input
    data_xls = pd.ExcelFile(data_file)
    all_sheet_name_list = [i for i in data_xls.sheet_names if not i.startswith('Sheet')]
    data_df_dict = dict()
    for i in all_sheet_name_list:
        data_df_dict[i] = pd.read_excel(data_xls, sheet_name=i, header=0,
                                        usecols=['order', 'family', 'genera', 'species', 'taphonomic grade'],
                                        nrows=3000)
        if data_df_dict[i].shape[0] < 3000:
            data_df_dict[i] = pd.concat([data_df_dict[i]] * ceil(3000 / data_df_dict[i].shape[0]), ignore_index=True)
            data_df_dict[i] = data_df_dict[i][:3000]

    color_list = ['#e64b35', '#e18727', '#ffdc91', '#0072b5', '#4dbbd5', '#00a087', '#925e9f', '#ee4c97', '#f39b7f',
                  '#7e6148',
                  '#19b4ca', '#298af0', '#00236e', '#ff8d4a', '#b2442a', '#ff5f78', '#6da160', '#11b368', '#0c6480',
                  '#819eb7',
                  '#fffc00', '#ff6c00', '#ff0000', '#9cff00', '#00ffd2', '#166700', '#bcadff', '#af1a5d', '#9c00ff',
                  '#9c9c9c',
                  '#1e00ff', '#a9ffea', '#7d6c4b', '#ffd409', '#502e03', '#e998ff', '#435200', '#50e5a2', '#bbff90',
                  '#000000']
    grade_list = ['A', 'B', 'C', 'D', 'E']

    args.func(args)
