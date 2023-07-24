import pandas as pd
import polars as pl
import os
import json
from .ADMDatamart import ADMDatamart
from .ADMTrees import ADMTrees
from collections import OrderedDict

import pydot

import zlib
import base64


class TreeAnalysis:

    def __init__(self, past_n_snapshots=2, compare_first_n_trees=2, folder_path=None, model_filename=None,
                 plots_folder_path=None, verbose=False):
        self.verbose = verbose

        self.folder_path = folder_path if folder_path is not None else "../../data/"

        self.model_df = None

        # picking an existing model snapshot file from data folder for now
        if model_filename is None:
            import glob
            model_snapshot_file = glob.glob(f'{self.folder_path}Data-Decision-ADM-ModelSnapshot*.zip')[0]
            self.model_filename = model_snapshot_file.split("/")[-1]
        else:
            self.model_filename = model_filename

        self.plots_folder_path = plots_folder_path if plots_folder_path is not None else "../../output"
        self.do_validate_folder(self.plots_folder_path)

        self.past_n_snapshots = past_n_snapshots
        self.compare_first_n_trees = compare_first_n_trees

        self.node_left_child = "left_child"
        self.node_right_child = "right_child"
        self.split = "split"

        # new properties
        self.flag = "flag"
        self.depth = "depth"

        # colors
        self.flag_identical = "white"
        self.flag_changed = "darkorchid"
        self.flag_leaf = "azure2"

        self.flag_split_to_prune = "crimson"
        self.flag_split_to_split = "goldenrod"
        self.flag_prune_to_split = "cornflowerblue"

        self.df = None

    @staticmethod
    def do_validate_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def write_to_file(self, info_dump, path_to_write, tree_name):
        reformat_path = path_to_write.split("/")[:-1]
        reformat_filename = f'{reformat_path[-1]}_{tree_name}_info.json'
        reformat_path = f'{"/".join(reformat_path)}/{reformat_filename}'
        with open(reformat_path, 'w') as wr:
            wr.write(json.dumps(info_dump))
        if self.verbose: print(f'Write to file: {reformat_path}')

    def check_if_identical(self, tree1, tree2, idx1, idx2):
        if self.split in tree1[idx1] and self.split in tree2[idx2]:
            split1 = tree1[idx1][self.split]
            split2 = tree2[idx2][self.split]
            separators = ["<", " in "]

            c1, c2 = 'a', 'b'

            if any(x in split1 for x in separators):
                c1 = split1.split(" ")[0].strip()

            if any(x in split2 for x in separators):
                c2 = split2.split(" ")[0].strip()

            if c1 == c2:
                return True

            if split1 == split2:
                return True

        return False

    def check_prune_to_split(self, tree1, tree2, idx1, idx2):
        if self.node_left_child not in tree1[idx1] and self.node_right_child not in tree1[idx1]:
            if self.node_left_child in tree2[idx2] or self.node_right_child in tree2[idx2]:
                return True
        return False

    def check_split_to_split(self, tree1, tree2, idx1, idx2):
        if self.node_left_child in tree1[idx1] or self.node_right_child in tree1[idx1]:
            if self.node_left_child in tree2[idx2] or self.node_right_child in tree2[idx2]:
                return True
        return False

    def check_split_to_prune(self, tree1, tree2, idx1, idx2):
        if self.node_left_child in tree1[idx1] or self.node_right_child in tree1[idx1]:
            if self.node_left_child not in tree2[idx2] and self.node_right_child not in tree2[idx2]:
                return True
        return False

    @staticmethod
    def add_to_info(key, info_dump, data):
        if key in info_dump:
            info_dump[key].append(data)
        else:
            info_dump[key] = [data]

        return info_dump

    def can_we_highlight_recurve(self, node, key, color, depth_level):
        node[key][self.flag] = color
        node[key][self.depth] = depth_level

        if self.node_left_child in node[key]:
            self.can_we_highlight_recurve(node, node[key][self.node_left_child], color, depth_level + 1)

        if self.node_right_child in node[key]:
            self.can_we_highlight_recurve(node, node[key][self.node_right_child], color, depth_level + 1)

        return node, key

    def can_we_recurve(self, node1, node2, key1, key2, info_dump, depth_level1, depth_level2):

        node1[key1][self.depth] = depth_level1
        node2[key2][self.depth] = depth_level2

        # for leaf node
        if self.node_left_child not in node1[key1] and self.node_right_child not in node1[key1]:
            if self.check_prune_to_split(node1, node2, key1, key2):
                node1, key1 = self.can_we_highlight_recurve(node1, key1, self.flag_prune_to_split, depth_level1)
                node2, key2 = self.can_we_highlight_recurve(node2, key2, self.flag_prune_to_split, depth_level2)
                info_dump = self.add_to_info("prune_to_split", info_dump, {"node1": node1[key1], "node2": node2[key2]})
                return node1, node2, info_dump
            else:
                node1[key1][self.flag] = self.flag_leaf
                node2[key2][self.flag] = self.flag_leaf
        else:
            if self.check_if_identical(node1, node2, key1, key2):
                node1[key1][self.flag] = self.flag_identical
                node2[key2][self.flag] = self.flag_identical
                info_dump = self.add_to_info("identical", info_dump, {"node1": node1[key1], "node2": node2[key2]})
            elif self.check_split_to_split(node1, node2, key1, key2):
                node1, key1 = self.can_we_highlight_recurve(node1, key1, self.flag_split_to_split, depth_level1)
                node2, key2 = self.can_we_highlight_recurve(node2, key2, self.flag_split_to_split, depth_level2)
                info_dump = self.add_to_info("split_to_split", info_dump, {"node1": node1[key1], "node2": node2[key2]})
                return node1, node2, info_dump
            elif self.check_split_to_prune(node1, node2, key1, key2):
                node1, key1 = self.can_we_highlight_recurve(node1, key1, self.flag_split_to_prune, depth_level1)
                node2, key2 = self.can_we_highlight_recurve(node2, key2, self.flag_split_to_prune, depth_level2)
                info_dump = self.add_to_info("split_to_prune", info_dump, {"node1": node1[key1], "node2": node2[key2]})
                return node1, node2, info_dump
            else:
                node1[key1][self.flag] = self.flag_changed
                node2[key2][self.flag] = self.flag_changed

        # do for left
        node1_left_child_key = node1[key1].get(self.node_left_child, None)
        node2_left_child_key = node2[key2].get(self.node_left_child, None)
        if node1_left_child_key is not None and node2_left_child_key is not None:
            self.can_we_recurve(node1, node2, node1_left_child_key, node2_left_child_key, info_dump, depth_level1 + 1,
                                depth_level2 + 1)
        else:
            node1[key1][self.flag] = self.flag_leaf
            node2[key2][self.flag] = self.flag_leaf

        # do for right
        node1_right_child_key = node1[key1].get(self.node_right_child, None)
        node2_right_child_key = node2[key2].get(self.node_right_child, None)
        if node1_right_child_key is not None and node2_right_child_key is not None:
            self.can_we_recurve(node1, node2, node1_right_child_key, node2_right_child_key, info_dump, depth_level1 + 1,
                                depth_level2 + 1)
        else:
            node1[key1][self.flag] = self.flag_leaf
            node2[key2][self.flag] = self.flag_leaf

        return node1, node2, info_dump

    def do_compare_trees(self, snapshot1, snapshot2, plots_file_name1, plots_file_name2):
        for i in range(1, self.compare_first_n_trees + 1):
            tree_name = f'tree{i}'
            if self.verbose: print(f'Comparing {tree_name} and {tree_name}')

            snap1_tree = snapshot1[tree_name]
            snap2_tree = snapshot2[tree_name]

            node1, node2, info_dump = self.can_we_recurve(snap1_tree, snap2_tree, 1, 1, {}, 1, 1)

            snapshot1["trees"].plotTreeWithAddnlInfo(node1, self.flag, "test1", show=False).write_png(
                f'{plots_file_name1}_{tree_name}.png')
            snapshot2["trees"].plotTreeWithAddnlInfo(node2, self.flag, "test2", show=False).write_png(
                f'{plots_file_name2}_{tree_name}.png')

            self.write_to_file(info_dump, plots_file_name1, tree_name)
            if self.verbose: print(f'Processed {tree_name} and {tree_name}')

    def do_compare_snapshots(self, in_comparison):

        for channel, snapshots in in_comparison.items():
            print(f'Processing for channel: {channel}')

            plots_folder_path_channel = f'{self.plots_folder_path}/{channel}'
            self.do_validate_folder(plots_folder_path_channel)

            s_keys = list(snapshots)
            for i in range(0, len(snapshots) - 1):
                snapshot_time1 = s_keys[i]
                snapshot_time2 = s_keys[i + 1]
                print(f'Comparing snapshots {snapshot_time1} and {snapshot_time2}')

                snap1 = snapshots[snapshot_time1]
                snap2 = snapshots[snapshot_time2]

                snapshot_date1 = snapshot_time1.split(" ")[0]
                snapshot_date2 = snapshot_time2.split(" ")[0]

                plots_folder_path_channel_snaps = f'{plots_folder_path_channel}/{snapshot_date1.replace("-", "")}_to_{snapshot_date2.replace("-", "")}'
                self.do_validate_folder(plots_folder_path_channel_snaps)

                make_filename1 = f'{plots_folder_path_channel_snaps}/{snapshot_date1}'
                make_filename2 = f'{plots_folder_path_channel_snaps}/{snapshot_date2}'

                self.do_compare_trees(snap1, snap2, make_filename1, make_filename2)

    def process(self):

        dm = ADMDatamart(self.folder_path, model_filename=self.model_filename, predictor_filename=None,
                         include_cols=["Modeldata"])

        multi_trees_obj = dm.get_AGB_models()

        comparison_dict = OrderedDict()
        for channel, properties in multi_trees_obj.items():
            trees_dict = OrderedDict(sorted(properties.trees.items()))

            if isinstance(self.past_n_snapshots, str):
                self.past_n_snapshots = len(trees_dict)

            past_n_snapshots_trees = dict(list(trees_dict.items())[-self.past_n_snapshots:])
            for snapshot_time, trees in past_n_snapshots_trees.items():
                nb_trees_in_snapshot = len(trees.treeStats)

                if isinstance(self.compare_first_n_trees, str):
                    self.compare_first_n_trees = nb_trees_in_snapshot

                compare_first_n_trees_list = [x for x in range(nb_trees_in_snapshot)][:self.compare_first_n_trees]

                inner_tree = OrderedDict(
                    {f'tree{x + 1}': trees.getTreeRepresentation(x) for x in compare_first_n_trees_list})
                inner_tree["trees"] = trees

                if channel in comparison_dict:
                    comparison_dict[channel][snapshot_time] = inner_tree
                else:
                    comparison_dict[channel] = {snapshot_time: inner_tree}

        self.do_compare_snapshots(comparison_dict)

    def check_processed(self):
        rootdir = self.plots_folder_path
        self.do_validate_folder(rootdir)

        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d):
                return True
        return False

    def get_df(self):
        if not self.check_processed():
            self.process()

        rows = []
        properties = ["score", "parent_node", "gain", "split", "left_child", "right_child", "depth", "flag"]

        for subdir, dirs, files in os.walk(self.plots_folder_path):
            for file in files:
                if file.endswith(".json"):
                    filename = os.path.join(subdir, file)
                    split_filename = filename.split("/")
                    info = split_filename[-1].split("_")

                    channel = split_filename[-3]
                    snapshot1 = info[0]
                    snapshot2 = info[2]
                    tree_name = info[3]

                    with open(filename) as jf:
                        to_dict = json.load(jf)

                        for change_type, nodes in to_dict.items():
                            for node_diff in nodes:
                                node_keys = node_diff.keys()
                                for node_key in node_keys:
                                    node = node_diff[node_key]
                                    row_node = [channel, snapshot1, snapshot2, snapshot1, tree_name, change_type,
                                                node_key]
                                    predictor_node = node.get(self.split).split(" ")[0] if self.split in node else None
                                    row_node.append(predictor_node)
                                    for pp in properties:
                                        prop_value = node.get(pp, None)
                                        row_node.append(prop_value)
                                    rows.append(row_node)

        columns = ["channel", "snapshot_from", "snapshot_to", "snapshot", "tree_id", "change_type", "node", "predictor"]
        columns.extend(properties)
        df = pd.DataFrame(rows, columns=columns)
        self.df = pl.from_dataframe(df)
        return self.df


class MiniAdmTrees:
    def __init__(self, model_filename):

        with open(model_filename) as f:
            file = json.load(f)

        self.successRate = file['successRate']
        self.auc = file['auc']
        self.model = file['model']['booster']['trees']
        self.predictors = None

    def get_nodes_recurve(self, tree, nodes_list, node_id, parent_node_id, depth_level):
        curr_node_id = node_id

        node_info = {}
        for key, val in tree.items():
            if key not in ['right', 'left']:
                node_info[key] = val
        node_info['depth'] = depth_level

        nodes_list[curr_node_id] = node_info

        if parent_node_id is not None:
            nodes_list[curr_node_id]['parent_node'] = parent_node_id

        if 'left' in tree.keys():
            left_node_id = node_id + 1
            nodes_list[curr_node_id]['left_child'] = left_node_id
            nodes_list, node_id = self.get_nodes_recurve(tree['left'], nodes_list,
                                                         left_node_id, curr_node_id, depth_level+1)

        if 'right' in tree.keys():
            right_node_id = node_id + 1
            nodes_list[curr_node_id]['right_child'] = right_node_id
            nodes_list, node_id = self.get_nodes_recurve(tree['right'], nodes_list,
                                                         right_node_id, curr_node_id, depth_level+1)

        return nodes_list, node_id

    def get_tree_representation(self, tree_idx):
        tree = self.model[tree_idx]

        node_info, nb_nodes = self.get_nodes_recurve(tree, nodes_list=OrderedDict(),
                                                     node_id=1, parent_node_id=None, depth_level=1)

        return node_info
    
    def get_split_statistics(self, tree_idx=None):
        param_split = 'split'

        columns = ['tree_idx', 'node_id', 'score', 'gain', 'sampleCount',
                   'predictor', 'sign', 'values',
                   'depth', 'left_child', 'right_child']

        def get_rows(in_tree_idx):
            rows_ = []
            nodes_dict = self.get_tree_representation(in_tree_idx)
            for node_key, node in nodes_dict.items():
                if param_split in node:
                    predictor, sign, values = self.parse_split_values(node[param_split], decode=False)

                    rows_.append([
                        in_tree_idx,
                        node_key,
                        node.get('score', None),
                        node.get('gain', None),
                        node.get('sampleCount', None),
                        predictor,
                        sign,
                        values,
                        node.get('depth', None),
                        node.get('left_child', None),
                        node.get('right_child', None)
                    ])
            return rows_

        selected_indexes = []
        if tree_idx is None:
            selected_indexes.extend(range(len(self.model)))
        else:
            selected_indexes = [tree_idx]

        rows_list = []
        for idx in selected_indexes:
            rows = get_rows(idx)
            rows_list.extend(rows)

        df = pd.DataFrame(rows_list, columns=columns)
        return pl.from_dataframe(df)

    def get_predictors(self):
        self.predictors = self.get_split_statistics()["predictor"].unique()
        return self.predictors
            
    @staticmethod
    def parse_split_values(value, decode=True):

        items = value.split(" ")
        predictor = items[0]
        sign = items[1]
        values = items[2:]

        if decode:
            if values[0] == "{" and values[-1] == "}":
                strip_values = " ".join(values[1:-1]).split(",")
                values = [x.strip() for x in strip_values]
                values = set(values)
            elif len(values) == 1 and isinstance(values, list):
                values = values[0]
        else:
            values = " ".join(values)

        return predictor, sign, values

    def plot_tree(self, tree_idx, highlight=None, show=True):

        nodes = self.get_tree_representation(tree_idx)

        graph = pydot.Dot("my_graph", graph_type="graph", rankdir="BT")

        for key, node in nodes.items():
            color = "white"

            label = f"Score: {node['score']}"

            if "split" in node:

                split = node["split"]

                variable, sign, values = self.parse_split_values(split)

                color = highlight.get(variable, "white")

                if sign == "in":
                    if len(values) <= 3:
                        label_name = values
                    else:
                        label_name = (f"{list(values)[0:2]+['...']}")
                    label += f"\nSplit: {variable} in {label_name}\nGain: {node['gain']}"

                else:
                    label += f"\nSplit: {node['split']}\nGain: {node['gain']}"

                graph.add_node(
                    pydot.Node(
                        name=key,
                        label=label,
                        shape="box",
                        style="filled",
                        fillcolor=color,
                    )
                )
            else:
                graph.add_node(
                    pydot.Node(
                        name=key,
                        label=label,
                        shape="ellipse",
                        style="filled",
                        fillcolor=color,
                    )
                )
            if "parent_node" in node:
                graph.add_edge(pydot.Edge(key, node["parent_node"]))

        if show:
            try:
                from IPython.display import Image, display
            except:
                raise ValueError(
                    "IPython not installed, please install it using `pip install IPython`."
                )
            try:
                display(Image(graph.create_png()))  # pragma: no cover
            except FileNotFoundError as e:
                print(
                    "Dot/Graphviz not installed. Please install it to your machine.", e
                )
        return graph
