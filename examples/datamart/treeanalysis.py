import polars as pl
import os
import json
from pdstools import ADMDatamart
from collections import OrderedDict

folder_path = "/Users/hasss/dev/bol.com/modelSnapshots/"
file_name = "Data-Decision-ADM-ModelSnapshot_AdmModelsSnapshot_20230308T113755_GMT/data.json"

plots_folder_path = "/Users/hasss/dev/pega-datascientist-tools/output"

past_n_snapshots = 2
compare_first_n_trees = 2

node_left_child = "left_child"
node_right_child = "right_child"
split = "split"
flag = "flag"
depth = "depth"

# colors
flag_identical = "white"
flag_changed = "darkorchid"
flag_leaf = "azure2"

flag_split_to_prune = "brown1"
flag_split_to_split = "crimson"
flag_prune_to_split = "cornflowerblue"


def do_validate_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_to_file(info_dump, path_to_write, tree_name):
    reformat_path = path_to_write.split("/")[:-1]
    reformat_filename = f'{reformat_path[-1]}_{tree_name}_info.json'
    reformat_path = f'{"/".join(reformat_path)}/{reformat_filename}'
    with open(reformat_path, 'w') as wr:
        wr.write(json.dumps(info_dump))
    print(f'Write to file: {reformat_path}')


def check_if_identical(tree1, tree2, idx1, idx2):
    if split in tree1[idx1] and split in tree2[idx2]:
        if tree1[idx1][split] == tree2[idx2][split]:
            return True
    return False


def check_prune_to_split(tree1, tree2, idx1, idx2):
    if node_left_child not in tree1[idx1] and node_right_child not in tree1[idx1]:
        if node_left_child in tree2[idx2] or node_right_child in tree2[idx2]:
            return True
    return False


def check_split_to_split(tree1, tree2, idx1, idx2):
    if node_left_child in tree1[idx1] or node_right_child in tree1[idx1]:
        if node_left_child in tree2[idx2] or node_right_child in tree2[idx2]:
            return True
    return False


def check_split_to_prune(tree1, tree2, idx1, idx2):
    if node_left_child in tree1[idx1] or node_right_child in tree1[idx1]:
        if node_left_child not in tree2[idx2] and node_right_child not in tree2[idx2]:
            return True
    return False


def add_to_info(key, info_dump, data):
    if key in info_dump:
        info_dump[key].append(data)
    else:
        info_dump[key] = [data]

    return info_dump


def can_we_highlight_recurve(node, key, color, depth_level):
    node[key][flag] = color
    node[key][depth] = depth_level

    if node_left_child in node[key]:
        can_we_highlight_recurve(node, node[key][node_left_child], color, depth_level+1)

    if node_right_child in node[key]:
        can_we_highlight_recurve(node, node[key][node_right_child], color, depth_level+1)

    return node, key


def can_we_recurve(node1, node2, key1, key2, info_dump, depth_level1, depth_level2):

    node1[key1][depth] = depth_level1
    node2[key2][depth] = depth_level2

    if node_left_child not in node1[key1] and node_right_child not in node1[key1]:
        if check_prune_to_split(node1, node2, key1, key2):
            info_dump = add_to_info("prune_to_split", info_dump, {"node1": node1[key1], "node2": node2[key2]})
            node1, key1 = can_we_highlight_recurve(node1, key1, flag_prune_to_split, depth_level1)
            node2, key2 = can_we_highlight_recurve(node2, key2, flag_prune_to_split, depth_level2)
            return node1, node2, key1, key2, info_dump
        else:
            node1[key1][flag] = flag_leaf
            node2[key2][flag] = flag_leaf
    else:
        if check_if_identical(node1, node2, key1, key2):
            node1[key1][flag] = flag_identical
            node2[key2][flag] = flag_identical
        elif check_split_to_split(node1, node2, key1, key2):
            info_dump = add_to_info("split_to_split", info_dump, {"node1": node1[key1], "node2": node2[key2]})
            node1, key1 = can_we_highlight_recurve(node1, key1, flag_split_to_split, depth_level1)
            node2, key2 = can_we_highlight_recurve(node2, key2, flag_split_to_split, depth_level2)
            return node1, node2, key1, key2, info_dump
        elif check_split_to_prune(node1, node2, key1, key2):
            info_dump = add_to_info("split_to_prune", info_dump, {"node1": node1[key1], "node2": node2[key2]})
            node1, key1 = can_we_highlight_recurve(node1, key1, flag_split_to_prune, depth_level1)
            node2, key2 = can_we_highlight_recurve(node2, key2, flag_split_to_prune, depth_level2)
            return node1, node2, key1, key2, info_dump
        else:
            node1[key1][flag] = flag_changed
            node2[key2][flag] = flag_changed

    # do for left
    node1_left_child_key = node1[key1][node_left_child] if node_left_child in node1[key1] else None
    node2_left_child_key = node2[key2][node_left_child] if node_left_child in node2[key2] else None
    if node1_left_child_key is not None and node2_left_child_key is not None:
        can_we_recurve(node1, node2, node1_left_child_key, node2_left_child_key, info_dump, depth_level1+1, depth_level2+1)
    else:
        node1[key1][flag] = flag_leaf
        node2[key2][flag] = flag_leaf

    # do for right
    node1_right_child_key = node1[key1][node_right_child] if node_right_child in node1[key1] else None
    node2_right_child_key = node2[key2][node_right_child] if node_right_child in node2[key2] else None
    if node1_right_child_key is not None and node2_right_child_key is not None:
        can_we_recurve(node1, node2, node1_right_child_key, node2_right_child_key, info_dump, depth_level1+1, depth_level2+1)
    else:
        node1[key1][flag] = flag_leaf
        node2[key2][flag] = flag_leaf

    return node1, node2, info_dump


def do_compare_trees(snapshot1, snapshot2, plots_file_name1, plots_file_name2):
    for i in range(1, compare_first_n_trees + 1):
        tree_name = f'tree{i}'
        print(f'Comparing {tree_name} and {tree_name}')

        snap1_tree = snapshot1[tree_name]
        snap2_tree = snapshot2[tree_name]

        node1, node2, info_dump = can_we_recurve(snap1_tree, snap2_tree, 1, 1, {}, 1, 1)

        snapshot1["trees"].plotTreeWithAddnlInfo(node1, flag, "test1", show=False).write_png(
            f'{plots_file_name1}_{tree_name}.png')
        snapshot2["trees"].plotTreeWithAddnlInfo(node2, flag, "test2", show=False).write_png(
            f'{plots_file_name2}_{tree_name}.png')

        write_to_file(info_dump, plots_file_name1, tree_name)
        print(f'Processed {tree_name} and {tree_name}')


def do_compare_snapshots(in_comparison):

    for channel, snapshots in in_comparison.items():
        print(f'Processing for channel: {channel}')
        plots_folder_path_channel = f'{plots_folder_path}/{channel}'
        do_validate_folder(plots_folder_path_channel)

        s_keys = list(snapshots)
        for i in range(0, len(snapshots)-1):
            snapshot_time1 = s_keys[i]
            snapshot_time2 = s_keys[i+1]
            print(f'Comparing snapshots {snapshot_time1} and {snapshot_time2}')

            snap1 = snapshots[snapshot_time1]
            snap2 = snapshots[snapshot_time2]

            snapshot_date1 = snapshot_time1.split(" ")[0]
            snapshot_date2 = snapshot_time2.split(" ")[0]

            plots_folder_path_channel_snaps = f'{plots_folder_path_channel}/{snapshot_date1.replace("-", "")}_to_{snapshot_date2.replace("-", "")}'
            do_validate_folder(plots_folder_path_channel_snaps)

            make_filename1 = f'{plots_folder_path_channel_snaps}/{snapshot_date1}'
            make_filename2 = f'{plots_folder_path_channel_snaps}/{snapshot_date2}'

            do_compare_trees(snap1, snap2, make_filename1, make_filename2)


def main():

    do_validate_folder(plots_folder_path)

    dm = ADMDatamart(folder_path, model_filename=file_name, predictor_filename=None, include_cols=["Modeldata"])

    multi_trees_obj = dm.get_AGB_models()

    comparison_dict = OrderedDict()
    for channel, properties in multi_trees_obj.items():
        trees_dict = OrderedDict(sorted(properties.trees.items()))
        past_n_snapshots_trees = dict(list(trees_dict.items())[-past_n_snapshots:])
        for snapshot_time, trees in past_n_snapshots_trees.items():
            compare_first_n_trees_list = [x for x in range(len(trees.treeStats))[:compare_first_n_trees]]
            inner_tree = OrderedDict({f'tree{x+1}': trees.getTreeRepresentation(x) for x in compare_first_n_trees_list})
            inner_tree["trees"] = trees
            if channel in comparison_dict:
                comparison_dict[channel][snapshot_time] = inner_tree
            else:
                comparison_dict[channel] = {snapshot_time: inner_tree}

    do_compare_snapshots(comparison_dict)

    # snapshots_list = modelData.select(
    #     pl.col("SnapshotTime").sort_by("SnapshotTime", descending=True)
    # ).unique().limit(past_n_snapshots).collect().rows()
    #
    # model_snapshots = []
    # for snapshot in snapshots_list:
    #     model_snapshots.append(modelData.filter(pl.col("SnapshotTime").is_in(snapshot)).collect())
    # modelData.filter(pl.col("SnapshotTime").is_in(snapshots_list))


if __name__ == "__main__":
    main()

