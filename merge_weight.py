import os.path
import sys

import torch
import shutil

from transformers import HfArgumentParser, AutoConfig, AutoTokenizer
from safetensors import safe_open
import glob
from dataclasses import dataclass, field


@dataclass
class Args:
    model_path: str = field(default="Mixtral-8x7B-Instruct-v0.1")
    output_dir: str = field(default="Mixtral-7x7B")
    gate_template: str = field(default="model.layers.{}.block_sparse_moe.gate.weight")
    expert_template: str = field(
        default="model.layers.{}.block_sparse_moe.experts.{}.w1.weight,model.layers.{}.block_sparse_moe.experts.{}.w2.weight,model.layers.{}.block_sparse_moe.experts.{}.w3.weight")
    layer_num: int = field(default=32)
    top_k: int = field(default=1)
    threshold: float = field(default=float('inf'))

    expert_num_key_in_config: str = field(default='num_local_experts')
    snake_case_model_name: str = field(default='mixtral')
    division_by_norm: bool = field(default=True)
    suffix: str = field(default='')
    merge_method: str = field(default='greedy_but_reject_mixed', metadata={'choices': [
                                                                    'greedy_but_reject_mixed',
                                                                    ]})
    mix_weight: float = field(default=None)
    merge_max_clusters: int = field(default=0)
    router_logits_file: str = field(default="router_logits.txt")
    load_path: str = field(default='')
    binary_search_target: int = field(default=None)
    divide_mean: bool = field(default=False)
    top_k_for_loss: int = field(default=4)




def get_state_dict(args: Args):
    state_dict = {}
    for f in glob.glob(os.path.join(args.model_path, "*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as f_r:
            for key in f_r.keys():
                state_dict[key] = f_r.get_tensor(key)
    return state_dict



def select_expert_pairs(gate_weight, topk=1, threshold=float('inf'), division_by_norm=True,
                        merge_method='', load_path="", layer_id=None, mix_weight=None,
                        divide_mean=False):
    if load_path:
        l2_dist = torch.load(load_path, map_location='cpu')
        l2_dist = -l2_dist[layer_id]

        if mix_weight:
            l2_dist /= l2_dist.mean().abs()
            # print(l2_dist)
            l2_dist = l2_dist * mix_weight
            l2_dist += torch.cdist(gate_weight.float(), gate_weight.float())
            # print(torch.cdist(gate_weight.float(), gate_weight.float()))

        elif divide_mean:
            l2_dist /= l2_dist.mean().abs()


    else:
        l2_dist = torch.cdist(gate_weight.float(), gate_weight.float())
        if division_by_norm:
            l2_dist /= torch.norm(gate_weight)

    values = l2_dist
    for i in range(len(gate_weight)):
        values[i][i] = float('inf')
    vertex_to_cluster = {i: {i} for i in range(len(gate_weight))}

    while torch.min(values) < threshold:
        num_clusters = len(set([tuple(sorted(list(s))) for s in vertex_to_cluster.values()]))
        if num_clusters <= len(gate_weight) - topk:
            break

        min_indices = torch.argmin(values.flatten())
        min_indices = torch.unravel_index(min_indices, values.shape)

        lengths = []
        v1, v2 = min_indices
        v1 = int(v1)
        v2 = int(v2)

        # need to check whether v1 has long path to v2 clusters
        for cluster in vertex_to_cluster[v1]:
            lengths.append(values[cluster, v2])
        for cluster in vertex_to_cluster[v2]:
            lengths.append(values[cluster, v1])

        if (max(lengths) < threshold) or (merge_method == 'greedy_mixed'):
            union_set = vertex_to_cluster[v1] | vertex_to_cluster[v2]
            for cluster in vertex_to_cluster[v1]:
                values[cluster, v2] = float('inf')
            for cluster in vertex_to_cluster[v2]:
                values[cluster, v1] = float('inf')
            for vertex in union_set:
                vertex_to_cluster[vertex] = union_set

        else:
            values[v1, v2] = float('inf')
            values[v2, v1] = float('inf')
            continue

    return list(set([tuple(sorted(list(s))) for s in vertex_to_cluster.values()]))





def dsatur_merge_gates(gate_weight, gate_bias, expert_pairs, merge_method='mean',
                       selected_expert_ids=[]):
    if len(expert_pairs) <= args.merge_max_clusters:
        if len(expert_pairs) == 0:
            return gate_weight, gate_bias
        gate_weights = []
        gate_biases = []
        for expert_pair in expert_pairs:
            if len(expert_pair) == 1:
                gate_weights.append(gate_weight[expert_pair[0]])
                gate_biases.append(gate_bias[expert_pair[0]])
            else:
                gate_weights.append(gate_weight[list(expert_pair)].mean(dim=0))
                gate_biases.append(torch.zeros([]))

        new_gate_weight = torch.stack(gate_weights)
        new_gate_bias = torch.stack(gate_biases)
        return new_gate_weight, new_gate_bias


    else:
        if len(expert_pairs) == 0:
            return gate_weight, gate_bias

        gate_weights = []
        gate_biases = []
        for expert_pair_id, expert_pair in enumerate(expert_pairs):
            if merge_method == 'greedy_but_reject_real_mixed' and selected_expert_ids[expert_pair_id] < 1:
                gate_weights.append(gate_weight[list(expert_pair)].mean(dim=0))
                gate_biases.append(torch.zeros([]))
            else:
                gate_weights.append(gate_weight[expert_pair[selected_expert_ids[expert_pair_id]]])
                gate_biases.append(gate_bias[expert_pair[selected_expert_ids[expert_pair_id]]])
        new_gate_weight = torch.stack(gate_weights)
        new_gate_bias = torch.stack(gate_biases)
        return new_gate_weight, new_gate_bias


def dsatur_merge_experts(expert_weights: list, expert_pairs, merge_method='mean'):
    return_list = []
    if len(expert_pairs) == 0:
        return expert_weights

    expert_weights = torch.stack(expert_weights)
    for expert_pair in expert_pairs:
        if len(expert_pair) == 1:
            return_list.append(expert_weights[expert_pair[0]])
        else:
            sub_weights = [expert_weights[expert_id] for expert_id in expert_pair]
            return_list.append(sum(sub_weights) / len(sub_weights))

    return return_list

def dsatur_select_mean_weight_merge(expert_weight_pairs: list, expert_pairs, merge_method, select_max_cluster_size):
    selected_expert_indices = []
    return_list = []
    for expert_pair in expert_pairs:
        if len(expert_pair) == 1:
            return_list.append(expert_weight_pairs[expert_pair[0]])
            selected_expert_indices.append(0)
        else:
            sub_weights = [torch.cat([t.flatten() for t in expert_weight_pairs[expert_id]]) for expert_id in expert_pair]
            centroid = sum(sub_weights) / len(sub_weights)
            closest_idx = torch.argmin(torch.stack([torch.norm(w - centroid) for w in sub_weights]))
            return_list.append(expert_weight_pairs[expert_pair[closest_idx]])
            selected_expert_indices.append(closest_idx)

    return selected_expert_indices, return_list



def merge_and_get_numexperts(args):
    state_dict = get_state_dict(args)

    starmap_args = []
    for layer_id in range(args.layer_num):
        gate_weight = state_dict[args.gate_template.format(layer_id)]
        starmap_args.append((gate_weight, layer_id, args))
    from multiprocessing import Pool
    results = Pool(args.layer_num).starmap(get_layer_expert_nums, starmap_args)

    return results

def get_layer_expert_nums(gate_weight, layer_id, args):
    #### get expert_pairs
    expert_pairs_list = select_expert_pairs(gate_weight, args.top_k, args.threshold, args.division_by_norm,
                                            args.merge_method, args.load_path, layer_id, args.mix_weight,
                                            args.divide_mean)
    # print(expert_pairs_list)

    local_expert_num = len(expert_pairs_list)

    return local_expert_num

def merge_and_get_statedict_numexperts(args):
    state_dict = get_state_dict(args)

    num_local_experts = []

    for layer_id in range(args.layer_num):
        gate_weight = state_dict[args.gate_template.format(layer_id)]
        gate_bias = torch.zeros(gate_weight.shape[0]).to(gate_weight)

        num_experts = getattr(config, args.expert_num_key_in_config)

        #### get expert_pairs
        expert_pairs_list = select_expert_pairs(gate_weight, args.top_k, args.threshold, args.division_by_norm,
                                                args.merge_method, args.load_path, layer_id, args.mix_weight,
                                                args.divide_mean)
        # print(expert_pairs_list)


        #### merging experts
        selected_expert_ids = []
        if len(expert_pairs_list) > args.merge_max_clusters:
            expert_weight_pairs = []
            for expert_id in range(num_experts):
                expert_weights = []
                for template in args.expert_template.split(','):
                    weight_name = template.format(layer_id, expert_id)
                    if weight_name in state_dict:
                        expert_weights.append(state_dict.pop(weight_name))
                expert_weight_pairs.append(expert_weights)

            selected_expert_ids, expert_weight_pairs = dsatur_select_mean_weight_merge(expert_weight_pairs,
                                                                                       expert_pairs_list, args.merge_method, args.merge_max_clusters)

            for expert_id, expert_weight_pair in enumerate(expert_weight_pairs):
                for pair_id, template in enumerate(args.expert_template.split(',')):
                    state_dict[template.format(layer_id, expert_id)] = expert_weight_pair[pair_id]

        else: # do it by specific layer-by-layer
            for template in args.expert_template.split(','):
                expert_weights = []
                for expert_id in range(num_experts):
                    weight_name = template.format(layer_id, expert_id)
                    if weight_name in state_dict:
                        expert_weights.append(state_dict.pop(weight_name))

                expert_weights = dsatur_merge_experts(expert_weights, expert_pairs_list, args.merge_method)

                for expert_id, expert_weight in enumerate(expert_weights):
                    state_dict[template.format(layer_id, expert_id)] = expert_weight

        ##### merging gates
        gate_weight, gate_bias = dsatur_merge_gates(gate_weight, gate_bias, expert_pairs_list, args.merge_method,
                                                    selected_expert_ids)


        local_expert_num = len(expert_pairs_list)

        num_local_experts.append(local_expert_num)

        state_dict[args.gate_template.format(layer_id)] = gate_weight
        state_dict[args.gate_template.format(layer_id).replace('weight', 'bias')] = gate_bias

    return state_dict, num_local_experts


if __name__ == '__main__':
    parser = HfArgumentParser((Args))
    args = parser.parse_args_into_dataclasses()[0]
    assert isinstance(args, Args)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if not args.binary_search_target:
        state_dict, num_local_experts = merge_and_get_statedict_numexperts(args)
    else:
        step = 0.1
        last_direction = 0
        never_changed = True
        state_dict = None
        experts = []
        while True:
            print(f"{args.output_dir} {experts} TRIAL: ", args.threshold)
            num_local_experts = merge_and_get_numexperts(args)

            if sum(num_local_experts) < args.binary_search_target:
                # need to decrease threshold to make remaining experts more
                current_direction = -1
            elif sum(num_local_experts) > args.binary_search_target:
                # need to increase threshold to make remaining experts less
                current_direction = 1
            else:
                print('FOUND: ', args.threshold)
                break

            experts.append(sum(num_local_experts))
            if len(experts) > 10 and min(experts[-10:]) == max(experts[-10:]):
                sys.exit(1)

            if last_direction * current_direction < 0:
                never_changed = False

            if never_changed:
                step = step * 2
            else:
                step = step / 2
            print("WAS: ", sum(num_local_experts))
            args.threshold += step * current_direction
            last_direction = current_direction

        if state_dict is None:
            state_dict, num_local_experts = merge_and_get_statedict_numexperts(args)

    setattr(config, args.expert_num_key_in_config, num_local_experts)
    print('total experts : ', sum(num_local_experts))

    pascal_case_model_name = ''.join(word.title() for word in args.snake_case_model_name.split('_'))
    config.auto_map = {
        "AutoConfig": f"configuration_{args.snake_case_model_name}.{pascal_case_model_name}Config",
        "AutoModel": f"modeling_{args.snake_case_model_name}.{pascal_case_model_name}Model",
        "AutoModelForCausalLM": f"modeling_{args.snake_case_model_name}.{pascal_case_model_name}ForCausalLM",
    }
    config.save_pretrained(args.output_dir)
    shutil.copy(f'utils/configuration_{args.snake_case_model_name}.py', args.output_dir)
    shutil.copy(f'utils/modeling_{args.snake_case_model_name}.py', args.output_dir)

    torch.save(state_dict, os.path.join(args.output_dir, 'pytorch_model.bin'))
    with open(os.path.join(args.output_dir, 'found'), 'w') as f:
        f.write(f"{args.threshold}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
