# STUN
We provide scripts that we used for our experiments.

We also provide our optimized version of OWL and wanda that prunes 480B model even only one 80GB GPU.

## STUN with snowflake-arctic
```bash
python utils/merge_weight.py --model_path snowflake-arctic --output_dir=new_arctic_snowflake_0.5_127_26_3570_nodivnorm_greedy_but_reject_mixed_load_max2 --layer_num 35 --gate_template "model.layers.{}.block_sparse_moe.gate.weight" --threshold=0.5 --top_k=127 --division_by_norm=False --merge_method=greedy_but_reject_mixed --snake_case_model_name=arctic --expert_num_key_in_config=num_local_experts --expert_template=model.layers.{}.block_sparse_moe.experts.{}.w1.weight,model.layers.{}.block_sparse_moe.experts.{}.w2.weight,model.layers.{}.block_sparse_moe.experts.{}.w3.weight --router_logits_file=arctic.txt  --load_path=arctic.pt --divide_mean  --merge_max_clusters=2  --binary_search_target=3570 
```
Then run OWL or wanda.