import collections
import glob
import sys, torch, os
import json
path = sys.argv[1]

write_dict = {}
with open(sys.argv[2] + '.txt', 'w') as f_w:
    for f in glob.glob(os.path.join(path, f"router_logits_*.pt")):
        router_logits = torch.load(f, map_location='cpu')

        for layer_id, router_logit in enumerate(router_logits):
            router_softmax = torch.nn.functional.softmax(router_logit, dim=1)
            router_topk = torch.topk(router_softmax, 2, dim=1)
            router_topk_weights = router_topk.values
            router_topk = router_topk.indices
            count = collections.defaultdict(int)
            for r_topk in router_topk:
                for i in r_topk:
                    count[i.item()] += 1
            write_dict[f"router_topk_{layer_id}"] = count
        f_w.write(json.dumps(write_dict) + '\n')


layerid2sum = {}
for f in glob.glob(os.path.join(path, f"router_logits_*.pt")):
    router_logits = torch.load(f)

    for layerid, logits in enumerate(router_logits):
        num_tokens, num_experts = logits.shape
        logits = torch.softmax(logits, 1)
        _, topk_indices = torch.topk(logits, 2, dim=1)

        zeros = torch.zeros(num_tokens, num_experts, num_experts, device=logits.device)

        r = torch.arange(num_tokens, device=logits.device)
        zeros[r, topk_indices[:, 0], topk_indices[:, 1]] = 1
        zeros[r, topk_indices[:, 1], topk_indices[:, 0]] = 1


        if layerid not in layerid2sum:
            layerid2sum[layerid] = zeros.sum(0)
        else:
            layerid2sum[layerid] += zeros.sum(0)

torch.save(layerid2sum, sys.argv[2] + '.pt')

