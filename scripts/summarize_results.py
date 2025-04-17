import glob


def parse_results(folder):
    res = []
    for file in glob.glob(folder + "*.txt"):
        with open(file) as f:
            content = f.read()
            mean_idx = content.find("Mean: ")
            if mean_idx != -1:
                res.append(file.split('/')[-1] + ": " + content[mean_idx:mean_idx + len(": ") + 5 + len("Std dev: ") + 25].rstrip("\n"))
            else:
                res.append("Mean and std dev not found in: " + file)
    return res


def print_results():
    nopretrain_res = parse_results("results/outputs_nopretrain/")
    pretrain_res_fclayers = parse_results("results/outputs_pretrain_fclayers/")
    pretrain_res_fullmodel = parse_results("results/outputs_pretrain_fullmodel/")
    #pretrain_res_transformer_fclayers = parse_results("results/outputs_pretrain_transformer_fclayers/")

    if len(nopretrain_res) != len(pretrain_res_fclayers):
        print("Error: nopretrain and pretrain number of subjects differ!")

    for (el1, el2, el3) in zip(nopretrain_res, pretrain_res_fullmodel, pretrain_res_fclayers):
        print(el1, "       ", el2, "        ", el3,)


print_results()