import glob


def parse_results(folder):
    res = []
    for file in glob.glob(folder + "*.txt"):
        with open(file) as f:
            content = f.read()
            acc_idx = content.find("Test Accuracy: ")
            if acc_idx != -1:
                res.append(file + ": " + content[acc_idx:acc_idx + len("Test Accuracy: ") + 5].rstrip("\n"))
            else:
                res.append("Test accuracy not found in: " + file)
    return res


def print_results():
    nopretrain_res = parse_results("outputs_nopretrain/")
    pretrain_res = parse_results("outputs_pretrain/")

    if len(nopretrain_res) != len(pretrain_res):
        print("Error: nopretrain and pretrain number of subjects differ!")

    for (el1, el2) in zip(nopretrain_res, pretrain_res):
        print(el1, "       ", el2)


print_results()

