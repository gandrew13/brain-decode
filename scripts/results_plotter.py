import sys
import csv
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt



def parse_results_file(results_file: str) -> Tuple[np.array, np.array]:
    with open(results_file, mode ='r')as f:
        data = csv.reader(f)
        next(data)                      # skip first (header) line
        steps = np.array([])
        losses = np.array([])
        [(steps := np.append(steps, int(line[0])), losses := np.append(losses, float(line[1]))) for line in data]
    return steps, losses
    


def plot(train_results_file: str, eval_resuls_file: str) -> None:
    train_steps, train_losses = parse_results_file(train_results_file)
    eval_steps, eval_losses =  parse_results_file(eval_resuls_file)
    
    plt.plot(train_steps, train_losses, label = "train loss")
    plt.plot(eval_steps, eval_losses, label = "eval loss")

    plt.legend()
    plt.show()

def main():
    if(len(sys.argv) == 3):
        plot(sys.argv[1], sys.argv[2])  # train and eval .csv results files
    else:
        print("Error: Invalid number of arguments!")


main()
