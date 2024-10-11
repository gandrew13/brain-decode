import argparse

from Datasets.EEGImageNet.ds_preprocess import EEGImageNetDataset # type: ignore



def data_prep(args: str):
    # open the download dataset files
    # take each file and preprocess it
    # save the preprocessed_data
    
    ds = EEGImageNetDataset(args)


def train():
    # load the data
    # run the training loop
    pass


def infer():
    pass



def parse_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="Path to the datasets folder.")
    parser.add_argument("-g", "--granularity", required=False, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-m", "--model", required=False, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=False, help="directory to save results")
    
    return parser.parse_args()  


def main():
    print("Hallo!")
    
    args = parse_args()

    data_prep(args)
    train()
    infer()


main()