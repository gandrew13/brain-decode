import argparse
from model_runner import Runner


def data_prep(args: str):
    # open the download dataset files
    # take each file and preprocess it
    # save the preprocessed_data
    
    #ds = EEGImageNetDataset(args)
    pass


def train():
    # load the data
    # run the training loop
    pass


def infer():
    pass



def parse_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name: [eegimagenet, alljoined1, bci_iv_2a, 52sub_64ch_2class]")
    parser.add_argument("-dp", "--dataset_path", required=False, help="Path of the dataset file")
    parser.add_argument("-m", "--model", required=True, help="Model name: [eegconformer, eegnet, mlp]")
    parser.add_argument("-e", "--epochs", required=True, help="Number of epochs to train the model for")
    parser.add_argument("-bs", "--batch_size", required=False, default=32, help="Batch size")
    parser.add_argument("-dt", "--deterministic", required=False, default=True, help="Whether to allow randomness or make everything deterministic")
    parser.add_argument("-l", "--load_pretrained", required=False, help="Path to the pretrained model")
    parser.add_argument("-f", "--freeze_model", required=True, help="Whether freeze the model or not")
    parser.add_argument("-ft", "--fine_tune", required=False, default=False, help="Whether to freeze or train the classification layers")
    parser.add_argument("-cv", "--cross_validation", required=True, help="Run NR_SUBJECTS experiments by leaving one subject out each time")
    parser.add_argument("-tss", "--train_subjects", required=False, help="Subjects to include in the training dataset split")
    parser.add_argument("-rts", "--random_pretrain_subjects", required=False, help="Whether subjects in the pretraining dataset split are randomly selected or not")
    parser.add_argument("-vs", "--valid_subject", required=True, help="Selected subject to validate on")
    parser.add_argument("-ts", "--test_subject", required=True, help="The left-out test subject")
    parser.add_argument("-p", "--plot_file", required=False, help="Path to the results file to plot")
    #parser.add_argument("-g", "--granularity", required=False, help="choose from coarse, fine0-fine4 and all")
    #parser.add_argument("-m", "--model", required=False, help="model")
    #parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    #parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    #parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    #parser.add_argument("-o", "--output_dir", required=False, help="directory to save results")
    
    return parser.parse_args()


def main():
    print("Hallo!")
    
    args = parse_args()

    test_subjects = [args.test_subject]
    if args.cross_validation == 'True':
        test_subjects = args.train_subjects.split(',')
        
    for subj in test_subjects:
        print("\n\n===================================== NEW RUN ==========================================")
        args.test_subject = subj
        runner = Runner(args)
        runner.run()
        runner.plot(args.plot_file)

    #data_prep(args)
    #train()
    #infer()


main()