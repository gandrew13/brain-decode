import argparse
import gc
from model_runner import Runner, np


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
    parser.add_argument("-ft", "--fine_tune_mode", required=False, default='0', help="1 - full freeze, 2 - unfreeze classification layer, 3 - unfreeze all FC layers")
    parser.add_argument("-cv", "--cross_validation", required=True, help="Run NR_SUBJECTS experiments by leaving one subject out each time")
    parser.add_argument("-tss", "--train_subjects", required=False, default="", help="Subjects to include in the training dataset split")
    parser.add_argument("-rts", "--random_pretrain_subjects", required=False, help="Whether subjects in the pretraining dataset split are randomly selected or not")
    parser.add_argument("-vs", "--valid_subject", required=True, help="Selected subject to validate on")
    parser.add_argument("-ts", "--test_subject", required=True, help="The left-out test subject")
    parser.add_argument("-nr", "--num_runs", required=False, default=1, help="The number of times to repeat an experiment")
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

    #test_subjects = filter(len, args.test_subject.split(','))
    test_subjects = [args.test_subject]
    if args.cross_validation == 'True':
        test_subjects = args.train_subjects.split(',')
        
    for subj in test_subjects:
        print("\n\n===================================== NEW RUN ==========================================")
        args.test_subject = subj
        exp_res = []
        for i in range(int(args.num_runs)):    
            print("Experiment: ", i)
            runner = Runner(args)
            runner.run()
            exp_res.append(runner.get_test_accuracy())
            del runner
            gc.collect()
        print(subj, exp_res, "Mean: ", np.mean(exp_res), "Std dev: ", np.std(exp_res))

    #data_prep(args)
    #train()
    #infer()


main()






def compute_mean_stddev():
    # Helper function to help compute the mean and standard deviation for per subject experiment results
    pass
    # no pretrain
    #l = [ 51.58,64.27 ,57.64 ,67.91 ,59.11 ,72.87 ,57.52 ,68.19 ,50.47 ,52.2  ,55.19 ,70.79 ,56.39 ,56.14 ,84.18 ,57.09 ,55.18 ,54.0  ,57.97 ,61.75]
    #print(np.mean(l), np.std(l))
    #
    #
    ## 2017, single head
    #fc = [83.31, 82.46, 92.62, 78.01, 92.51, 92.7,78.81,67.9,78.05,53.33,62.5,84.15,74.29,74.19,87.24,68.41,76.22,93.16,81.07,78.02]
    #print(np.mean(fc), np.std(fc))
    #
    #fm = [ 83.08,87.19,92.68,80.03,92.44,90.59,81.36,75.07,79.46,56.22,62.79,85.36,74.0 ,74.87,86.6 ,68.02,76.65,93.84,81.86,78.19]
    #print(np.mean(fm), np.std(fm))
    #
    ## 2017 + physionet, single head
    #fc = [89.03, 78.74, 91.31, 81.79, 90.99, 93.63, 75.3 , 68.11, 81.73, 53.54, 57.47, 81.84, 72.5 , 79.58, 83.96, 63.49, 74.47, 95.33, 79.14, 70.96]
    #print(np.mean(fc), np.std(fc))
    #
    #fm = [ 86.32,79.98,90.46,81.05,90.71,90.2 ,72.45,68.94,80.65,54.93,57.99,84.24,71.34,77.78,84.0 ,67.84,78.96,94.73,78.83,74.91]
    #print(np.mean(fm), np.std(fm))
    #
    #
    ## 2017 + physionet, multihead (multitask)
    #fc = [ 79.37, 70.77 , 93.0  , 82.57 , 91.3  , 93.93 , 85.15 , 66.49 , 80.44 , 55.55 , 63.98 , 78.48 , 80.09 , 73.6  , 87.24 , 65.5  , 74.77 , 87.15 , 82.87 , 72.87]
    #print(np.mean(fc), np.std(fc))
    #
    #fm = [ 82.16, 71.02 , 91.47 , 79.46 , 83.95 , 90.25 , 80.22 , 72.0  , 78.03 , 57.58 , 64.15 , 79.6  , 77.79 , 73.49 , 84.7  , 68.67 , 77.4  , 85.39 , 80.25 , 75.58]
    #print(np.mean(fm), np.std(fm))
    #
    #
    #
    ### 2017 + physionet + 2019 single-head
    #
    #fc = [ 81.303, 52.826, 77.192, 53.950, 55.875, 61.420, 74.539, 55.698, 62.607]
    #print(np.mean(fc), np.std(fc))
    #
    #fm = [ 86.943, 64.091  , 81.003  , 57.413  , 64.86   , 61.385  , 83.842  , 55.2630 , 67.265]
    #print(np.mean(fm), np.std(fm))
    #
    ##no pretrain
    #l = [ 57.39, 52.621 , 55.574 , 53.806 , 54.975 , 50.0 , 60.085 , 52.481 , 48.477]
    #print(np.mean(l), np.std(l))
    #
    #
    ### 2017 + physionet + 2019 mulitask, testing on BCI IV 2a
    #fc = [ 82.882,58.165 ,73.190 ,51.788 ,56.540 ,56.421 ,65.961 ,57.332 ,59.322]
    #print(np.mean(fc), np.std(fc))
    #
    #fm = [ 86.743, 70.535 , 85.345 , 58.379 , 69.735 , 59.531 , 85.628 , 52.529 , 71.995]
    #print(np.mean(fm), np.std(fm))