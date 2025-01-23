import os.path
import random
import glob
import scipy.io
import pickle
from .eegdataset import EEGDataset, EEGDataModule, np, torch


subject = False

class BCI2017(EEGDataset):
    '''
    Paper: https://academic.oup.com/gigascience/article/6/7/gix034/3796323  (2017)
    52 subjects, 64 channels, 2 classes (MI, imagining moving left or right hand)
    '''
    def __init__(self, data):
        # one-hot encoding

        #data = [entry for entry in data if entry['subject'] == 'subject 1']
        labels = [entry['label'] for entry in data]

        super().__init__(data,
                         labels,
                         chans_order=['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
                         #chans_to_keep=["O1", "OZ", "O2", "PO7", "PO3", "POZ", "PO4", "PO8"]) # 8 channels, keep only the occipital lobe channels, since they are related to the visual cortex
                         #chans_to_keep=['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
                         chans_to_keep=['FP1', 'AF7', 'AF3', 'F3', 'F7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FP2', 'AF8', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P8', 'PO4', 'O2'],  # common channels for the 2017 and 2019 datasets
                         ch_groups=[])  # temporal cortices
    
        print("Loading the data...")
        print(self._data[0].keys())
        print("Number of samples:", len(self._data))
        print("Number of EEG channels:", len(self._data[0]['eeg']))
        print("EEG sample length:", len(self._data[0]['eeg'][0]))
        print("Number of classes:", self.get_num_classes())

    def __getitem__(self, index):
        eeg = self._data[index]['eeg']
        eeg = eeg[:, 2 * 512:]  # skip the 2 seconds before the cue, 5 seconds remain

        return self._preprocess_sample(eeg, normalize=True), self._labels[index]
    
    def __len__(self):
        return len(self._data)
    
    def get_final_fc_length(self):
        #return 9320 # TODO: don't hardcode this, compute it based on the transformer output
        return 6600 # TODO: don't hardcode this, compute it based on the transformer output
    
    def plot(self):
        raw = torch.tensor(self._data[0]['eeg_data']).unsqueeze(0)
        super().plot(raw, 1000)
    
    @staticmethod
    def setup(dataset_path, train_subjects, random_pretrain_subjects, valid_subj, test_subj, batch_size):
        if not os.path.isfile(dataset_path):
            BCI2017.create_ds(dataset_path)

        ds = None
        with open(dataset_path, "rb") as f:
            ds = pickle.load(f)

        # TODO: Review this code, check it works on all cases, remove hardcoded values.
        total_subjs = list(range(1, 53)) # subjects list 1-52
        total_subjs.remove(32), total_subjs.remove(29)  # remove bad subjects
        if random_pretrain_subjects == 'True':       # randomly select N subjects to train on
            train_subjs = [random.choice(total_subjs) for _ in range(1, 26)]     # split the dataset in half, this should be a param
            left_out_subjs = [subj for subj in total_subjs if subj not in train_subjs]
        elif train_subjects:
            train_subjs = [int(subj) for subj in train_subjects.split(',')]
            left_out_subjs = [subj for subj in total_subjs if subj not in train_subjs]
        else:
            train_subjs = total_subjs
            left_out_subjs = []

        print("Train subjs: ", train_subjs)
        print("Left-out subjects (for fine-tuning): ", left_out_subjs)
        print("Valid subject: ", valid_subj)
        print("Test subject: ", test_subj)

        #train_ds = [sample for sample in ds if sample['subject'] != 'subject 50' and sample['subject'] != test_subj]
        test_subj = int(test_subj)
        valid_subj = int(valid_subj)
        #train_ds = [sample for sample in ds if sample['subject'] in train_subjs and sample['subject'] != test_subj]
        train_ds = [sample for sample in ds if sample['subject'] in train_subjs] # don't exclude the test subject from training, we're training on the entire dataset for TL

        # Continue from here, implement valid and test ds subjects, push, and sync with the code on JEUWELS
        valid_ds = [sample for sample in ds if sample['subject'] == valid_subj]
        test_ds = [sample for sample in ds if sample['subject'] == test_subj]

        train_ds = BCI2017(train_ds)
        valid_ds = BCI2017(valid_ds)
        test_ds = BCI2017(test_ds)

        return EEGDataModule(train_ds, valid_ds, test_ds, int(batch_size))

    @staticmethod
    def create_ds(dataset_path):
        '''
        Creates a single dataset file out of all the subject files.
        '''
        #temp = scipy.io.loadmat(dataset_path + "s1_trial_sequence_v1.mat")
        #temp = temp['trial_sequence'].item()

        ds = []
        sample_len_ms = np.array([[-2000,  5000]], dtype=np.int16)
        for subj_file in glob.glob(dataset_path + "*.mat"):
            subj_data = scipy.io.loadmat(subj_file)
            eeg_data = subj_data["eeg"].item()
            if eeg_data[13].item() == 'subject 34' or eeg_data[13].item() == 'subject 29':  # skip these subjects, too many bad trials
                continue
            print(eeg_data[13].item())
            #if eeg_data[13].item() == 'subject 45' or eeg_data[13].item() == 'subject 26' or eeg_data[13].item() == 'subject 37' or eeg_data[13].item() == 'subject 33' or eeg_data[13].item() == 'subject 44':
            #    global subject
            #    subject = True
            print(subj_data["eeg"].dtype)
            num_trials = eeg_data[9].item()
            print(eeg_data[10])
            #assert sample_len_ms == eeg_data[10]
            assert np.array_equal(sample_len_ms, eeg_data[10])
                 
            print(eeg_data[9], eeg_data[7].shape)
            print("Frame:", eeg_data[10])
            #print(eeg_data[14].item()[0].size)
            #print("Imagery event:", np.where(eeg_data[11] == 1)[1])
            
            # remove the EMG channels
            #eeg_data[7] = eeg_data[7]
            #eeg_data[8] = eeg_data[7]

            left_hand_trials = np.split(eeg_data[7], num_trials, axis=1)
            right_hand_trials = np.split(eeg_data[8], num_trials, axis=1)

            #left_hand_trials = BCI52sub_64ch_2class.elim_bad_trials(left_hand_trials, eeg_data[14], 0)       # left hand
            #right_hand_trials = BCI52sub_64ch_2class.elim_bad_trials(right_hand_trials, eeg_data[14], 1)     # right hand

            subject = int(eeg_data[13].item().split(' ')[1])
            left_hand_samples = [{'subject': subject, 'eeg':sample[:64], 'label': 0} for sample in left_hand_trials]
            right_hand_samples = [{'subject': subject, 'eeg':sample[:64], 'label': 1} for sample in right_hand_trials]
            ds.extend(left_hand_samples)
            ds.extend(right_hand_samples)
            # TODO: Process the data !!! Maybe also check all datasets again, and keep only the best ones to test on
        
        random.shuffle(ds)
        with open(dataset_path, "wb") as f:
            pickle.dump(ds, f)

    @staticmethod
    def elim_bad_trials(data, bad_trials, hand): 
        temp = bad_trials['bad_trial_idx_voltage']
        print(type(temp), temp.shape, temp.dtype)
        temp = temp.item()
        print(type(temp), temp.shape, temp.dtype)
        temp = temp[0,0]
        print(type(temp), temp.shape, temp.dtype)
        #if temp.shape != (0, 0):
            
        #    temp = temp.item()
        #    print(type(temp), temp)

        return []
        temp1 = temp.item().ravel()[hand]
        
        temp = bad_trials['bad_trial_idx_mi']
        temp2 = temp.item().ravel()[hand]

        out = np.concatenate(temp1.item(), temp2.item())

        print(out)
        return
        # TODO: eliminate bad trials
        if subject == True:
            print("AAA")
        for arr in bad_trials:   # a tuple of 2 elements, trials eliminated by either magnitude or EMG
            #print(eeg_data[13].item(), bad_trial_ind[1][0][0])
            hand_trials = np.ravel(bad_trials).item()
            hand_trials = arr.tolist()[0]
            hand_trials = hand_trials[hand]
            hand_trials = hand_trials.flatten()
            #if type(hand_trials[0]) is list:
            #    print("AAA")
            bad_idx = [trial.item() - 1 for trial in hand_trials.flatten().item()]
            data = np.delete(data, bad_idx)
            #[print(trial.item()) for trial in hand_trials] 