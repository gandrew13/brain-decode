import os.path
from copy import deepcopy
import random
import glob
import scipy.io
import pickle
import mne
from .eegdataset import EEGDataset, EEGDataModule, np, torch
from moabb.datasets import PhysionetMI
from moabb.utils import set_download_dir


subject = False

class PhysioNet(EEGDataset):
    '''
    Source: https://physionet.org/content/eegmmidb/1.0.0/
            https://neurotechx.github.io/moabb/generated/moabb.datasets.PhysionetMI.html#moabb.datasets.PhysionetMI
            https://www.researchgate.net/publication/342856705_An_Accurate_EEGNet-based_Motor-Imagery_Brain-Computer_Interface_for_Low-Power_Edge_Computing
            https://www.sciencedirect.com/science/article/pii/S0957417418305359?via%3Dihub
    109 subjects, 64 channels, 4 classes (MI, left or right fist, feet lor hands movement)
    '''
    def __init__(self, data):
        # one-hot encoding

        labels = [entry['label'] for entry in data]

        super().__init__(data,
                         labels,
                         chans_order=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'], # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets
                         chans_to_keep=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'],  # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets
                         ch_groups=[])
    
        print("Loading the data...")
        print(self._data[0].keys())
        print("Number of samples:", len(self._data))
        print("Number of EEG channels:", len(self._data[0]['eeg']))
        print("EEG sample length:", len(self._data[0]['eeg'][0]))
        print("Number of classes:", self.get_num_classes())
    
    def __len__(self):
        return len(self._data)
    
    def get_final_fc_length(self):
        #return 9320 # TODO: don't hardcode this, compute it based on the transformer output
        return 1880 # TODO: don't hardcode this, compute it based on the transformer output

    @staticmethod
    def setup(dataset_path, batch_size):
        if not os.path.isfile(dataset_path):
            PhysioNet.create_ds(dataset_path, batch_size=batch_size)

        return EEGDataset.setup([dataset_path], PhysioNet.get_final_fc_length(None), batch_size)

    
    @staticmethod
    def create_ds(dataset_path, batch_size = 32, align_subjects = True):
        '''
        Creates a single dataset file out of all the subject files.
        '''
        set_download_dir('/path/physionet/raw/')
        
        ds = PhysionetMI()

        hand_imagery_runs = ['R04', 'R08', 'R12']
        #for sub_nr in range(1, 110):
        #    ds.data_path(sub_nr)
        
        #ds.get_data(list(range(1, 110)))
        #TODO Monday: preprocess + run experiment 35. on JSC
        dataset = []
        subj_id = 0    
        for sub_nr in range(1, 110):
            print("Subject: ", sub_nr, end="     ")

            if sub_nr in [88, 92, 100]:     # bad subject, with bad shape/number of trials
                continue

            raw = ds.get_data([sub_nr])[sub_nr]['0']

            runs = []
            for run_id, data in raw.items():
                run_filepath = raw[run_id].filenames[0]
                is_hand_imagery_run = True in [run in run_filepath for run in hand_imagery_runs]
                if is_hand_imagery_run:
                    runs.append(data)
            
            # concat the 3 hand imagery runs
            # TODO: not sure if I should do this, not sure if the events are concatenated correctly, to check
            raw = mne.concatenate_raws(runs, preload = True)

            #print(raw.ch_names)
            selected_channels = mne.pick_channels(raw.info.ch_names, include=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])  # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets
            epochs = mne.Epochs(raw, tmin = 0.0, tmax = 5.0, event_id={"left_hand":1, "right_hand":3}, baseline=None, preload=True, picks=selected_channels)  # trial has 4 seconds of task data + 2 sec resting state before + 2 sec resting state after. I only keep 4s + 1s, to match 2017 dataset
            # reorder
            epochs = epochs.reorder_channels(['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])
            # filter
            epochs = epochs.filter(0.5, 45)

            left_hand_trials = epochs['left_hand'].get_data()[:,:,:-1]
            right_hand_trials = epochs['right_hand'].get_data()[:,:,:-1]
            
            #(events_from_annot, event_dict) = mne.events_from_annotations(runs[0], event_id={"left_hand":1, "rest":2, "right_hand":3})

            if sub_nr > 104:        # leave a few subjects for validation
                left_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 0, 'subject_label': subj_id, 'dataset_label': 0, 'split': 'valid'} for sample in left_hand_trials]
                right_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 1, 'subject_label': subj_id, 'dataset_label': 0, 'split': 'valid'} for sample in right_hand_trials]
            else:
                left_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 0, 'subject_label': subj_id, 'dataset_label': 0, 'split': 'train'} for sample in left_hand_trials]
                right_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 1, 'subject_label': subj_id, 'dataset_label': 0, 'split': 'train'} for sample in right_hand_trials]

            data = left_hand_samples + right_hand_samples
            print("Nr. trials:", len(data), end = "      ")
            print("EEG shape:", data[0]['eeg'].shape)

            # align
            if align_subjects:
                data = EEGDataset.align_data(data)
            
            dataset.extend(data)

            subj_id += 1

        #ds = BCI2017.create_dataset_splits(ds, train_subjects, valid_subj, test_subj, random_pretrain_subjects)
        print("Last subject's ID: ", subj_id)
        with open("Datasets/physionet/ds.pkl", "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

            #left_hand_trials = []
            #right_hand_trials = []
            #for i, run in enumerate(runs):
            #    print(run.ch_names)
            #    selected_channels = mne.pick_channels(run.info.ch_names, include=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])  # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets
#
            #    runs[i] = mne.Epochs(run, tmin = 0.0, tmax = 5.0, event_id={"left_hand":1, "right_hand":3}, baseline=None, preload=True, picks=selected_channels)  # trial has 4 seconds of task data + 2 sec resting state before + 2 sec resting state after. I only keep 4s + 1s, to match 2017 dataset
#
            #    # reorder
            #    runs[i] = runs[i].reorder_channels(['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])
#
            #    # filter
            #    runs[i] = runs[i].filter(0.5, 45)
#
#
#
            #    # get raw data
            #    left_hand_trials.extend(runs[i]['left_hand'].get_data())
            #    left_hand_trials = runs[i]['left_hand'].get_data()
#
#
#
            #raw = runs[0]
            #raw.plot()
            #from matplotlib import pyplot
            #pyplot.show(block=True)
#
            ## normalize
            #raw -= np.mean(raw, axis=-1, keepdims=True)
#
            ## volts -> microvolts
            #raw *= 1e-6
