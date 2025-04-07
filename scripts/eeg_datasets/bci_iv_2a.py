import os
import mne
#import scipy.io
from .eegdataset import EEGDataset, EEGDataModule, np, torch
from torcheeg.datasets import BCICIV2aDataset
from moabb.datasets import BNCI2014001
from moabb.utils import set_download_dir
from torcheeg import transforms
import pickle


class BCIIV2a(EEGDataset):
    def __init__(self, data):       # deprecated
        
        super().__init__(data,
                         data.dataset.get_labels(),
                         #chans_order=['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'OZ', 'T1', 'T2'],
                         chans_order=['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'],
                         #chans_to_keep=['EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'])
                         chans_to_keep=['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'])
    
        self.__freq = 250 #Hz
        self.__seq_start = 2 * self.__freq # 2 sec
        self.__seq_end = int(3.2 * self.__freq) # 4 sec

        print("Loading the data...")
        print("Number of samples:", len(self._data))
        print("Number of EEG channels:", self._data[0][0].shape[1])
        print("EEG sample length:", self._data[0][0].shape[2])
        print("Number of classes:", self.get_num_classes())

    def __getitem__(self, index):   # unused
        return None
        eeg, label = self._data[index]
        eeg = eeg.squeeze(0)
        eeg = eeg[:,  self.__seq_start: self.__seq_end]
        eeg = eeg.numpy()

        return self._preprocess_sample(eeg, normalize=True), label
    
    def __len__(self):      # deprecated
        return len(self._data)
    
    def get_final_fc_length(self):
        #return 2440 # TODO: don't hardcode this, compute it based on the transformer output
        #return 1080 # TODO: don't hardcode this, compute it based on the transformer output
        #return 560 # TODO: don't hardcode this, compute it based on the transformer output
        return 1880 # TODO: don't hardcode this, compute it based on the transformer output

    @staticmethod
    def setup_deprecated(dataset_path):
        '''
        https://www.bbci.de/competition/iv/desc_2a.pdf
        https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00055/full
        '''
        ds = BCICIV2aDataset(root_path=dataset_path,
                                  online_transform=transforms.Compose([ transforms.To2d(), transforms.ToTensor()]),
                                  label_transform=transforms.Compose([transforms.Select('label'), transforms.Lambda(lambda x: x - 1)]), io_path=".torcheeg/datasets_1731575917613_4KanG", num_worker=7)
        
        train_ind, val_ind, test_ind = [], [], []

        for (i, row) in ds.info.iterrows():
            session = row['session'] # train or test set
            if session == "T":
                train_ind.append(i)
            elif session == "E":
                if row["subject_id"] == "A01":
                    val_ind.append(i)
                else:
                    test_ind.append(i)
        
        #train_set_perc = (80 / 100) * len(ds)
        #test_set_perc = (20 / 100) * len(ds)
        #train_data, test_data = torch.utils.data.random_split(ds, [int(train_set_perc) + 1, int(test_set_perc)])
    
        return EEGDataModule(BCIIV2a(torch.utils.data.dataset.Subset(ds, train_ind)),
                             BCIIV2a(torch.utils.data.dataset.Subset(ds, val_ind)),
                             BCIIV2a(torch.utils.data.dataset.Subset(ds, test_ind)))
    
    @staticmethod
    def setup(dataset_path, train_subject, batch_size):
        ds_file = dataset_path + "ds" + train_subject + ".pkl"
        if not os.path.isfile(ds_file):
            BCIIV2a.create_ds(dataset_path, train_subject)

        return EEGDataset.setup([ds_file], BCIIV2a.get_final_fc_length(None), batch_size)

    @staticmethod
    def create_ds(dataset_path, train_subject, align_subjects=True):   # WIP
        '''
        https://www.bbci.de/competition/iv/desc_2a.pdf
        https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00055/full
        '''
        set_download_dir('/path/to/bciiv2a/raw/')

        ds = BNCI2014001()

        data = []
        sub_nr = int(train_subject)
        
        print("Subject: ", sub_nr, end="     ")

        raw = ds.get_data([sub_nr])[sub_nr]

        train_runs = []
        test_runs = []
        for i in range(0, 6):   # 6 runs in total
            train_runs.append(raw['0train'][str(i)])
            test_runs.append(raw['1test'][str(i)])
            
        # concat the 6 runs, not sure if I should do this, not sure if the events are concatenated correctly
        train_raw = mne.concatenate_raws(train_runs, preload = True)
        test_raw = mne.concatenate_raws(test_runs, preload = True)

        selected_channels = mne.pick_channels(train_raw.info.ch_names, include=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])  # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets

        #events = mne.find_events(train_raw, stim_channel="stim")
        #(events_from_annot, event_dict) = mne.events_from_annotations(train_raw, event_id={"left_hand":1, "foot":3, "tongue":4, "right_hand":2})

        train_epochs = mne.Epochs(train_raw, tmin = 2.0, tmax = 7.0, event_id={"left_hand":1, "right_hand":2}, baseline=None, preload=True, picks=selected_channels)  # 5 seconds, 4s from cue start (to match the other datasets) + 1s resting-state
        test_epochs = mne.Epochs(test_raw, tmin = 2.0, tmax = 7.0, event_id={"left_hand":1, "right_hand":2}, baseline=None, preload=True, picks=selected_channels)  # 5 seconds, 4s from cue start (to match the other datasets) + 1s resting-state
            
        # reorder
        train_epochs = train_epochs.reorder_channels(['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])
        test_epochs = test_epochs.reorder_channels(['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])
            
        # downsample, to match PhysioNet at 160 Hz
        train_epochs = train_epochs.resample(160)
        test_epochs = test_epochs.resample(160)

        # filter
        train_epochs = train_epochs.filter(0.5, 45)
        test_epochs = test_epochs.filter(0.5, 45)

        for i, ds in enumerate([train_epochs, test_epochs]):
            left_hand_trials = ds['left_hand'].get_data()[:,:,:-1]
            right_hand_trials = ds['right_hand'].get_data()[:,:,:-1]
        
            if i == 0:  # train
                left_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 0, 'subject_label': sub_nr, 'dataset_label': 3, 'split': 'train' if idx < 50 else 'valid'} for idx, sample in enumerate(left_hand_trials)]
                right_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 1, 'subject_label': sub_nr, 'dataset_label': 3, 'split': 'train' if idx < 50 else 'valid'} for idx, sample in enumerate(right_hand_trials)]
            else:       # test
                left_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 0, 'subject_label': sub_nr, 'dataset_label': 3, 'split': 'test'} for sample in left_hand_trials]
                right_hand_samples = [{'subject': sub_nr, 'eeg':sample, 'task_label': 1, 'subject_label': sub_nr, 'dataset_label': 3, 'split': 'test'} for sample in right_hand_trials]
        
            data += left_hand_samples + right_hand_samples

        # align
        if align_subjects:
            data = EEGDataset.align_data(data)

        print("Nr. trials:", len(data), end = "      ")

        with open(dataset_path + "ds" + train_subject + ".pkl", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod #@unused
    def read_eeg_gdf_files(files, event_ids):
        data = []
        labels = []
        for f in files:
            try:
                raw = mne.io.read_raw_gdf(f, preload = True, eog = ['EOG-left', 'EOG-central', 'EOG-right'])
                raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

                events = mne.events_from_annotations(raw)
                picks = mne.pick_types(raw.info, meg=False, eeg=True)
                epochs = mne.Epochs(raw, events[0], event_id=event_ids, picks=picks, baseline=None, tmin=0, tmax=4, detrend=1, preload=True)
                data.append(epochs.get_data())
                labels.append(epochs.events[:,-1])
            except:
                continue

    @staticmethod   #@unused
    def read_eeg_mat_files(files, event_ids):
        data = []
        labels = []
        for f in files:
            raw = scipy.io.loadmat(f)
            raw = raw["data"][0]
            print(type(raw[0][0][0]))
        
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        return data, labels