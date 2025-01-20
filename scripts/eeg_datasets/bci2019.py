import os.path
#import random
import glob
import scipy.io
import pickle
from scipy.signal import resample
#from concurrent.futures import ThreadPoolExecutor

from .eegdataset import EEGDataset, EEGDataModule, np, torch


subject = False

class BCI2019(EEGDataset):
    '''
    Paper: https://academic.oup.com/gigascience/article/8/5/giz002/5304369#494016103  (2019)
    54 subjects, 62 channels, 2 classes (MI, imagining grasping with left or right hand)
    '''
    def __init__(self, data):
        # one-hot encoding

        labels = [entry['label'] for entry in data]

        super().__init__(data,
                         labels,
                         chans_order=['FP1', 'AF7', 'AF3', 'F3', 'F7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FP2', 'AF8', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P8', 'PO4', 'O2'],
                         chans_to_keep=['FP1', 'AF7', 'AF3', 'F3', 'F7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FP2', 'AF8', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P8', 'PO4', 'O2'],
                         ch_groups=[])  # temporal cortices
    
        print("Loading the data...")
        print(self._data[0].keys())
        print("Number of samples:", len(self._data))
        print("Number of EEG channels:", len(self._data[0]['eeg']))
        print("EEG sample length:", len(self._data[0]['eeg'][0]))
        print("Number of classes:", self.get_num_classes())

    def __getitem__(self, index):
        eeg = self._data[index]['eeg']

        return self._preprocess_sample(eeg, normalize=True), self._labels[index]
    
    def __len__(self):
        return len(self._data)
    
    def get_final_fc_length(self):
        # return 10440 # TODO: don't hardcode this, compute it based on the transformer output
        return 5200 # TODO: don't hardcode this, compute it based on the transformer output
        #return 6600 # TODO: don't hardcode this, compute it based on the transformer output
    
    def plot(self):
        raw = torch.tensor(self._data[0]['eeg_data']).unsqueeze(0)
        super().plot(raw, 1000)
    
    @staticmethod
    def setup(dataset_path, train_subject, batch_size):
        ds_file = dataset_path + "ds" + train_subject + ".pkl"
        if not os.path.isfile(ds_file):
            BCI2019.create_ds(dataset_path, train_subject)

        ds = None
        with open(ds_file, "rb") as f:
            ds = pickle.load(f)

        train_ds = [sample for sample in ds if sample['split'] == "train"]
        valid_ds = [sample for sample in ds if sample['split'] == "valid"]
        test_ds = [sample for sample in ds if sample['split'] == "test"]

        train_ds = BCI2019(train_ds)
        valid_ds = BCI2019(valid_ds)
        test_ds = BCI2019(test_ds)

        return EEGDataModule(train_ds, valid_ds, test_ds, int(batch_size))
    
    @staticmethod
    def print_channel_names(data):
        chans = []
        for chan in data:
            chans.append(chan[0])
        print(chans, '\', \'')

    @staticmethod
    def get_subj_name(file_name: str):
        subj_nr_start_idx = file_name.find('subj') + len('subj')
        subj_nr_end_idx = file_name.find('subj') + len('subj') + 2   # every subject number has 2 digits
        return int(file_name[subj_nr_start_idx:subj_nr_end_idx])     # extract the subject number from the file name

    @staticmethod
    def filter_channels(datasets, orig_chans, chans_to_keep: list):
        filtered_ds = []
        for ds in datasets:
            filtered_ds.append(np.stack([ds[:,i,:] for i, ch_name in enumerate(orig_chans) if ch_name in chans_to_keep], axis=1))
        return filtered_ds
    
    @staticmethod
    def reorder_channels(datasets, orig_ch_order: list, new_ch_order: list):
        reordered_ds = []
        for ds in datasets:
            temp_ds = np.zeros(ds.shape)
            for old_ch_pos, ch in enumerate(orig_ch_order):
                if ch not in new_ch_order:
                    print(ch)
                new_ch_pos = new_ch_order.index(ch)
                temp_ds[:, new_ch_pos, :] = ds[:, old_ch_pos, :]
            reordered_ds.append(temp_ds)
        return reordered_ds
    
    @staticmethod
    def downsample(datasets, new_freq: int, curr_freq: int):
        downsampled = []
        for ds in datasets:
            signal_len = ds[0].shape[1]     # current signal length
            new_nr_samples = int(signal_len * (new_freq / curr_freq))   # number of points to be left in the signal after downsampling the signal
            downsampled.append([resample(signal, new_nr_samples, axis = 1) for signal in ds])
        return downsampled

    @staticmethod
    def create_ds(dataset_path, train_subject, print_ch_names = False):
        '''
        Creates a single dataset file out of all the subject files.
        Currently just reads one .mat file (one session of an object)
        '''
        ds = []
        if train_subject:
            files = [dataset_path + "raw/sess01_subj" + train_subject + "_EEG_MI.mat"]
        else:
            files = list(glob.glob(dataset_path + "raw/" + "*.mat"))
        [ds.extend(BCI2019.process_file(file, dataset_path)) for file in files]

        with open(dataset_path + "ds" + train_subject + ".pkl", "wb") as f:
            pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
        
        #with ThreadPoolExecutor(max_workers=8) as threads:
        #    t_res = threads.map(BCI2019.process_file, files)
        
    @staticmethod
    def process_file(subj_file, dataset_path = "", print_ch_names = False):
        print("Processing file: ", subj_file)
        subj_data = scipy.io.loadmat(subj_file)
        train_data = subj_data['EEG_MI_train'].item()
        test_data = subj_data['EEG_MI_test'].item()

        assert np.array_equal(train_data[8], test_data[8])    # check train and test data have the exact same channels
        
        if print_ch_names:
            BCI2019.print_channel_names(train_data[8][0])  # 8 = index of the channel names

        train_labels = train_data[5][0]
        test_labels = test_data[5][0]

        train_data = np.transpose(train_data[0], (1, 2, 0))   # (seq_len, trials, channels) -> (trials, channels, seq_len)
        test_data = np.transpose(test_data[0], (1, 2, 0))

        train_data, test_data = BCI2019.filter_channels([train_data, test_data],
                                    ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
                                    #['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'FT9', 'TP7', 'FT10', 'TP8', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'])
                                    ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'TP7', 'TP8', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'])       # common channels to 2017 and 2019 datasets 
        
        # Mapping 1 (2019 -> 2017 datasets)
        # FP1 = FP1
        # FP2 = FP2
        # NZ (missing) = FPZ
        # AF7 = AF7
        # AF3 = AF3
        # AFZ = AFZ
        # AF4 = AF4
        # AF8 = AF8
        # F9, F7, F3 = F7, F5, F3
        # FZ = FZ
        # F4, F8, F10 = F4, F6, F8
        # FT9, FC5, FC3, FC1 = FT7, FC5, FC3, FC1
        # Missing = FCZ
        # FC2, FC4, FC6, FT10 = FC2, FC4, FC6, FT8 
        # T7, C5, C3, C1 = T7, C5, C3, C1
        # CZ = CZ
        # C2, C4, C6, T8 = C2, C4, C6, T8
        # TP9 = None
        # TP7, CP5, CP3, CP1 = TP7, CP5, CP3, CP1
        # CPZ = CPZ
        # CP2, CP4, CP6, TP8 = CP2, CP4, CP6, TP8
        # TP10 = None
        # P7 = P7
        # Missing = P5
        # P3, P1 = P3, P1
        # PZ = PZ
        # P2, P4 = P2, P4
        # Missing = P6
        # P8 = P8
        # PO3, POZ, PO4 = PO3, POZ, PO4
        # O1, OZ, O2, = O1, OZ, O2
        # PO9 = P9
        # PO10 = P10
        # Missing = IZ

        # Mapping 2 (2019 -> 2017 datasets) correct and current
        # FP1 = FP1
        # FP2 = FP2
        # NZ (missing) = FPZ
        # AF7 = AF7
        # AF3 = AF3
        # AFZ = AFZ
        # AF4 = AF4
        # AF8 = AF8
        # F9 = Missing
        # F7, NONE, F3, NONE, FZ, NONE, F4, NONE, F8 = F7, F5, F3, F1, FZ, F2, F4, F6, F8
        # F10 = Missing
        # FT9, FT10 = Missing, Missing
        # NONE, FC5, FC3, FC1, NONE, FC2, FC4, FC6, NONE = FT7, FC5, FC3, FC1, FCZ, FC2, FC4, FC6, FT8
        # T7, C5, C3, C1 = T7, C5, C3, C1
        # CZ = CZ
        # C2, C4, C6, T8 = C2, C4, C6, T8
        # TP9, TP10 = NONE, NONE
        # TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8 = TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8
        # NONE, P7, NONE, P3, P1, PZ, P2, P4, NNE, P8, NONE = P9, P7, P5, P3, P1, PZ, P2, PZ, P6, P8, P10
        # NONE, PO3, POZ, PO4, NONE = PO7, PO3, POZ, PO4, PO8
        # O1, OZ, O2 = O1, OZ, O2
        # PO9, PO10 = NONE, NONE

        train_data, test_data = BCI2019.reorder_channels([train_data, test_data],
                                                ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'TP7', 'TP8', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
                                                #['FP1', 'AF7', 'AF3', 'F3', 'F7', 'F9', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']) # for TL purposes
                                                ['FP1', 'AF7', 'AF3', 'F3', 'F7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FP2', 'AF8', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P8', 'PO4', 'O2']) # for TL purposes, same order as 2017 source dataset
    
        train_data, test_data = BCI2019.downsample([train_data, test_data], 512, 1000)
        
        subj_nr = BCI2019.get_subj_name(subj_file)
        
        valid_perc = int((10 / 100) * len(train_data))
        train_samples = train_data[valid_perc:]
        valid_samples = train_data[:valid_perc]

        train_samples = [{'subject': subj_nr, 'eeg':trial, 'label': train_labels[i], "split": "train"} for i, trial in enumerate(train_samples)]
        valid_samples = [{'subject': subj_nr, 'eeg':trial, 'label': train_labels[i], "split": "valid"} for i, trial in enumerate(valid_samples)]
        test_samples  = [{'subject': subj_nr, 'eeg':trial, 'label': test_labels[i], "split": "test"} for i, trial in enumerate(test_data)]
        
        #ds.extend(train_samples)
        #ds.extend(test_samples)
    
        return train_samples + valid_samples + test_samples

        #random.shuffle(train_ds)
        #random.shuffle(train_ds)
        #with open(dataset_path + "ds.pkl", "ab") as f:
        #    pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
