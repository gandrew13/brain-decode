from typing import Dict
import numpy as np
from scipy import linalg
import mne
import pickle
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader



class EEGDataModule(L.LightningDataModule):
    '''
    Common EEG dataset loading and processing.
    Should be inherited by all specific dataset classes.
    '''
    def __init__(self, train_data, valid_data, test_data, batch_size: int = 64):
        super().__init__()

        self.__train_data = train_data
        self.__valid_data = valid_data
        self.__test_data = test_data

        self.__batch_size = batch_size
        self.__num_workers = 8
    
    def setup(self, stage):
        # common setup operations
        pass

    def train_dataloader(self):
        return DataLoader(self.__train_data, batch_size=self.__batch_size, num_workers=self.__num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.__valid_data, batch_size=self.__batch_size, num_workers=self.__num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.__test_data, batch_size=self.__batch_size, num_workers=self.__num_workers, pin_memory=True)

    def get_num_chans(self):
        return self.__train_data.get_num_chans()

    def get_num_classes(self):
        return self.__train_data.get_num_classes()
    
    def get_final_fc_length(self):
        return self.__train_data.get_final_fc_length()




class EEGDataset(Dataset):
    def __init__(self, data, labels, final_fc_length, chans_order, chans_to_keep, ch_groups=[]):
        super().__init__()

        self._data = data
        self._labels = labels
        self._final_fc_length = final_fc_length

        self.__chan_order = chans_order
        self.__chans_to_keep = chans_to_keep
        self.__ch_groups = ch_groups

    def __len__(self):
        return len(self._data)

    def get_num_chans(self):
        return len(self.__chans_to_keep)
    
    def get_num_classes(self):
        return len(list(set(self._labels)))
    
    def get_final_fc_length(self):
        return self._final_fc_length
    
    def __getitem__(self, index):
        eeg = self._data[index]['eeg']

        #return self._preprocess_sample(eeg, normalize=True), self._labels[index]
        return eeg, self._labels[index]

    def _preprocess_sample(self, eeg, filter_channels=False, normalize=True, zero_pad=0):
        # filter channels (used to experiment with different number of channels)
        if filter_channels:
            eeg = self._filter_channels(eeg)

        if self.__ch_groups != []:
            eeg = self._group_channels(eeg)

        if normalize:
            eeg = self.normalize(eeg)

        if zero_pad != 0:
            # zero-pad last dimension (which is the temporal dimension, the length of each channel) with enough zeros to get to self.zero_pad
            eeg = torch.nn.functional.pad(torch.tensor(eeg), (-1, self.zero_pad - len(eeg[0]) + 1), "constant", 0.0)

        return eeg

    @staticmethod
    def setup(dataset_path: list, final_fc_length, batch_size):
        train_ds = []
        valid_ds = []
        test_ds = []

        for file in dataset_path:
            ds = None
            with open(file, "rb") as f:
                ds = pickle.load(f)
            print("Full dataset length: ", len(ds))
            #d = {}
            for sample in ds:
                # store number of trials per subject
                #if sample['subject'] not in d:
                #    d[sample['subject']] = 0
                #d[sample['subject']] += 1

                match sample['split']:
                    case 'train':
                        train_ds.append(sample)
                    case 'valid':
                        valid_ds.append(sample)
                    case 'test':
                        test_ds.append(sample)
                    case _:
                        print("Error: No dataset split specified for sample.")
            #for sub in d:
            #    print(sub, d[sub])
        train_ds = EEGDataset(train_ds, [(entry['task_label'], entry['subject_label'], entry['dataset_label']) for entry in train_ds], final_fc_length, [], ['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])
        valid_ds = EEGDataset(valid_ds, [(entry['task_label'], entry['subject_label'], entry['dataset_label']) for entry in valid_ds], final_fc_length, [], ['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])
        test_ds  = EEGDataset(test_ds,  [(entry['task_label'], entry['subject_label'], entry['dataset_label']) for entry in test_ds],  final_fc_length, [], ['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])

        return EEGDataModule(train_ds, valid_ds, test_ds, int(batch_size))
    
    @staticmethod
    def align_data(data):
        '''
        Euclidian Alignment
        https://www.semanticscholar.org/paper/Transfer-Learning-for-Brain%E2%80%93Computer-Interfaces%3A-A-He-Wu/2509582652372582c2d2cf37cfef7ef3f77e6cb7
        '''
        ref_mat = sum([np.cov(trial['eeg']) for trial in data]) / len(data)
        ref_mat = linalg.sqrtm(linalg.inv(ref_mat))
        for trial in data:
            trial['eeg'] = ref_mat @ trial['eeg']
        return data

    def _filter_channels(self, eeg: list) -> list:
        return [eeg[idx] for idx, ch in enumerate(self.__chan_order) if ch in self.__chans_to_keep]

    def _group_channels(self, eeg):
        '''Groups channels by computing their mean.'''
        # TODO: Optimize this method (flatten the list list channel groups and put a symbol between each group)
        new_chs = []
        for group in self.__ch_groups:
            chans = [eeg[self.__chan_order.index(ch)] for ch in group]
            avg = np.mean(chans, axis = 0, keepdims = True)
            new_chs.append(avg)
        return np.array(new_chs).squeeze(1)

    def normalize(self, eeg):
        mean = np.mean(eeg, axis=-1, keepdims=True)
        std = np.std(eeg, axis=-1, keepdims=True)
        # Ensure std is not zero to avoid division by zero.
        # If std is zero, normalization doesn't make sense, 
        # so you might set std to a small positive value or handle it in another way.
        # std = np.where(std == 0, 1e-23, std)
        return (eeg - mean) / (std + 1e-25)
    
    def plot(self, raw, freq): # (epochs, channels, time)
        info = mne.create_info(self.__chan_order, freq, 'eeg')
        raw = mne.EpochsArray(raw, info)
        #raw.
        #raw = mne.filter.notch_filter(raw, 1000, [50])


        picks = mne.pick_types(info, meg=False, eeg=True, misc=False)

        #raw.plot(picks=picks, show=True, block=False)
        raw = raw.filter(0.5, 80)
        ica = mne.preprocessing.ICA(n_components=20, random_state=0)
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

        #raw.set_cha
        #raw.set_montage(ten_twenty_montage)
        #ica.fit(raw.copy().filter(0.5, 80))
        #ica.plot_components(outlines='skirt')

        #raw.plot(picks=picks, show=True, block=True)
        
