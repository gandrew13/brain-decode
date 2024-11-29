
import mne
#import scipy.io
from .eegdataset import EEGDataset, EEGDataModule, np, torch
from torcheeg.datasets import BCICIV2aDataset
from torcheeg import transforms


class BCIIV2a(EEGDataset):
    def __init__(self, data):
        

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

    def __getitem__(self, index):
        eeg, label = self._data[index]
        eeg = eeg.squeeze(0)
        eeg = eeg[:,  self.__seq_start: self.__seq_end]
        eeg = eeg.numpy()

        return self._preprocess_sample(eeg, normalize=True), label
    
    def __len__(self):
        return len(self._data)
    
    def get_final_fc_length(self):
        #return 2440 # TODO: don't hardcode this, compute it based on the transformer output
        #return 1080 # TODO: don't hardcode this, compute it based on the transformer output
        return 560 # TODO: don't hardcode this, compute it based on the transformer output

    @staticmethod
    def setup(dataset_path):
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