import os.path
from copy import deepcopy
import random
import glob
#import moabb.datasets
import scipy.io
import pickle
import mne
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
                         chans_order=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'], # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets
                         #chans_order=['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],  # orig channels from the paper
                         #chans_to_keep=["O1", "OZ", "O2", "PO7", "PO3", "POZ", "PO4", "PO8"]) # 8 channels, keep only the occipital lobe channels, since they are related to the visual cortex
                         #chans_to_keep=['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
                         #chans_to_keep=['FP1', 'AF7', 'AF3', 'F3', 'F7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FP2', 'AF8', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P8', 'PO4', 'O2'],  # common channels for the 2017 and 2019 datasets
                         chans_to_keep=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'],  # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets
                         ch_groups=[])  # temporal cortices
    
        print("Loading the data...")
        print(self._data[0].keys())
        print("Number of samples:", len(self._data))
        print("Number of EEG channels:", len(self._data[0]['eeg']))
        print("EEG sample length:", len(self._data[0]['eeg'][0]))
        print("Number of classes:", self.get_num_classes())

    def __getitem__(self, index):
        ### Unused, the parent method is called.
        return None
    
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
            BCI2017.create_ds(dataset_path, train_subjects=train_subjects, valid_subj=valid_subj, test_subj=test_subj, random_pretrain_subjects=random_pretrain_subjects)

        return EEGDataset.setup([dataset_path], BCI2017.get_final_fc_length(None), batch_size)

        #ds = None
        #with open(dataset_path, "rb") as f:
        #    ds = pickle.load(f)
#
        #train_ds = []
        #valid_ds = []
        #test_ds = []
#
        #for sample in ds:
        #    match sample['split']:
        #        case 'train':
        #            train_ds.append(sample)
        #        case 'valid':
        #            valid_ds.append(sample)
        #        case 'test':
        #            test_ds.append(sample)
        #        case _:
        #            print("Error: No dataset split specified for sample.")
#
        ##train_ds = train_ds[:1000]
        ##valid_ds = valid_ds[:1000]
#
        #train_ds = BCI2017(train_ds)
        #valid_ds = BCI2017(valid_ds)
        #test_ds = BCI2017(test_ds)
#
        #return EEGDataModule(train_ds, valid_ds, test_ds, int(batch_size))
    
    @staticmethod
    def filter_channels(data, orig_chans, chans_to_keep: list):
        for entry in data:
            entry['eeg'] = np.stack([entry['eeg'][i,:] for i, ch_name in enumerate(orig_chans) if ch_name in chans_to_keep], axis=0)
        return data

    @staticmethod
    def filter(data, sample_rate = 512, min_freq = 8, max_freq = 30):
        nyq = 0.5 * sample_rate
        min_freq = min_freq / nyq
        max_freq = max_freq / nyq
        #a, b = scipy.signal.iirfilter(3, [min_freq, max_freq])
        sos = scipy.signal.butter(3, [min_freq, max_freq], 'bandpass', analog=False, fs=sample_rate, output='sos')
        for entry in data:
            entry['eeg'] = scipy.signal.sosfiltfilt(sos, entry['eeg'], axis = 1)
            #entry['eeg'] = scipy.signal.filtfilt(a, b, entry['eeg'], axis = 1)
            #import mne
            #entry['eeg'] = mne.filter.filter_data(np.float64(entry['eeg']), 512, 5, 100)
        return data
    
    @staticmethod
    def create_dataset_splits(ds, train_subjects, valid_subj, test_subj, random_pretrain_subjects):
        # TODO: Review this code, check it works on all cases, remove hardcoded values.
        total_subjs = list(range(1, 53)) # subjects list 1-52
        #total_subjs.remove(32), total_subjs.remove(29)  # remove bad subjects
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
        train_ds = []
        valid_ds = []
        test_ds = []

        train_on_all_subjects = True    # enabled if we want to train on the entire dataset (all data of all subjects), in order to create a train/valid split
        if train_on_all_subjects:
            subj_dict = {}  # dataset divided by subject

            for sample in ds:
                subject = sample['subject']
                if subject not in subj_dict:
                    subj_dict[subject] = []
                subj_dict[subject].append(sample)

            train_perc = 97 #TODO: don't hardocde
            train_ds, valid_ds = [], []
            for _, v in subj_dict.items():
                nr_samples = int((train_perc / 100) * len(v))
                train_ds.extend(v[:nr_samples])
                valid_ds.extend(v[nr_samples:])
            for sample in train_ds:
                sample['split'] = 'train'
            for sample in valid_ds:
                sample['split'] = 'valid'
        else:      # split in train/valid/test by subject 
            for sample in ds:
                sample_subj = sample['subject']
                if sample_subj in train_subjs: #and sample['subject'] != test_subj # don't exclude the test subject from training, we're training on the entire dataset for TL
                    train_ds.append(deepcopy(sample))
                    train_ds[-1]['split'] = 'train'
                if sample_subj == valid_subj:
                    valid_ds.append(deepcopy(sample))
                    valid_ds[-1]['split'] = 'valid'
                if sample_subj == test_subj:
                    test_ds.append(deepcopy(sample))
                    test_ds[-1]['split'] = 'test'

        return train_ds + valid_ds + test_ds
    
    @staticmethod
    def create_ds(dataset_path, train_subjects, valid_subj, test_subj, random_pretrain_subjects, filter_data = False, align_subjects = False):
        '''
        Creates a single dataset file out of all the subject files.
        '''
        ds = []
        for subj_file in glob.glob(dataset_path + "raw/" + "*.mat"):
            print("Processing: ", subj_file, "...")
            subj_data = scipy.io.loadmat(subj_file, simplify_cells=True)
            eeg_data = subj_data["eeg"]

            left_hand_trials = eeg_data['imagery_left'][:64]           # keep only first 64 channels
            right_hand_trials = eeg_data['imagery_right'][:64]         # keep only first 64 channels

            # normalize
            left_hand_trials -= np.mean(left_hand_trials, axis=-1, keepdims=True)# / (np.std(left_hand_trials, axis =-1, keepdims=True) + 1e-25)
            right_hand_trials -= np.mean(right_hand_trials, axis=-1, keepdims=True)

            # volts -> microvolts
            left_hand_trials *= 1e-6
            right_hand_trials *= 1e-6

            # create MNE raw arrays
            info = mne.create_info(['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
                                   sfreq=512, ch_types=["eeg"] * 64)
            
            left_hand_raw_array = mne.io.RawArray(left_hand_trials, info)
            right_hand_raw_array = mne.io.RawArray(right_hand_trials, info)
            
            montage = mne.channels.make_standard_montage("standard_1005")
            left_hand_raw_array.set_montage(montage)
            right_hand_raw_array.set_montage(montage)

            # create events
            event_onset_indices = np.where(eeg_data['imagery_event'] == 1)[0]
            events_struct = [[idx, 0, 1] for idx in event_onset_indices]
            #events_struct_right = [[idx, 0, 2] for idx in event_onset_indices]

            # select channels
            selected_channels = mne.pick_channels(info.ch_names, include=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])  # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets

            # create MNE Epochs
            #left_hand_epochs = mne.Epochs(left_hand_raw_array, events_struct, tmin = -0.5, tmax = 5.0, event_id=dict(left_hand=1), preload=True, baseline=(-0.5, 0.0), picks=selected_channels)         # 3 sec of stimulus + 2s resting state after
            #right_hand_epochs = mne.Epochs(right_hand_raw_array, events_struct, tmin = -0.5, tmax = 5.0, event_id=dict(right_hand=1), preload=True, baseline=(-0.5, 0.0), picks=selected_channels)      # 3 sec of stimulus + 2s resting state after

            left_hand_epochs = mne.Epochs(left_hand_raw_array, events_struct, tmin = 0.0, tmax = 5.0, event_id=dict(left_hand=1), preload=True, baseline=None, picks=selected_channels)         # 3 sec of stimulus + 2s resting state after
            right_hand_epochs = mne.Epochs(right_hand_raw_array, events_struct, tmin = 0.0, tmax = 5.0, event_id=dict(right_hand=1), preload=True, baseline=None, picks=selected_channels)      # 3 sec of stimulus + 2s resting state after

            # remove pre-stimulus (before 0 seconds) data
            #left_hand_epochs = left_hand_epochs.crop(0, 5)     # no need to crop since I removed the pre-stimulus segment
            #right_hand_epochs = right_hand_epochs.crop(0, 5)

            # filter
            left_hand_epochs = left_hand_epochs.filter(0.5, 45)
            right_hand_epochs = right_hand_epochs.filter(0.5, 45)

            # combine epochs
            #epochs = mne.concatenate_epochs([left_hand_epochs, right_hand_epochs])
            #epochs = epochs.set_eeg_reference()
            
            #left_hand_epochs = epochs['left_hand']
            #right_hand_epochs = epochs['right_hand']

            # re-reference
            left_hand_epochs = left_hand_epochs.set_eeg_reference()
            right_hand_epochs = right_hand_epochs.set_eeg_reference()

            # remove artifacts
            #ica = mne.preprocessing.ICA(n_components=21, random_state=42)
            #ica.fit(left_hand_epochs)
            #bad_idx, scores = ica.find_bads_eog(epochs1, 'O2', threshold=2)
            #ica.plot_components();

            #left_hand_epochs[0].plot(scalings=dict(eeg=1e-3))
            #left_hand_epochs[0].compute_psd().plot()
            #plt.show(block=True)

            left_hand_trials = left_hand_epochs.get_data()
            right_hand_trials = right_hand_epochs.get_data()

            # remove bad trials
            #left_hand_trials, right_hand_trials = BCI2017.elim_bad_trials(left_hand_trials, right_hand_trials, eeg_data['bad_trial_indices'])
            
            subject = int(eeg_data['subject'].split(' ')[1])

            left_hand_samples = [{'subject': subject, 'eeg':sample, 'task_label': 0, 'subject_label': subject - 1} for sample in left_hand_trials]
            right_hand_samples = [{'subject': subject, 'eeg':sample, 'task_label': 1, 'subject_label': subject - 1} for sample in right_hand_trials]

            if len(left_hand_samples) < 50 or len(right_hand_samples) < 50:
                print("Subject ", subject, " has too few trials.")

            data = left_hand_samples + right_hand_samples
            if len(data) == 0:
                continue
    
            # align
            if align_subjects:
                data = EEGDataset.align_data(data)
            
            ds.extend(data)
        
        #random.shuffle(ds)

        ds = BCI2017.create_dataset_splits(ds, train_subjects, valid_subj, test_subj, random_pretrain_subjects)
        with open(dataset_path, "wb") as f:
            pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def elim_bad_trials(left_hand_trials, right_hand_trials, bad_trials):
        left_hand_bad_trials = []
        right_hand_bad_trials = []
        bad_trials_voltage = bad_trials['bad_trial_idx_voltage']

        assert bad_trials_voltage.shape == (2,)
        
        lh = bad_trials_voltage[0]
        rh = bad_trials_voltage[1]

        if isinstance(lh, int):
            lh = np.array([lh])

        if isinstance(rh, int):
            rh = np.array([rh])

        if lh.shape != (0,):
            left_hand_bad_trials.extend(lh)

        if rh.shape != (0,):
            right_hand_bad_trials.extend(rh)

        bad_trials_emg = bad_trials['bad_trial_idx_mi']
        assert bad_trials_emg.shape == (2,)
        
        lh = bad_trials_emg[0]
        rh = bad_trials_emg[1]

        if isinstance(lh, int):
            lh = np.array([lh])

        if isinstance(rh, int):
            rh = np.array([rh])

        if lh.shape[0] != 0:
            left_hand_bad_trials.extend(lh)

        if rh.shape[0] != 0:
            right_hand_bad_trials.extend(rh)

        print(left_hand_bad_trials)     #TODO append all bad trails for left hand for a subject, then -1 then remove the trials
        print(right_hand_bad_trials)
        left_hand_bad_trials = np.unique(left_hand_bad_trials)
        right_hand_bad_trials = np.unique(right_hand_bad_trials)
        print(left_hand_bad_trials)     #TODO append all bad trails for left hand for a subject, then -1 then remove the trials
        print(right_hand_bad_trials)
        
        if len(left_hand_bad_trials) > 45:     # if more than 50 bad trials then remove subject's specific hand data
            left_hand_trials = []
        elif len(left_hand_bad_trials) > 0:
            left_hand_bad_trials = [x - 1 for x in left_hand_bad_trials]
            left_hand_trials = np.delete(left_hand_trials, left_hand_bad_trials, axis=0)
        
        if len(right_hand_bad_trials) > 45:    # if more than 50 bad trials then remove subject's specific hand data
            right_hand_trials = []
        elif len(right_hand_bad_trials) > 0:
            right_hand_bad_trials = [x - 1 for x in right_hand_bad_trials]
            right_hand_trials = np.delete(right_hand_trials, right_hand_bad_trials, axis=0)
        return left_hand_trials, right_hand_trials


    @staticmethod
    def create_ds_deprecated(dataset_path, train_subjects, valid_subj, test_subj, random_pretrain_subjects, filter_data = False, align_subjects = True):
        '''
        Creates a single dataset file out of all the subject files.
        Deprecated, used for testing purposes.
        '''
        use_moabb = True
        if use_moabb:
           import moabb
           ds = moabb.datasets.Cho2017()
           data = ds.get_data(subjects=[1], cache_config=moabb.datasets.base.CacheConfig(path='/media/agalbenus/EXTERN/moabb/'))
           data = data[1]['0']['0']
        #   data.plot()
        #   matplotlib.pyplot.show(block=True)
        #   import mne
        #   data = mne.Epochs(data, data.info).get_data()
           import mne   
           #events = mne.events_from_annotations(data)
           events_orig = mne.find_events(data, stim_channel='Stim')
           data = data.drop_channels(['EMG1', 'EMG2', 'EMG3', 'EMG4', 'Stim'])

        
        #temp = scipy.io.loadmat(dataset_path + "s1_trial_sequence_v1.mat")
        #temp = temp['trial_sequence'].item()

        ds = []
        sample_len_ms = np.array([[-2000,  5000]], dtype=np.int16)
        for subj_file in glob.glob(dataset_path + "raw/" + "*.mat"):
            #if "s03.mat" not in subj_file:
            #    continue
            subj_data = scipy.io.loadmat(subj_file)
            eeg_data = subj_data["eeg"].item()

            events = eeg_data[11][0]
            event_onset_indices = np.where(events == 1)[0]
            mne_events = [[idx, 0, 1] for idx in event_onset_indices]

            #if eeg_data[13].item() == 'subject 34' or eeg_data[13].item() == 'subject 29':  # skip these subjects, too many bad trials
            #    continue
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
            left_hand_trials = np.split(eeg_data[7][:64,:], num_trials, axis=1)
            right_hand_trials = np.split(eeg_data[8], event_onset_indices, axis=1)

            #for i, elem in enumerate(left_hand_trials):
            #    left_hand_trials[i] = elem[:,1023:]

            import mne
            from matplotlib import pyplot
            #raw_array = [sample[:64,:] for sample in left_hand_trials]
            #raw_array = [sample[:64] for sample in left_hand_trials]
            #raw_array = left_hand_trials[:][:,64]#[:64,:]
            raw_array = eeg_data[7][:64,:] * 1e-6
            #raw_array = eeg_data[7][:64,:]
            #raw_array = np.hstack(left_hand_trials)
            #raw_array *= 1e-6
            #for i, el in enumerate(raw_array):
            #    raw_array[i] = raw_array[i] * 1e-6
            #pyplot.plot(list(range(3584)), raw_array.T)
            #pyplot.show()
            ch_types = ["eeg"] * 64
            montage = mne.channels.make_standard_montage("standard_1005")
            info = mne.create_info([
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ],
                                   512, ch_types)
            raw_array = mne.io.RawArray(raw_array, info)
            raw_array.set_montage(montage)
            #raw_array.plot(show=True)
            #pyplot.show(block=True)
            
            #events =
            #events_orig = events_orig[:100]
            events_orig = mne_events
            #epochs = mne.EpochsArray(raw_array, info, events_orig, -2.0, dict(left_hand=1))
            #from mne.preprocessing import create_eog_epochs
            #eog_average = create_eog_epochs(raw_array, reject=dict(mag=5e-12, grad=4000e-13),
            #                    picks=picks_meg).average()
            left_hand_trials = np.array(left_hand_trials) * 1e-6
            #print(left_hand_trials.shape)
            #print(left_hand_trials[0,1,0:20])
            #print(raw_array[1,1023:1023 + 20])
            #epochs3 = mne.EpochsArray(left_hand_trials, info, events_orig, tmin = 0.0, event_id=dict(left_hand=1), baseline = (0,0))
            #epochs3.set_montage(montage)
            selected_channels = mne.pick_channels(info.ch_names, include=['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz', 'FC4', 'FC2', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2'])
            epochs1 = mne.Epochs(raw_array, events_orig, tmin = -1.0, tmax = 5.0, event_id=dict(left_hand=1), preload=True, baseline=(-1.0, 0.0), picks=selected_channels)
            #epochs2 = mne.Epochs(data, events_orig, tmin = -2.0, tmax = 5.0, event_id=dict(left_hand=1), preload=True)
            epochs1[0].plot(scalings=dict(eeg=1e-3))
            #data = epochs1.get_data()
            #data3 = epochs3.get_data()
            #print(data[0,0,0:20])
            #print(data3[0,0,0:20 + 20])
            #epochs3 = epochs3.filter(0, 50)
            #epochs1 = epochs1.filter(0, 50)
            #epochs1 = epochs1.notch_filter([60])
            #epochs1 = epochs1.resample(128)
            #left_hand1 = epochs1['left_hand'].average()
            #left_hand3 = epochs3['left_hand'].average()
            #left_hand1.plot(scalings=dict(eeg=1e-6))
            #left_hand3.plot(scalings=dict(eeg=1e-6))
            #left_hand2 = epochs2['left_hand'].average()
            times = np.arange(3.0, 5, 0.1)
            #left_hand1.plot_topomap(times, ch_type="eeg")
            #left_hand2.plot_topomap(times, ch_type="eeg")
            #epochs1 = epochs1.filter(0, 50)
            #data = epochs1.get_data()
            #left_hand1.plot_joint()
            #epochs1.compute_psd().plot()
            #left_hand1.plot(spatial_colors=True, scalings=dict(eeg=1e-6))
            #left_hand2.plot(spatial_colors=True, scalings=dict(eeg=1e-6))
            #left_hand3.plot_joint(times)

            random_state = 42   # ensures ICA is reproducible each time it's run

            # Fit ICA
            #ica = mne.preprocessing.ICA(n_components=64,
            #                            random_state=random_state,
            #                            )
            #ica.fit(epochs3, decim=3)
            #bad_idx, scores = ica.find_bads_eog(epochs1, 'O2', threshold=2)
            #ica.plot_components();
            #ica.exclude = [0,1,2,3,4,5,6,7,8,9,10]
            #ica.exclude += [11,12,13,14,15,16,17,18,19,20]
            #ica.exclude += [21,22,23,24,25,26,27,28,29,30]
            #ica.exclude += [31,32,33,34,35,36,37,38,39,40]
            #ica.exclude += [41,42,43,44,45,46,47,48,49,50]
            #ica.exclude += [51,52,53,54,55,56,57,58,59,60]
            #ica.exclude += [61,62,63,64]
            #ica.apply(epochs3, exclude=ica.exclude)['left_hand'].average().plot()
            pyplot.show(block=True)

            #data = epochs.get_data()
            
            exit()
            left_hand_trials, right_hand_trials = BCI2017.elim_bad_trials(left_hand_trials, right_hand_trials, eeg_data[14], 0)       # left hand
            #right_hand_trials = BCI52sub_64ch_2class.elim_bad_trials(right_hand_trials, eeg_data[14], 1)     # right hand
            #continue
            subject = int(eeg_data[13].item().split(' ')[1])

            left_hand_samples = [{'subject': subject, 'eeg':sample[:64,2 * 512:], 'label': 0} for sample in left_hand_trials]
            right_hand_samples = [{'subject': subject, 'eeg':sample[:64,2 * 512:], 'label': 1} for sample in right_hand_trials]

            if len(left_hand_samples) < 50 or len(right_hand_samples) < 50:
                print("Subject ", subject, " has too few trials.")

            data = left_hand_samples + right_hand_samples
            if len(data) == 0:
                continue

            data = BCI2017.filter_channels(data,
                                           ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
                                           ['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POZ', 'PZ', 'CPZ', 'FZ', 'FC4', 'FC2', 'CZ', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2']) # 21 channels common to BCI2017, BCI2019 and BCI IV 2a datasets

            if filter_data:
                data = BCI2017.filter(data, 512, 8, 30)

            # TODO: Normalize here and remove normalization from the __getitem__ method, from _preprocess_sample, for both 2017 and 2019 datasets
            # Also either normalize each EEG sample or normalize with respect to all samples, like compute the mean of each FP3 channel etc.
            #if normalize: ...
            #Also, before this, try without EA, see if the results change, to write the report
            # Also maybe use the same EEG length, so as to not be required to change the head of the NN
            # Also freeze the CNN and fine-tune the attention module.

            normalize = True
            if normalize:
                for entry in data:
                    entry['eeg'] = EEGDataset.normalize(None, entry['eeg'])
            
            if align_subjects:
                data = EEGDataset.align_data(data)

            ds.extend(data)
            # TODO: Process the data !!! Maybe also check all datasets again, and keep only the best ones to test on
        
        #random.shuffle(ds)

        ds = BCI2017.create_dataset_splits(ds, train_subjects, valid_subj, test_subj, random_pretrain_subjects)
        with open("Datasets/bci2017/" + "ds.pkl", "wb") as f:
            pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def elim_bad_trials_deprecated(left_hand_trials, right_hand_trials, bad_trials, hand):
        left_hand_bad_trials = []
        right_hand_bad_trials = []
        bad_trials_voltage = bad_trials['bad_trial_idx_voltage'].item()
        #bad_trials_emg = eeg_data[14]['bad_trial_idx_mi'].item()

        assert bad_trials_voltage.shape == (1, 2)
        
        lh = bad_trials_voltage[0][0]
        rh = bad_trials_voltage[0][1]

        def align_indices(idx):
            idx -= 1

        if lh.shape != (0, 0):
            lh = lh[0]
            left_hand_bad_trials.extend(lh)
            #align_indices(lh)
            #left_hand_trials = np.delete(left_hand_trials, lh, axis=0)

        if rh.shape != (0, 0):
            rh = rh[0]
            right_hand_bad_trials.extend(rh)
            #align_indices(rh)
            #right_hand_trials = np.delete(right_hand_trials, rh, axis=0)

        bad_trials_emg = bad_trials['bad_trial_idx_mi'].item()
        assert bad_trials_emg.shape == (1, 2)
        
        lh = bad_trials_emg[0][0]
        rh = bad_trials_emg[0][1]

        assert lh.shape[1] == 1 and rh.shape[1] == 1

        if lh.shape[0] != 0:
            lh = np.squeeze(lh, axis = 1)
            left_hand_bad_trials.extend(lh)
        #    align_indices(lh)
        #    left_hand_trials = np.delete(left_hand_trials, lh, axis=0)

        if rh.shape[0] != 0:
            rh = np.squeeze(rh, axis = 1)
            #print(rh)
            right_hand_bad_trials.extend(rh)
            #align_indices(rh)
            #right_hand_trials = np.delete(right_hand_trials, rh, axis=0)


        if lh.shape != (0, 0):
            pass
            #print(lh, lh.shape)
            #lh = lh[0]
            #align_indices(lh)
            #left_hand_trials = np.delete(left_hand_trials, lh, axis=0)

        #if rh.shape != (0, 0):
        #    rh = rh[0]
        #    align_indices(rh)
        #    right_hand_trials = np.delete(right_hand_trials, rh, axis=0)

        print(left_hand_bad_trials)     #TODO append all bad trails for left hand for a subject, then -1 then remove the trials
        print(right_hand_bad_trials)
        left_hand_bad_trials = np.unique(left_hand_bad_trials)
        right_hand_bad_trials = np.unique(right_hand_bad_trials)
        print(left_hand_bad_trials)     #TODO append all bad trails for left hand for a subject, then -1 then remove the trials
        print(right_hand_bad_trials)
        
        if len(left_hand_bad_trials) > 45:     # if more than 50 bad trials then remove subject's specific hand data
            left_hand_trials = []
        elif len(left_hand_bad_trials) > 0:
            left_hand_bad_trials = [x - 1 for x in left_hand_bad_trials]
            left_hand_trials = np.delete(left_hand_trials, left_hand_bad_trials, axis=0)
        
        if len(right_hand_bad_trials) > 45:    # if more than 50 bad trials then remove subject's specific hand data
            right_hand_trials = []
        elif len(right_hand_bad_trials) > 0:
            right_hand_bad_trials = [x - 1 for x in right_hand_bad_trials]
            right_hand_trials = np.delete(right_hand_trials, right_hand_bad_trials, axis=0)
        return left_hand_trials, right_hand_trials

        

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