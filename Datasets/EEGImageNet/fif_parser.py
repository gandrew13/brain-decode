import mne
import matplotlib.pyplot as plt




def plot_montage(filename): # EEGImageNet
    montage = mne.channels.read_dig_fif(filename)

    montage.ch_names[-2] = 'EEG068' # last but one channel name is wrong in the montage file

    new_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2", "HEO", "VEO", "EKG", "EMG", "Trigger"]

    tuples = [(key, value) for i, (key, value) in enumerate(zip(montage.ch_names, new_channels))]

    montage.rename_channels(dict(tuples))
    
    #plt.figure(1)
    montage.plot()
    #plt.show()


def plot_alljoined1_montage(filenames):
    print(filenames[0])
    montage = mne.io.read_raw_fif(filenames[0], preload=True)
    
    tmp = mne.channels.make_standard_montage('standard_1020')
    montage.set_montage(tmp)
    montage.set_eeg_reference('average', projection=False)
    print(montage.ch_names)
    print(len(montage.ch_names))

    montage = montage.get_montage()
    #plt.figure(2)
    montage.plot()
    plt.show()
    #print(montage.get_data().shape)
    
    # check if there is any different montage
    #montages = []
    #for f in filenames: 
    #    montage = mne.io.read_raw_fif(f)
    #    montages.append(''.join(montage.ch_names))
    #print(montages)
    #print(any(montages.count(x) != len(montages) for x in montages))
    




plot_montage("../Datasets/EEGImageNet/montage.fif")
fif_files = ["../Datasets/Alljoined1/subj01_session1_eeg.fif",
            "../Datasets/Alljoined1/subj01_session2_eeg.fif",
            "../Datasets/Alljoined1/subj02_session1_eeg.fif",
            "../Datasets/Alljoined1/subj03_session1_eeg.fif",
            "../Datasets/Alljoined1/subj03_session2_eeg.fif",
            "../Datasets/Alljoined1/subj04_session1_eeg.fif",
            "../Datasets/Alljoined1/subj04_session2_eeg.fif",
            "../Datasets/Alljoined1/subj05_session1_eeg.fif",
            "../Datasets/Alljoined1/subj05_session2_eeg.fif",
            "../Datasets/Alljoined1/subj06_session1_eeg.fif",
            "../Datasets/Alljoined1/subj06_session2_eeg.fif",
            "../Datasets/Alljoined1/subj07_session1_eeg.fif",
            "../Datasets/Alljoined1/subj08_session1_eeg.fif"]
#plot_alljoined1_montage("../Datasets/Alljoined1/subj01_session1_eeg.fif")
plot_alljoined1_montage(fif_files)
