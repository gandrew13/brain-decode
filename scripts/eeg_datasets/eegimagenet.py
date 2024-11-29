from .eegdataset import EEGDataset, EEGDataModule, np, torch



class EEGImageNetDataset(EEGDataset):
    def __init__(self, data):
        granularity = "coarse"
        
        #labels = ['n02124075', 'n02389026', 'n02607072', 'n02690373', 'n02906734', 'n02951358', 'n03063599', 'n03100240', 'n03180011', 'n03197337', 'n03272010', 'n03272562', 'n03297495', 'n03376595', 'n03445777', 'n03590841', 'n03773504', 'n03775071', 'n03792972', 'n04069434', 'n04120489', 'n07753592', 'n11939491', 'n13054560', 'n02099601', 'n02099712', 'n02106166', 'n02106550', 'n02110185', 'n02630281', 'n02643566', 'n07740461', 'n07745940', 'n07749192', 'n07756951', 'n07758680', 'n12144580', 'n02701002', 'n03384352', 'n04465666', 'n03495258', 'n04487394']

        if granularity == 'coarse':
            data = [i for i in data if i['granularity'] == 'coarse']
            #data = [i for i in data if i['label'] in labels]
        elif granularity == 'all':
            pass
        else:
            fine_num = int(granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            data = [i for i in data if i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]

        # one-hot encoding
        labels = []
        [labels.append(entry['label']) for entry in data if entry['label'] not in labels and entry['granularity'] == 'coarse']

        super().__init__(data,
                         labels,
                         chans_order=["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"],
                         #chans_to_keep=["O1", "OZ", "O2", "PO7", "PO3", "POZ", "PO4", "PO8"]) # 8 channels, keep only the occipital lobe channels, since they are related to the visual cortex
                         chans_to_keep=["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"],
                         ch_groups=[])  # temporal cortices
        
        ch_groups=[["FP1", "FPZ", "FP2", "AF3", "AF4"], # front head
                   ["F7", "F5", "F3", "F1"], ["FZ"], ["F2", "F4", "F6", "F8"],
                   ["FC5", "FC3", "FC1",], ["FCZ"], ["FC2", "FC4", "FC6"],
                   ["C5", "C3", "C1"], ["CZ"], [ "C2", "C4", "C6"],
                   ["CP5", "CP3", "CP1"], ["CPZ"], ["CP2", "CP4", "CP6", ],
                   ["P7", "P5", "P3", "P1",], ["PZ",], ["P2", "P4", "P6", "P8"],
                   ["PO7", "PO5", "PO3"], ["POZ"], ["PO4", "PO6", "PO8"],
                   ["CB1", "O1", "OZ", "O2", "CB2"], # back head
                   ["FT7", "T7", "TP7"], ["FT8", "T8", "TP8"]]
    
        print("Loading the data...")
        print(self._data[0].keys())
        print("Number of samples:", len(self._data))
        print("Number of EEG channels:", len(self._data[0]['eeg_data']))
        print("EEG sample length:", len(self._data[0]['eeg_data'][0]))
        print("Number of classes:", self.get_num_classes())

        #self.plot()        
        #exit()


    def __getitem__(self, index):
        eeg = self._data[index]['eeg_data']
        eeg = eeg[:, 40:440]
        eeg = eeg.numpy()

        label_idx = self._labels.index(self._data[index]['label'])
        if label_idx == -1:
            print("Error! Label not found!")
            return

        return self._preprocess_sample(eeg, normalize=True), label_idx
    
    def __len__(self):
        return len(self._data)
    
    def get_final_fc_length(self):
        return 840 # TODO: don't hardcode this, compute it based on the transformer output
    
    def plot(self):
        raw = torch.tensor(self._data[0]['eeg_data']).unsqueeze(0)
        super().plot(raw, 1000)

    @staticmethod
    def setup(dataset_path):
        ds = torch.load(dataset_path)
        if not ds:
            print("Ooops, couldn't load the data!")
            return

        # TODO: maybe split into train/val/test, not only train/test
        leave_out_subject = 7 # [0, 7]
        if leave_out_subject == -1:
            ds = EEGImageNetDataset(ds)

            train_set_perc = (80 / 100) * len(ds)
            test_set_perc = (20 / 100) * len(ds)
            train_data, test_data = torch.utils.data.random_split(ds, [int(train_set_perc), int(test_set_perc)])
        else:
            train_data = [ds['dataset'][i] for i in range(len(ds['dataset'])) if ds['dataset'][i]['subject'] != leave_out_subject]
            test_data = [ds['dataset'][i] for i in range(len(ds['dataset'])) if ds['dataset'][i]['subject'] == leave_out_subject]

        return EEGDataModule(EEGImageNetDataset(test_data), EEGImageNetDataset(test_data), EEGImageNetDataset(test_data), 64) # TODO: training on one subject (7). Change this back when needed.

