
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from .eegdataset import EEGDataset, EEGDataModule, np
import datasets
import json

class Alljoined1(EEGDataset):
    def __init__(self, data, labels):

        data = [elem for elem in data if elem['subject_id'] == 1]
        data = np.array(data)
        #data = [elem['subject_id'] for elem in data]
        #data = list(set(data))
        #print(data)
        super().__init__(data,
                         labels,
                         chans_order=['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'], # removed 'Iz' and 'Status' channels)
                         chans_to_keep=["O1", "OZ", "O2", "PO7", "PO3", "POZ", "PO4", "PO8"]) # 8 channels, keep only the occipital lobe channels, since they are related to the visual cortex
    
        print("Loading the data...")
        print(self._data[0].keys())
        print("Number of samples:", self._data.shape[0])
        print("Number of EEG channels:", len(self._data[0]['EEG']))
        print("EEG sample length:", len(self._data[0]['EEG'][0]))
        print("Number of classes:", len(list(set(self._labels.values()))))

    def __getitem__(self, index):
        eeg = self._data[index]['EEG']
        image_id = self._data[index]['73k_id']

        # TODO: reorder channels for TL, they should be in same order in both datasets
        
        return self._preprocess_sample(eeg, normalize=True), self._labels[image_id]
    
    def get_num_classes(self):
        return len(list(set(self._labels.values())))

    def get_final_fc_length(self):
        return 640 # TODO: don't hardcode this, compute it based on the transformer output
    
    @staticmethod
    def setup(num_chunks: int = 1, chunk_len: int = 200, single_label = True):
        ds = datasets.load_dataset("Alljoined/05_125") # dataset on HuggingFace

        # process the labels
        with open("Datasets/Alljoined1/captions_and_categories.json") as f:
            labels_json = json.load(f)

        # make a list of all supercategories
        categories = []
        labels = {}
        for entry in labels_json:
            image_categories_list = entry["categories"] # the categories an image belongs to
            image_cats = [cat["supercategory_name"] for cat in image_categories_list]
            categories.extend(image_cats)
            labels[int(entry["nsdId"]) - 1] = set(image_cats) # in the HuggingFace dataset the NSD ID (73k_id) is saved as nsd_id - 1, so I have to subtract 1 here as well, to match IDs

        # eliminate the duplicates
        categories = list(set(categories))
        onehot_encoding = Alljoined1.onehot_encode(categories)

        # make dict = {image_id: onehot_encoding}
        if single_label:
            for image_id, cats in labels.items():
                labels[image_id] = categories.index(list(cats)[0])
        else:
            for image_id, cats in labels.items():
                labels[image_id] = sum([onehot_encoding[cat] for cat in cats])

        return EEGDataModule(Alljoined1(ds["train"], labels), Alljoined1(ds["test"], labels), None)
    
    @staticmethod
    def onehot_encode(categories_list: list) -> dict[str, list]:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(categories_list)
        
        # binary encode
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        onehot_encoded_dict = {}
        for el in onehot_encoded:
            category_str = label_encoder.inverse_transform([np.argmax(el[:])])
            onehot_encoded_dict[category_str[0]] = el
        return onehot_encoded_dict

