import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


# Not used anymore.
class EEGImageNetDataset(Dataset):
    def __init__(self, args, transform=None):
        self.dataset_dir = args.dataset_dir
        self.transform = transform
        
        print("Loading the data from ", self.dataset_dir, "...")
        loaded = torch.load(os.path.join(self.dataset_dir, "EEG-ImageNet_full.pth"))

        if loaded:
            print("Yaay, dataset loaded successfully!")
        else:
            print("Ooops, couldn't load the data!")
            return

        print(loaded["dataset"][10]["eeg_data"].size())
        print(loaded["dataset"][10]["granularity"])
        print(loaded["dataset"][10]["subject"])
        print(loaded["dataset"][10]["label"])
        print(loaded["dataset"][10]["image"])

        self.labels = loaded["labels"]
        self.images = loaded["images"]
        if args.subject != -1:
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == args.subject]
        else:
            chosen_data = loaded['dataset']
        if args.granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif args.granularity == 'all':
            self.data = chosen_data
        else:
            fine_num = int(args.granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
        
        self.use_image_label = False

    def __getitem__(self, index):
        if self.use_image_label:
            path = self.data[index]["image"]
            label = Image.open(os.path.join(self.dataset_dir, "imageNet_images", path.split('_')[0], path))
            if label.mode == 'L':
                label = label.convert('RGB')
            if self.transform:
                label = self.transform(label)
            else:
                label = path
        else:
            label = self.labels.index(self.data[index]["label"])

        eeg_data = self.data[index]["eeg_data"].float()
        feat = eeg_data[:, 40:440]
        
        return feat, label

    def __len__(self):
        return len(self.data)
        
    def merge_2_part_ds(self, dataset_path: str) -> None:
        '''
        The original dataset was split into 2 parts. This method reads the 2 parts and merges them.
        '''

        ds_part1 = torch.load(os.path.join(dataset_path, "EEG-ImageNet_1.pth"))
        ds_part2 = torch.load(os.path.join(dataset_path, "EEG-ImageNet_2.pth"))

        if(ds_part1["labels"] != ds_part2["labels"]):
            print("Error: The labels of the 2 parts of the dataset do not match.")
            return
        
        if(ds_part1["images"] != ds_part2["images"]):
            print("Error: The image labels of the 2 parts of the dataset do not match.")
            return
        
        ds = {}
        ds["labels"] = ds_part1["labels"]
        ds["images"] = ds_part1["images"]
        ds["dataset"] = ds_part1["dataset"] + ds_part2["dataset"]

        torch.save(ds, dataset_path + "EEG-ImageNet_full.pth")