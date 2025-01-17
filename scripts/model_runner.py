import csv
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, manual_seed, default_generator, backends
from lightning.pytorch.loggers import CSVLogger


from eeg_datasets.alljoined1 import Alljoined1
from eeg_datasets.eegimagenet import EEGImageNetDataset
from eeg_datasets.bci_iv_2a import BCIIV2a
from scripts.eeg_datasets.bci2017 import BCI2017
from scripts.eeg_datasets.bci2019 import BCI2019
from model_wrapper import L, LModelWrapper

class Runner:
    def __init__(self, args):
        super().__init__()

        self.__args = args

        self.dataset_loader = self.setup_dataset()
        self.model = LModelWrapper(self.__args.model, self.__args.load_pretrained, self.__args.freeze_model == 'True', int(self.__args.fine_tune_mode), self.dataset_loader.get_num_classes(), self.dataset_loader.get_num_chans(), self.dataset_loader.get_final_fc_length())        

        if self.__args.deterministic == 'True':
            self.seed()

    def run(self):
        experiment_name = "%s_%s" % (self.__args.dataset, self.__args.model)
        trainer = L.Trainer(max_epochs=int(self.__args.epochs), logger=CSVLogger("scripts/logs",  experiment_name), num_sanity_val_steps=0, enable_checkpointing=False, deterministic=(self.__args.deterministic == 'True'))
        trainer.fit(model=self.model, train_dataloaders=self.dataset_loader)
        trainer.test(model=self.model, dataloaders=self.dataset_loader)
        self.plot(self.__args.plot_file)

    def setup_dataset(self):
        match self.__args.dataset:
            case "eegimagenet":
                return EEGImageNetDataset.setup(self.__args.dataset_path)
            case "alljoined1":
                return Alljoined1.setup()
            case "bci_iv_2a":
                return BCIIV2a.setup(self.__args.dataset_path)
            case "bci2017":
                return BCI2017.setup(self.__args.dataset_path, self.__args.train_subjects, self.__args.random_pretrain_subjects, self.__args.valid_subject, self.__args.test_subject, self.__args.batch_size)
            case "bci2019":
                return BCI2019.setup(self.__args.dataset_path, self.__args.batch_size)
            case _:
                print("Error: Unknown dataset!")

    def plot(self, file):
        if not file:
            return
        with open(file, mode ='r') as f:
            data = csv.reader(f)
            next(iter(data)) # skip header
            steps = np.array([], dtype=int)
            train_losses = np.array([])
            eval_losses = np.array([])
            for line in data:
                steps = np.append(steps, int(float(line[0])))
                train_losses = np.append(train_losses, float(line[3]))
                eval_losses = np.append(eval_losses, float(line[1]))

            plt.plot(steps, train_losses, label = "train loss")
            plt.plot(steps, eval_losses, label = "eval loss")

            plt.legend()
            plt.title('Figure 1 - Full model train/eval loss', y=-0.13)
            plt.show()

    def seed(self):
        seed = 42
        import random
        random.seed(seed)
        np.random.seed(seed)
        np.random.default_rng(seed)
        manual_seed(seed)
        default_generator.manual_seed(seed)
        backends.cudnn.benchmark = False
        backends.cudnn.deterministic = True
        L.seed_everything(42, workers=True)

    