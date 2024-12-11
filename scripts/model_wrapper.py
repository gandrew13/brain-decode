

from torch import optim, cuda, save, load
from torch.nn import CrossEntropyLoss
import lightning as L
from torchmetrics.classification import Accuracy
import braindecode.models as models
import torch

class LModelWrapper(L.LightningModule):
    '''
    PyTorch Lightning wrapper class. 
    '''
    def __init__(self, model, pretrained_model, freeze_model, fine_tune_mode, num_classes, num_channels, final_fc_length):
        super().__init__()

        match model:
            case "eegconformer":
                self.model = models.EEGConformer(num_classes, num_channels, final_fc_length=final_fc_length, add_log_softmax=False) # we use CrossEntropy loss, so no need to add a LogSoftmax layer
            case "eegnet":
                self.model = models.EEGNetv4(num_channels, num_classes, 400)
            case "mlp":
                self.model = torch.nn.Sequential(torch.nn.Flatten(),
                                                 torch.nn.Linear(6600, 256),
                                                 torch.nn.BatchNorm1d(256),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Dropout1d(0.5),
                                                 torch.nn.Linear(256, 128),
                                                 torch.nn.BatchNorm1d(128),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Dropout1d(0.5),
                                                 torch.nn.Linear(128, num_classes))
            case _:
                print("Error: Unknown model!")

        if pretrained_model:
            self.model.load_state_dict(load(pretrained_model))

        if freeze_model:
            self.freeze()

        #for name, params in self.model.named_parameters():
        #    if "fc" not in name:
        #       params.requires_grad = False

        self.loss = CrossEntropyLoss()
        self.eval_loss = CrossEntropyLoss()
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.epoch_loss = 0.0

        self.prev_losses = {"prev_train_loss": 10000.0, "prev_eval_loss": 100000.0}

        #self.batch_results = {"train_batch_logits": [], "train_batch_labels": [], "valid_batch_logits": [], "valid_batch_labels": []}

        self.my_log_dict = {"epoch": -1, "train_loss": 1000.0, "eval_loss": 1000.0, "train_acc": -1.0, "eval_acc": -1.0, "test_acc": -1.0}

        self.batch_logits = []
        self.batch_labels = []


    def training_step(self, batch, batch_idx):
        assert self.model.training
        self.step(batch)
        loss = self.loss(self.batch_logits[-1], self.batch_labels[-1])

        self.epoch_loss += loss.item()
        #self.my_log(loss.item())

        return loss
    
    def validation_step(self, batch, batch_idx):
        assert not self.model.training
        self.step(batch)
        loss = self.eval_loss(self.batch_logits[-1], self.batch_labels[-1])
        self.epoch_loss += loss.item()

    def test_step(self, batch, batch_idx):
        assert not self.model.training
        self.step(batch)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)
        #return optim.SGD(self.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
        #return optim.SGD(self.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)
    
    def my_log(self, values_dict):
        self.log_dict(values_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.current_epoch % 10 == 0:
            pass
            #self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            #self.log("acc", self.accuracy(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # save model

    def on_train_epoch_end(self):
        # This executes after validation_step and on_validation_epoch_end
        self.my_log(self.my_log_dict)

    def on_validation_epoch_start(self):
        mean_loss = round(self.epoch_loss / self.trainer.num_training_batches, 3)
        print("\nMean loss:", mean_loss)
        self.my_log_dict["epoch"] = int(self.current_epoch)
        self.my_log_dict["train_loss"] = mean_loss
        self.save_model()

        self.epoch_loss = 0.0
        
        if self.current_epoch == (self.trainer.max_epochs - 1):
            # compute accuracy on last epoch
            self.my_log_dict["train_acc"] = self.compute_accuracy(self.train_accuracy)
            print("\nTrain Accuracy:", self.my_log_dict["train_acc"])
            
        self.batch_logits = []
        self.batch_labels = []


    def on_validation_epoch_end(self):
        assert not self.model.training
        self.my_log_dict["valid_acc"] = self.compute_accuracy(self.valid_accuracy)
        print("\nValidation Accuracy:", self.my_log_dict["valid_acc"], "\n")
        mean_loss = round(self.epoch_loss / self.trainer.num_val_batches[0], 3)
        self.my_log_dict["eval_loss"] = mean_loss
        self.epoch_loss = 0.0

    def on_test_epoch_end(self):
        assert not self.model.training
        #if self.current_epoch == (self.trainer.max_epochs - 1):        
        self.my_log_dict["test_acc"] = self.compute_accuracy(self.test_accuracy)
        print("Test Accuracy:", self.my_log_dict["test_acc"])

    def step(self, batch):
        '''
        Process (forward propagate) a batch.
        '''
        inputs, labels = batch
        inputs, labels = inputs.type(cuda.FloatTensor), labels.type(cuda.LongTensor)
        out = self.model(inputs)
        self.batch_logits.append(out)
        self.batch_labels.append(labels)

    def compute_accuracy(self, metric):
        self.model.eval()
        logits = torch.argmax(torch.cat(self.batch_logits), dim=1)
        labels = torch.cat(self.batch_labels)

        self.batch_logits = []
        self.batch_labels = []

        return round(metric(logits, labels).item(), 3)
    
    def freeze(self, fine_tune_mode):
        excluded_layers = [] 
        match fine_tune_mode:
            case 1:
                self.freeze()
            case 2:
                excluded_layers = ["final_layer"]
            case 3:
                excluded_layers = ["final_layer", "fc"]
            case _:
                return # do nothing
            
        for name, params in self.model.named_parameters():
                should_freeze = [layer_type in name for layer_type in excluded_layers]
                if len(should_freeze) == 0:
                    params.requires_grad = False
                else:
                    print(name)

    def save_model(self):
        if self.current_epoch % 5 == 0:
            if self.prev_losses["prev_train_loss"] > self.epoch_loss:
                print("Saving model...")
                save(self.model.state_dict(), self.logger.log_dir + "/best_train_loss.pth")
                self.prev_losses["prev_train_loss"] = self.epoch_loss
