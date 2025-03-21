
from model_wrapper import *
from models.multi_task_model import MultiTaskModel


class MultiTaskAccuracy(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.acc_task = BinaryAccuracy()
        self.acc_subjects = Accuracy(task="multiclass", num_classes=num_classes[1])

    def forward(self, x):
        pass


class LMultiTaskModelWrapper(LModelWrapper):
    '''
    PyTorch Lightning wrapper class for multi task model and training. 
    '''
    def __init__(self, model, pretrained_model, freeze_model, fine_tune_mode, num_classes, num_channels, final_fc_length):
        #super().__init__(model, pretrained_model, freeze_model, fine_tune_mode, num_classes[0], num_channels, final_fc_length)
        L.LightningModule.__init__(self)

        self.__pretrained_model = pretrained_model
        self.train_simultaneously = True

        if self.train_simultaneously:
            print("Simultaneous training")

        if pretrained_model == None and not self.train_simultaneously:                # pretraining phase
            self.automatic_optimization = False

        self.model = None
        match model:
            case 'multi_task':
                self.model = MultiTaskModel(num_classes, num_channels, final_fc_length=final_fc_length, add_log_softmax=False, is_pretrained=pretrained_model)
            case _:
                print("Error: Unknown model!")

        if self.__pretrained_model:
            print("Loading pretrained model from: ", self.__pretrained_model)
            self.__checkpoint = load(self.__pretrained_model)
            if "model" in self.__checkpoint:
                self.load_weights(self.__checkpoint["model"])
            else:
                self.load_weights(self.__checkpoint)

        if freeze_model:
            self.freeze(fine_tune_mode)

        #self.task_loss = CrossEntropyLoss()
        self.task_loss = BCEWithLogitsLoss()
        self.subject_loss = CrossEntropyLoss()

        #self.eval_task_loss = CrossEntropyLoss()
        self.eval_task_loss = BCEWithLogitsLoss()
        self.eval_subject_loss = CrossEntropyLoss()

        self.train_accuracy = MultiTaskAccuracy(num_classes)
        self.valid_accuracy = MultiTaskAccuracy(num_classes)
        self.test_accuracy = MultiTaskAccuracy(num_classes)

        #self.epoch_loss = 0.0
        self.epoch_loss = [0.0, 0.0]

        self.prev_losses = {"prev_train_loss": self.__checkpoint["prev_train_loss"] if pretrained_model and "prev_train_loss" in self.__checkpoint else [100000.0, 100000.0], "prev_eval_loss": 100000.0}

        self.my_log_dict = {"epoch": -1, "train_loss": 1000.0, "eval_loss": 1000.0, "train_acc": -1.0, "valid_acc": -1.0, "test_acc": -1.0}

        self.batch_logits = []
        self.batch_labels = []

        self.best_val_acc = -1.0

    def training_step(self, batch, batch_idx):
        assert self.model.training
        self.step(batch, batch_idx)
        
        task_logits = self.batch_logits[-1][0].squeeze(-1)
        task_labels = self.batch_labels[-1][0].type(cuda.FloatTensor)
        loss = self.task_loss(task_logits, task_labels)                # task loss
        self.epoch_loss[0] += loss.item()
        if self.__pretrained_model == None:
            subject_logits = self.batch_logits[-1][1]
            subject_labels = self.batch_labels[-1][1]
            subject_loss = self.subject_loss(subject_logits, subject_labels)    # subject loss
            self.epoch_loss[1] += subject_loss.item()     
            loss += subject_loss

        #print("Train step task loss: ", round(task_loss.item(), 3))
        #print("Train step subject loss: ", round(subject_loss.item(), 3))

        #self.epoch_loss += loss.item()
        return loss
    
    def validation_step(self, batch, batch_idx):
        assert not self.model.training
        self.step(batch)

        eval_loss = self.eval_task_loss(self.batch_logits[-1][0].squeeze(-1), self.batch_labels[-1][0].type(cuda.FloatTensor))              # eval task loss
        #print("Eval step task loss: ", round(eval_loss.item(), 3))
        self.epoch_loss[0] += eval_loss.item()
        if self.__pretrained_model == None:
            eval_subject_loss = self.eval_subject_loss(self.batch_logits[-1][1], self.batch_labels[-1][1])
            self.epoch_loss[1] += eval_subject_loss.item()
            #eval_loss += eval_subject_loss                                                  # eval subject loss
            #print("Eval step subject loss: ", round(eval_subject_loss.item(), 3))

        #self.epoch_loss += eval_loss.item()

    def step(self, batch, batch_idx = None):
        '''
        Process (forward propagate) a batch.
        '''
        inputs, labels = batch
        inputs, labels = inputs.type(cuda.FloatTensor), [labels[0].type(cuda.LongTensor), labels[1].type(cuda.LongTensor)]
        alpha = 1.0 # default
        if self.training:
            ds_size = self.trainer.num_training_batches
            p = float(batch_idx + self.current_epoch * ds_size) / self.trainer.max_epochs / ds_size
            import numpy as np
            alpha = 2. / (1. + np.exp(-0.1 * p)) - 1
            #alpha = 0.1

            if self.current_epoch % 10 == 0 and batch_idx == self.trainer.num_training_batches - 1:
                print("Alpha: ", alpha)

        if self.train_simultaneously:
            out = self.step_simult(inputs, labels, alpha)
        else:
            out = self.step_alt(inputs, labels, alpha)

        #out = self.model(inputs, alpha)

        self.batch_logits.append(out)
        self.batch_labels.append(labels)

    def step_simult(self, inputs, labels, alpha = -1.0):
        # zero the gradients
        #opt = self.optimizers()
        #opt.zero_grad()

        # forward pass
        features = self.model.feature_extractor(inputs)
        features = features.contiguous().view(features.size(0), -1)
    
        out_task = self.model.label_classifier(features)
        out_subject = None
        if self.__pretrained_model == None:
            features = self.model.GRL(features, alpha)
            out_subject = self.model.subject_classifier(features)

        # compute loss
        #task_logits = out_task.squeeze(-1)
        #task_labels = labels[0].type(cuda.FloatTensor)
        #loss = self.task_loss(task_logits, task_labels)                         # task loss
        #self.epoch_loss[0] += loss.item()
        #if self.__pretrained_model == None:
        #    subject_logits = out_subject
        #    subject_labels = labels[1].type(cuda.FloatTensor)
        #    subject_loss = self.subject_loss(subject_logits, subject_labels)    # subject loss
        #    self.epoch_loss[1] += subject_loss.item()     
        #    loss += subject_loss

        # backprop
        #self.manual_backward(loss)
        #opt.step()

        return out_task, out_subject    # for accuracy computing

    def step_alt(self, inputs, labels, alpha = -1.0):
        # zero the gradients
        opt_task, opt_subjects = self.optimizers()

        ### train (subject/domain) discriminator
        if self.__pretrained_model == None and self.model.training:
            opt_subjects.zero_grad()
            
            # enable/disable gradients
            self.model.feature_extractor.requires_grad_(False)
            self.model.label_classifier.requires_grad_(False)
            self.model.GRL.requires_grad_(True)
            self.model.subject_classifier.requires_grad_(True)

            # forward pass
            features = self.model.feature_extractor(inputs)
            features = features.contiguous().view(features.size(0), -1)
            features = self.model.GRL(features, alpha)
            out_subject = self.model.subject_classifier(features)

            # loss compute
            subject_labels = labels[1]
            subject_loss = self.subject_loss(out_subject, subject_labels)    # subject loss
            #self.epoch_loss[1] += subject_loss.item()

            # backward pass
            self.manual_backward(subject_loss)
            opt_subjects.step()

        ### train encoder and label (task) classifier
        if self.model.training:
            opt_task.zero_grad()

            # enable/disable gradients
            self.model.feature_extractor.requires_grad_(True)
            self.model.label_classifier.requires_grad_(True)
            self.model.GRL.requires_grad_(True)
            self.model.subject_classifier.requires_grad_(False)  #TODO: should I freeze the subject classifier or not

        # forward pass
        features = self.model.feature_extractor(inputs)
        features = features.contiguous().view(features.size(0), -1)
        out_task = self.model.label_classifier(features)
        features = self.model.GRL(features, alpha)
        out_subject = self.model.subject_classifier(features)     

        # loss compute
        if self.model.training:
            task_labels = labels[0].type(cuda.FloatTensor)
            subject_labels = labels[1]
            task_loss = self.task_loss(out_task.squeeze(-1), task_labels)                # task loss
            subject_loss = self.subject_loss(out_subject, subject_labels)    # subject loss
            loss = task_loss + subject_loss
            #self.epoch_loss[0] += task_loss.item()
            
            # backward pass
            self.manual_backward(loss)
            opt_task.step()

        return out_task, out_subject    # for accuracy computing

    def configure_optimizers(self):
        if self.automatic_optimization:
            opt = optim.AdamW(self.model.parameters(), lr=1e-4)     # best for training on 2017 so far
            #opt = optim.AdamW(self.model.parameters(), lr=1e-5)
            #opt = optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
            return opt
        else:
            opt_task, opt_subjects = optim.AdamW(self.model.parameters(), lr=1e-4), optim.AdamW(self.model.parameters(), lr=1e-4)
            return opt_task, opt_subjects
        #try:
        #    if self.__pretrained_model and "optimizer" in self.__checkpoint:
        #        opt.load_state_dict(self.__checkpoint["optimizer"])        # TODO: Re-enable this, load optimizer state
        #except:
        #    print("Warning: Couldn't load pretrained optimizer.")
        #return opt
        #return {
        #    "optimizer": opt, "lr_scheduler": {"scheduler": optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 10), "interval": "epoch", "frequency": 1, "monitor": "train_loss"}
        #}
        #return optim.SGD(self.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
        #return optim.SGD(self.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)

    def compute_accuracy(self, metric):
        self.model.eval()
        task_logits = torch.cat([torch.sigmoid(logit[0].squeeze(-1)) for logit in self.batch_logits])

        if self.__pretrained_model == None:
            subject_logits = torch.cat([logit[1] for logit in self.batch_logits])
            subject_logits = torch.argmax(torch.softmax(subject_logits, dim=1), dim=1)
        
        task_labels = torch.cat([logit[0] for logit in self.batch_labels])

        if self.__pretrained_model == None:
            subject_labels = torch.cat([logit[1] for logit in self.batch_labels])

        self.batch_logits = []
        self.batch_labels = []

        acc = round(metric.acc_task(task_logits, task_labels).item(), 3)
        print("Task acc: ", acc)
        if self.__pretrained_model == None:
            print("Subjects acc: ", round(metric.acc_subjects(subject_logits, subject_labels).item(), 3))
            acc += round(metric.acc_subjects(subject_logits, subject_labels).item(), 3)    # don't use subject accuracy to save best accuracy model, use only classification accuracy. Or maybe I should use only best subject accuracy.

        return acc
    
    def load_weights(self, state_dict):
        own_state = self.model.state_dict()
        for name, param in state_dict.items():
            print(name, param.shape)
            try:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
            except:
                print("Couldn't load: ", name)
    
    def freeze(self, fine_tune_mode):
        match fine_tune_mode:
            case 5:
                for name, params in self.model.feature_extractor.named_parameters():
                    print("Frozen: ", name)
                    params.requires_grad = False
            case _:
                return # do nothing
