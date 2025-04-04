from torch import nn, autograd, tensor
import braindecode.models as models



class GradientReversalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, tensor(alpha, requires_grad=False))
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = (alpha * grad_output.neg())
        if alpha < 0.0:
            raise ValueError("Alpha is negative.")
        return grad_input, None
    
class GradientReversal(nn.Module):
    def __init__(self, alpha=-1.0):
        super().__init__()
        self.alpha = tensor(alpha, requires_grad = False)

    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)
        

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, num_channels, final_fc_length, add_log_softmax=False, is_pretrained=False, reconstruct=False):
        super().__init__()

        self.__is_pretrained = is_pretrained

        self.feature_extractor = models.EEGConformer(num_classes[0], num_channels, final_fc_length=final_fc_length, add_log_softmax=False) # num_classes not really used cause we only get the features from the model, we don't care about the output classes
        
        if reconstruct:     # WIP: use reconstruction loss
            self.feature_extractor.patch_embedding = nn.Identity()
            self.patch_embedding = models.eegconformer._PatchEmbedding(n_filters_time=40, filter_time_length=25, n_channels=num_channels, pool_time_length=75, stride_avg_pool=15, drop_prob=0.5)


        # keep only the feature extractor and create different classification heads
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.final_layer = nn.Identity()
        self.label_classifier = nn.Sequential(nn.Linear(final_fc_length, 256),
                                              nn.ELU(),
                                              nn.Dropout(0.5),
                                              nn.Linear(256, 32),
                                              nn.ELU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(32, 1))                   # final classification layer, one neuron for binary classification
                                              #nn.Linear(32, num_classes[0]))     # final classification layer
        
        if self.__is_pretrained == None:
            self.GRL = GradientReversal(alpha=1.0)
            self.subject_classifier = nn.Sequential(          # TODO I modified this !!! I should see when to stop training and save the best model (what does best model mean, best classification of validation labels?)
                                                    nn.Linear(final_fc_length, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.5),
                                                    nn.Linear(256, 32),
                                                    nn.ELU(),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(32, num_classes[1]))     # final classification layer, 52 subjects or 2, for 2 source datasets             

    #def forward(self, x, alpha):
    #    features = self.feature_extractor(x)
    #
    #    features = features.contiguous().view(features.size(0), -1)
    #
    #    out_task = self.label_classifier(features)
    #    out_subject = None
    #    if self.__is_pretrained == None:
    #        features = self.GRL(features, alpha)
    #        out_subject = self.subject_classifier(features)
    #
    #    return out_task, out_subject
