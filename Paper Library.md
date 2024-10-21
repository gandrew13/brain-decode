# Reviewed

## Datasets
https://arxiv.org/abs/2406.07151 - EEG-IamgeNet dataset, biggest I have found for mental image classification.
https://arxiv.org/abs/2404.05553 - Alljoined1 dataset (64 channels, 8 subjects)

## Models
https://arxiv.org/abs/2311.03764 - Neuro-GPT (EEGConformer + GPT3), pretrained on TUH, didn't have good results, probably because mental image classification data (EEGImageNet, Alljoined1) have very short EEG length (300 - 500 points)
https://arxiv.org/pdf/2010.11929 : ViT
https://arxiv.org/abs/2106.08254 : BeiT, BERT pre-training of image transformers, uses learned visual codebook learned based on a vector quantizer 
https://arxiv.org/abs/2208.06366 : BeiTv2, 
LaBraM  : 



## Other



# To Review

## Datasets
https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub  : EEG dataset for human visual object recognition
https://archive.ics.uci.edu/dataset/121/eeg+database                            : EEG correlates of genetic predisposition to alcoholism (64 channels, 256Hz)
https://mindbigdata.com/                                                        : Single-subject EEG data for image classification
https://openbci.com/community/publicly-available-eeg-datasets/                  : Publicly Available EEG Datasets



## Models
https://openreview.net/forum?id=QzTpTRVtrP              :  LaBraM
https://arxiv.org/abs/2404.14869                        :  EEGEncoder: Advancing BCI with Transformer-Based Motor Imagery Classification
https://arxiv.org/abs/2409.00101                        :  NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals
https://ieeexplore.ieee.org/abstract/document/9387339   :  Automatic Detection of Motor and Mental Imagery EEG Signals



## Surveys
https://www.arxiv.org/abs/2410.08224                                                        : A Survey of Spatio-Temporal EEG data Analysis: from Models to Applications
https://www.sciencedirect.com/science/article/pii/S0925231224011251                         : Survey on cross-subject, cross-session EEG emotion recognition
https://www.sciencedirect.com/science/article/pii/S0168010224000750#bib122                  : Foundation models and generative AI for neuroscience
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1203104/full  : Generative AI for brain image computing and brain network computing
https://link.springer.com/article/10.1007/s00521-021-06352-5                                : EEG DL on motor imagery (2021)
https://pmc.ncbi.nlm.nih.gov/articles/PMC7664219/                                           : EEG TL for BCI (2020)
https://www.sciencedirect.com/science/article/pii/S093336572300252X                         : EEG DL on MI (motor imagery), 2024
https://www.sciencedirect.com/science/article/pii/S0925231220314223                         : EEG TL, 2021
https://arxiv.org/abs/1907.01332                                                            : 



## Others (must read)
https://arxiv.org/pdf/2403.15415                                            : EEG domain adaptation
https://www.mdpi.com/1424-8220/19/13/2999                                   : Inter-subject modeling for EEG affect recognition
https://arxiv.org/abs/2309.04153                                            : Mapping EEG Signals to Visual Stimuli, 2023
https://www.sciencedirect.com/science/article/pii/S1046202321001018	        : Cross-subject EEG-based driver states awareness recognition
https://pubmed.ncbi.nlm.nih.gov/2791314/                                    : Inter- vs intra-subject variance in topographic mapping of the EEG
https://arxiv.org/abs/2007.06407                                            : Deep Cross-Subject Mapping of Neural Activity
https://arxiv.org/abs/2403.15415                                            : Physics-informed and Unsupervised Riemannian Domain Adaptation for Machine Learning on Heterogeneous EEG Datasets
https://iopscience.iop.org/article/10.1088/1741-2552/aaf3f6/meta            : Inter-subject TL using EEG for BCI (2019)
https://ieeexplore.ieee.org/abstract/document/8786636                       : TL MI (motor imagery) using CNNs, 2019
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274847   : EEG classification + reconstruction, 2022
https://cdn.aaai.org/ojs/6817/6817-13-10046-1-10-20200524.pdf               : Cross-Aligned Latent Embeddings for Zero-Shot Cross-Modal Retrieval
https://arxiv.org/abs/2111.06377                                            : Masked Autoencoder are Scalable Vision Learners
https://link.springer.com/article/10.1007/s00521-022-08178-1                : NeuroGAN (attention-based GAN for EEG image reconstruction, good for embeddings maybe), end of 2022
https://paperswithcode.com/sota/image-classification-on-imagenet            : SotA on ImageNet (try to apply on EEG/fMRI)
https://paperswithcode.com/sota/image-generation-on-imagenet-256x256        : SotA on ImageNet image reconstruction (try to apply on EEG/fMRI, also use DALLE and others, VAEs, diffusion, ViT, multimodal models, CNN+Transformer etc)
https://upcommons.upc.edu/bitstream/handle/2117/109756/Personalized-Image-Classification-of-EEG-Signals-using-Deep-Learning.pdf  : Image classification using simple LSTMs


## Others (optional)

