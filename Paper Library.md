# EEG

## Reviewed

### Datasets
https://arxiv.org/abs/2406.07151 - EEG-ImageNet dataset, biggest I have found for mental image classification.
https://arxiv.org/abs/2404.05553 - Alljoined1 dataset (64 channels, 8 subjects)

### Models
https://arxiv.org/abs/2311.03764 : Neuro-GPT (EEGConformer + GPT3), pretrained on TUH, didn't have good results, probably because mental image classification data (EEGImageNet, Alljoined1) have very short EEG length (300 - 500 points)
LaBraM                           : 



### Other
https://arxiv.org/pdf/2010.11929 : ViT
https://arxiv.org/abs/2012.12877 : DeiT (distilled ViT using a teacher CNN)
https://arxiv.org/abs/2106.08254 : BeiT, BERT pre-training of image transformers, uses learned visual codebook learned based on a vector quantizer
https://arxiv.org/abs/2208.06366 : BeiTv2, 


## To Review

### Datasets
https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub  : EEG dataset for human visual object recognition
https://archive.ics.uci.edu/dataset/121/eeg+database                            : EEG correlates of genetic predisposition to alcoholism (64 channels, 256Hz)
https://mindbigdata.com/                                                        : Single-subject EEG data for image classification
https://openbci.com/community/publicly-available-eeg-datasets/                  : Publicly Available EEG Datasets
https://www.sciencedirect.com/science/article/pii/S2213158223001730             : EEG pathology decoding


### Models
https://openreview.net/forum?id=QzTpTRVtrP                                                          :  LaBraM
https://arxiv.org/abs/2404.14869                                                                    :  EEGEncoder: Advancing BCI with Transformer-Based Motor Imagery Classification
https://arxiv.org/abs/2409.00101                                                                    :  NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals
https://arxiv.org/abs/2401.18006                                                                    : EEG-GPT, 2024
https://arxiv.org/abs/2401.10278                                                                    : EEGFormer, foundation model, 2024
https://www.biorxiv.org/content/10.1101/2024.01.18.576245v3.full                                    : Self-supervised transformer for sleep, 2024

https://ieeexplore.ieee.org/abstract/document/9387339                                               : Automatic Detection of Motor and Mental Imagery EEG Signals
https://braindecode.org/stable/generated/braindecode.models.EEGNetv4.html                           :  EEGNet v4
https://link.springer.com/chapter/10.1007/978-3-030-21642-9_8                                       : ChronoNet for abnormal EEG identification
https://www.mdpi.com/1424-8220/23/13/5960                                                           : Abnormal EEG Signals Detection Using WaveNet and LSTM
https://github.com/perceivelab/eeg_visual_classification                        
https://ieeexplore.ieee.org/abstract/document/9987523                                               : Multiscale Convolutional Transformer for EEG Classification of Mental Imagery
https://www.sciencedirect.com/science/article/pii/S1746809423005633                                 : Transformer + ensemble models for EEG classificaiton, 2023
https://arxiv.org/html/2405.16901v2                                                                 : RNN + CNN for EEG classification for guided imagery, 2024
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1148855/full          : EEGformer, transformer for brain activity classification
https://pmc.ncbi.nlm.nih.gov/articles/PMC9955721/                                                   : Subject-independent EEG classification, 2023
https://peijin.medium.com/deep-learning-for-eegs-nad-bci-some-notes-and-some-warnings-28cfc3015a98  : DL for BCI
https://ieeexplore.ieee.org/document/9991178                                                        : EEG Conformer (Conv transformer) for EEG decoding
https://huggingface.co/evegarcianz/eega-embedding                                                   : sentence compressor
https://huggingface.co/ms57rd/Llama-3.1-8B-quantized-EEG-TimeLLM                                    : EEG forecasting (Llama + Time-LLM)
https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1                                           : sentence embedding model

### Surveys
https://www.arxiv.org/abs/2410.08224                                                        : A Survey of Spatio-Temporal EEG data Analysis: from Models to Applications
https://www.sciencedirect.com/science/article/pii/S0925231224011251                         : Survey on cross-subject, cross-session EEG emotion recognition
https://www.sciencedirect.com/science/article/pii/S0168010224000750#bib122                  : Foundation models and generative AI for neuroscience
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1203104/full  : Generative AI for brain image computing and brain network computing
https://link.springer.com/article/10.1007/s00521-021-06352-5                                : EEG DL on motor imagery (2021)
https://pmc.ncbi.nlm.nih.gov/articles/PMC7664219/                                           : EEG TL for BCI (2020)
https://www.sciencedirect.com/science/article/pii/S093336572300252X                         : EEG DL on MI (motor imagery), 2024
https://ieeexplore.ieee.org/abstract/document/10305163                                      : Transformers on BCI, 2023
https://www.sciencedirect.com/science/article/pii/S0925231220314223                         : EEG TL, 2021
https://pmc.ncbi.nlm.nih.gov/articles/PMC9200984/                                           : DL in neuroimaging
https://www.sciencedirect.com/science/article/pii/S0925231224011251                         : Cross-subject, cross-session emotion recogniton
https://arxiv.org/abs/1907.01332                                                            : 
https://www.mdpi.com/2227-7080/10/4/79                                                      : ML model evaluation for EEG classification, 2022
https://arxiv.org/abs/2203.10009                                                            : DL for EEG benchmark, 2022
https://mindbigdata.com/opendb/2022-ACCS8%20Proceedings%20Book-81-82.pdf                    : Evaluating ML models on MindBigData
https://www.sciencedirect.com/science/article/pii/S0925231220314223#b0740                   : EEG TL, 2021
https://ieeexplore.ieee.org/abstract/document/9492294                                       : TL for emotion recognition, 2021
https://www.mdpi.com/1424-8220/20/21/6321                                                   : TL for BCI

### Others (must read)
https://arxiv.org/pdf/2403.15415                                                                    : EEG domain adaptation
https://www.mdpi.com/1424-8220/19/13/2999                                                           : Inter-subject modeling for EEG affect recognition
https://arxiv.org/abs/2309.04153                                                                    : Mapping EEG Signals to Visual Stimuli, 2023
https://www.sciencedirect.com/science/article/pii/S1046202321001018	                                : Cross-subject EEG-based driver states awareness recognition
https://pubmed.ncbi.nlm.nih.gov/2791314/                                                            : Inter- vs intra-subject variance in topographic mapping of the EEG
https://arxiv.org/abs/2007.06407                                                                    : Deep Cross-Subject Mapping of Neural Activity 
https://arxiv.org/abs/2403.15415                                                                    : Physics-informed and Unsupervised Riemannian Domain Adaptation for Machine Learning on Heterogeneous EEG Datasets
https://arxiv.org/abs/2408.08065                                                                    : Scalable EEG pre-processing for self-supervised learning, 2024
https://arxiv.org/abs/2403.16540                                                                    : Cross-Dataset emotion recognition, 2024
https://www.sciencedirect.com/science/article/pii/S1746809423009308                                 : GAN for EEG image reconstruction, 2024
https://arxiv.org/abs/2403.07721                                                                    : EEG embedings decoding and reconstruction with guided diffusion, 2024
https://arxiv.org/abs/2403.06532                                                                    : Reconstructing images from EEG, 2024
https://arxiv.org/abs/2404.01250                                                                    : Image reconstruction from EEG using latent diffusion, 2024
https://www.sciencedirect.com/science/article/abs/pii/S174680942300558X                             : Diffusion model for image reconstruction from EEG
https://www.sciencedirect.com/science/article/pii/S1746809422008941                                 : VAE for EEG to image, 2023
https://ieeexplore.ieee.org/abstract/document/10096587                                              : EEG2Image, 2023
https://www.nature.com/articles/s41598-024-66228-1                                                  : low density EEG image classification and reconstruction, 2024
https://www.sciencedirect.com/science/article/pii/S1746809420302901                                 : Hybrid TL NN for MI (motor imagery) decoding, 2021
https://www.mdpi.com/1424-8220/21/7/2369                                                            : EEG TL for cross-subject fatigue prediction, 2021
https://arxiv.org/abs/1907.01332                                                                    : EEG TL, 2019
https://ieeexplore.ieee.org/abstract/document/8462115                                               : EEG TL for BCI, 2018
https://ieeexplore.ieee.org/abstract/document/4400838                                               : Single-trial EEG source reconstruction for BCI, 2008
https://link.springer.com/chapter/10.1007/978-3-030-83704-4_6                                       : EEG reconstruction using transformers, 2021
https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1194751/full    : Subject-to-subject style transfer for BCI, 2023
https://www.sciencedirect.com/science/article/pii/S0893608020304305                                 : EEG TL for MI using CNN (https://github.com/zhangks98/eeg-adapt)
https://arxiv.org/abs/2106.03746                                                                    : Efficient Training of Visual Transformers with Small Datasets, 2021
https://iopscience.iop.org/article/10.1088/1741-2552/aaf3f6/meta                                    : Inter-subject TL using EEG for BCI (2019)
https://ieeexplore.ieee.org/abstract/document/8786636                                               : TL MI (motor imagery) using CNNs, 2019
https://link.springer.com/article/10.1007/s11517-020-02176-y                                        : Cross-session, cross-subject TL for MI, 2020
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274847                           : EEG classificati`on + reconstruction, 2022
https://cdn.aaai.org/ojs/6817/6817-13-10046-1-10-20200524.pdf                                       : Cross-Aligned Latent Embeddings for Zero-Shot Cross-Modal Retrieval
https://www.sciencedirect.com/science/article/pii/S0010482522000804                                 : CNN + LSTM TL for motor imagery, 2022
https://arxiv.org/abs/2111.06377                                                                    : Masked Autoencoder are Scalable Vision Learners
https://link.springer.com/article/10.1007/s00521-022-08178-1                                        : NeuroGAN (attention-based GAN for EEG image reconstruction, good for embeddings maybe), end of 2022
https://arxiv.org/abs/2206.03950                                                                    : TL for decoding brain states
https://ieeexplore.ieee.org/abstract/document/8675478                                               : Multisource TL for cross-subject emotion recognition
https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2017.00334/full      : Conditional TL for emotion classification, 2017
https://www.mdpi.com/1424-8220/19/13/2999                                                           : Combining Inter-Subject Modeling with a Subject-Based Data Transformation to Improve Affect Recognition from EEG Signals
https://github.com/ptirupat/ThoughtViz?tab=readme-ov-file                                           : ThoughtViz, thought visualizer using GANs
https://www.sciencedirect.com/science/article/pii/S105381192200163X                                 : EEG variability: Task-driven or subject-driven signal of interest?
https://paperswithcode.com/sota/image-classification-on-imagenet                                    : SotA on ImageNet (try to apply on EEG/fMRI)
https://paperswithcode.com/sota/image-generation-on-imagenet-256x256                                : SotA on ImageNet image reconstruction (try to apply on EEG/fMRI, also use DALLE and others, VAEs, diffusion, ViT, multimodal models, CNN+Transformer etc)  : random EEG time-series transfomer
https://huggingface.co/JLB-JLB/EEG_TimeSeriesTransformer_336_history_96_horizon
https://upcommons.upc.edu/bitstream/handle/2117/109756/Personalized-Image-Classification-of-EEG-Signals-using-Deep-Learning.pdf  : Image classification using simple LSTMs


### Others (optional)
https://braindecode.org/0.7/auto_examples/plot_tuh_eeg_corpus.html          : TUH dataset processing on braindecode.org
https://arxiv.org/abs/1609.02200                                            : Discrete VAEs
https://huggingface.co/models?other=text-embeddings-inference&sort=trending : HuggingFace text embedding models


# fMRI

## To Review

### Datasets
https://naturalscenesdataset.org/                           : 7T fMRI, 8 subjects viewing color natural scenes
https://www.nature.com/articles/s41597-023-02471-x          : 30 subjects, naturalistic images
https://www.nature.com/articles/s41593-021-00962-x          : 7T, 8 subjects
https://www.nature.com/articles/s41597-019-0052-3           : BOLD5000, 5000 images, 4 subjects
https://openneuro.org/datasets/ds001506/versions/1.3.1      : 3 subjects (Kamitani, 2019)


### Models
https://www.sciencedirect.com/science/article/pii/S0893608023006470                                                     : Image reconstruction, 2024
https://www.sciencedirect.com/science/article/pii/S0925231220301041                                                     : Alzheimer's disease (AD) using 4D fMRI, 2020
https://github.com/KamitaniLab/DeepImageReconstruction                                                                  : Image reconstruction, 2019
https://www.nature.com/articles/s41467-024-48114-6                                                                      : Computational reconstruction of mental representations using human behavior, 2024
https://dr.library.brocku.ca/bitstream/handle/10464/17138/Alzheimers.pdf?sequence=1&isAllowed=y                         : AD DL
https://arxiv.org/abs/1603.08631                                                                                        : AD CNN, 2016
https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-7/issue-5/056001/Spatiotemporal-feature-extraction-and-classification-of-Alzheimers-disease-using-deep/10.1117/1.JMI.7.5.056001.full     : AD 3D-CNN spatiotemporal extraction, 2020
https://ieeexplore.ieee.org/abstract/document/8681247                                                                   : Behavior TL, 2019
https://www.sciencedirect.com/science/article/pii/S2451902221003451                                                     : OCD cross-diagnosis TL, 2022
https://www.sciencedirect.com/science/article/pii/S016503272300928X                                                     : Depression classification using time series model for rs-fMRI, 2023
https://link.springer.com/article/10.1007/s12559-021-09946-2                                                            : Early AD detection, 2022
https://www.sciencedirect.com/science/article/pii/S1361841522003073                                                     : SD-CNN, 2023
https://www.researchsquare.com/article/rs-3290143/v1                                                                    : 3DCAE-MRI, for small sample size, 2023
https://link.springer.com/article/10.1007/S12559-019-09688-2                                                            : AD, 2020



### Reviews
https://www.sciencedirect.com/science/article/pii/S0010482522004267                             : AD, 2022
https://pmc.ncbi.nlm.nih.gov/articles/PMC10092597/                                              : DL for AD, 2022
https://journals.sagepub.com/doi/full/10.1177/1094428118802631                                  : 2018
https://www.igi-global.com/chapter/functional-magnetic-resonance-imaging/338978                 : 2024
https://pmc.ncbi.nlm.nih.gov/articles/PMC10092597/#jon13063-bib-0037                            : AD, 2022

### Others
https://github.com/athms/learning-from-brains                                                       : Self-supervised, brain dynamics, 2022
https://www.sciencedirect.com/science/article/pii/S0925231223009141                                 : Multi-site, brain analysis framework, 2023
https://www.sciencedirect.com/science/article/pii/S2213158218300329?ref=cra_js_challenge&fr=RR-1    : Cognitive deficits in infants, 2018




# Courses
https://www.youtube.com/watch?v=3ExL6J4BIeo&list=PLvgasosJnUVl_bt8VbERUyCLU93OG31h_                 : FSL, Oxford, 2021
https://www.youtube.com/watch?v=ASEyg5nxj3A&list=PL7B6LR3JHY84vVSrrjdtOCRsBPz7kR7Zr&index=13        : Neurohackademy fMRI ML
https://www.youtube.com/watch?v=KykjMNVLp6s&list=PL7B6LR3JHY84vVSrrjdtOCRsBPz7kR7Zr&index=8         : fMRI ML
https://www.youtube.com/watch?v=3P_hN0hrp-U&list=PL7B6LR3JHY84vVSrrjdtOCRsBPz7kR7Zr&index=6         : fMRI ML
https://www.youtube.com/watch?v=GDkLQuV4he4&list=PLfXA4opIOVrGHncHRxI3Qa5GeCSudwmxM&index=24



# Time series

## To Review

### Models
https://paperswithcode.com/paper/time-series-classification-from-scratch-with       : 2016, has code


### Reviews
https://arxiv.org/abs/1809.04356                                : DL for time series classification, 2018




## Reviewed