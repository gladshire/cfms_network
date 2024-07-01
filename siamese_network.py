import matplotlib.pyplot as plt
import numpy as np
import random
import math
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pylab
import scipy.stats
from scipy.stats import norm

import pandas as pd
import qnorm
import seaborn as sns

import curses
import sys
import os




# Use random subset samples of test/validation sets for speed
# Should not use this on actual runs
__FAST_VALID_TEST__ = False

DATADIR_ELUT = "data/elut/"
DATADIR_PPIS = "data/ppi/"

NUM_EPOCHS = 250
BATCH_SIZE = 16
LEARN_RATE = 0.001
NUM_THREAD = 2
SUBSET_SIZE = 1000 # For __FAST_VALID_TEST__
TEMPERATURE = 25.0 # For cosine similarity contrastive loss
MOMENTUM = 0

SAMPLE_RATE = 10 # How many batches between samples (for loss curve)


# Goal: to produce a network for discerning similarity in elution profiles between two proteins

# Want to split proteins from complexes into datasets as partners. All proteins from a given complex must
# be in the same set
# How do produce positive, negative labels?
# How many proteins shared between test and training set? C3, C2, C1 splits
# C1 = when both proteins are in TRAIN and TEST
# C2 = when one of the proteins is in TRAIN and TEST
# C3 = when both proteins are in either TRAIN or TEST
# Loss function: contrastive loss


print("Reading data from files ...")

# Read in training complexes
training_complexes = []
training_complex_file = DATADIR_PPIS + "intact_complex_merge_20230309.train.txt"
f = open(training_complex_file)
for line in f.readlines():
    training_complexes.append(set(line.split()))
f.close()


# Read in training PPIs
training_pos_ppis = []
training_pos_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.train_ppis.txt"
f = open(training_pos_ppis_file)
for line in f.readlines():
    training_pos_ppis.append(set(line.split()))
f.close()

training_neg_ppis = []
training_neg_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.neg_train_ppis.txt"
f = open(training_neg_ppis_file)
for line in f.readlines():
    training_neg_ppis.append(set(line.split()))
f.close()


# Read in test PPIs
test_pos_ppis = []
test_pos_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.test_ppis.txt"
f = open(test_pos_ppis_file)
for line in f.readlines():
    test_pos_ppis.append(set(line.split()))
f.close()

test_neg_ppis = []
test_neg_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.neg_test_ppis.txt"
f = open(test_neg_ppis_file)
for line in f.readlines():
    test_neg_ppis.append(set(line.split()))
f.close()


# Read elution files into dataframe, thresholding peak specificity at 10 psm
#   Two normalization methods:
#     1. Normalization is performed by dividing each replicate by its maximum intensity row-wise
#     2. Quantile normalization is performed across all samples (currently in-use)
#   
#   Notes: Quantile normalization smoothes out loss curve considerably, still get
#          very rugged topology in main valley

elut1_df = pd.read_csv(DATADIR_ELUT + "HEK293_EDTA_minus_SEC_control_20220626.elut", sep='\t')
elut1_df = elut1_df.set_index('Unnamed: 0')
elut1_t10_df = elut1_df[elut1_df.sum(axis=1) > 10]
elut1_t10_rwn_df = elut1_t10_df.div(elut1_t10_df.max(axis=1), axis=0)
elut1_t10_rsm_df = elut1_t10_df.div(elut1_t10_df.sum(axis=1), axis=0)
elut1_t10_qtn_df = qnorm.quantile_normalize(elut1_t10_df, axis=0)

elut2_df = pd.read_csv(DATADIR_ELUT + "HEK293_EDTA_plus_SEC_treatment_20220626_trimmed.elut", sep='\t')
elut2_df = elut2_df.set_index('Unnamed: 0')
elut2_t10_df = elut2_df[elut2_df.sum(axis=1) > 10]
elut2_t10_rwn_df = elut2_t10_df.div(elut2_t10_df.max(axis=1), axis=0)
elut2_t10_rsm_df = elut2_t10_df.div(elut2_t10_df.sum(axis=1), axis=0)
elut2_t10_qtn_df = qnorm.quantile_normalize(elut2_t10_df, axis=0)

elut3_df = pd.read_csv(DATADIR_ELUT + "Anna_HEK_urea_SEC_0M_050817_20220314b_trimmed.elut", sep='\t')
elut3_df = elut3_df.set_index('Unnamed: 0')
elut3_t10_df = elut3_df[elut3_df.sum(axis=1) > 10]
elut3_t10_rwn_df = elut3_t10_df.div(elut3_t10_df.max(axis=1), axis=0)
elut3_t10_rsm_df = elut3_t10_df.div(elut3_t10_df.sum(axis=1), axis=0)
elut3_t10_qtn_df = qnorm.quantile_normalize(elut3_t10_df, axis=0)

elut4_df = pd.read_csv(DATADIR_ELUT + "Anna_HEK_urea_SEC_0p5M_052317_20220315_reviewed_trimmed.elut", sep='\t')
elut4_df = elut4_df.set_index('Unnamed: 0')
elut4_t10_df = elut4_df[elut4_df.sum(axis=1) > 10]
elut4_t10_rwn_df = elut4_t10_df.div(elut4_t10_df.max(axis=1), axis=0)
elut4_t10_rsm_df = elut4_t10_df.div(elut4_t10_df.sum(axis=1), axis=0)
elut4_t10_qtn_df = qnorm.quantile_normalize(elut4_t10_df, axis=0)

elut5_df = pd.read_csv(DATADIR_ELUT + "Hs_helaN_1010_ACC.prot_count_mFDRpsm001.elut", sep='\t')
elut5_df = elut5_df.set_index('Unnamed: 0')
elut5_t10_df = elut5_df[elut5_df.sum(axis=1) > 10]
elut5_t10_rwn_df = elut5_t10_df.div(elut5_t10_df.max(axis=1), axis=0)
elut5_t10_rsm_df = elut5_t10_df.div(elut5_t10_df.sum(axis=1), axis=0)
elut5_t10_qtn_df = qnorm.quantile_normalize(elut5_t10_df, axis=0)

elut6_df = pd.read_csv(DATADIR_ELUT + "PXD001220.elut", sep='\t')
elut6_df = elut6_df.set_index('Unnamed: 0')
elut6_t10_df = elut6_df[elut6_df.sum(axis=1) > 10]
elut6_t10_rwn_df = elut6_t10_df.div(elut6_t10_df.max(axis=1), axis=0)
elut6_t10_rsm_df = elut6_t10_df.div(elut6_t10_df.sum(axis=1), axis=0)
elut6_t10_qtn_df = qnorm.quantile_normalize(elut6_t10_df, axis=0)

elut7_df = pd.read_csv(DATADIR_ELUT + "PXD003754.elut", sep='\t')
elut7_df = elut7_df.set_index('Unnamed: 0')
elut7_t10_df = elut7_df[elut7_df.sum(axis=1) > 10]
elut7_t10_rwn_df = elut7_t10_df.div(elut7_t10_df.max(axis=1), axis=0)
elut7_t10_rsm_df = elut7_t10_df.div(elut7_t10_df.sum(axis=1), axis=0)
elut7_t10_qtn_df = qnorm.quantile_normalize(elut7_t10_df, axis=0)

elut8_df = pd.read_csv(DATADIR_ELUT + "PXD009833.elut", sep='\t')
elut8_df = elut8_df.set_index('Unnamed: 0')
elut8_t10_df = elut8_df[elut8_df.sum(axis=1) > 10]
elut8_t10_rwn_df = elut8_t10_df.div(elut8_t10_df.max(axis=1), axis=0)
elut8_t10_rsm_df = elut8_t10_df.div(elut8_t10_df.sum(axis=1), axis=0)
elut8_t10_qtn_df = qnorm.quantile_normalize(elut8_t10_df, axis=0)

elut9_df = pd.read_csv(DATADIR_ELUT + "PXD009834.elut", sep='\t')
elut9_df = elut9_df.set_index('Unnamed: 0')
elut9_t10_df = elut9_df[elut9_df.sum(axis=1) > 10]
elut9_t10_rwn_df = elut9_t10_df.div(elut9_t10_df.max(axis=1), axis=0)
elut9_t10_rsm_df = elut9_t10_df.div(elut9_t10_df.sum(axis=1), axis=0)
elut9_t10_qtn_df = qnorm.quantile_normalize(elut9_t10_df, axis=0)

elut10_df = pd.read_csv(DATADIR_ELUT + "PXD014820.elut", sep='\t')
elut10_df = elut10_df.set_index('Unnamed: 0')
elut10_t10_df = elut10_df[elut10_df.sum(axis=1) > 10]
elut10_t10_rwn_df = elut10_t10_df.div(elut10_t10_df.max(axis=1), axis=0)
elut10_t10_rsm_df = elut10_t10_df.div(elut10_t10_df.sum(axis=1), axis=0)
elut10_t10_qtn_df = qnorm.quantile_normalize(elut10_t10_df, axis=0)

elut11_df = pd.read_csv(DATADIR_ELUT + "PXD015406.elut", sep='\t')
elut11_df = elut11_df.set_index('Unnamed: 0')
elut11_t10_df = elut11_df[elut11_df.sum(axis=1) > 10]
elut11_t10_rwn_df = elut11_t10_df.div(elut11_t10_df.max(axis=1), axis=0)
elut11_t10_rsm_df = elut11_t10_df.div(elut11_t10_df.sum(axis=1), axis=0)
elut11_t10_qtn_df = qnorm.quantile_normalize(elut11_t10_df, axis=0)

elut12_df = pd.read_csv(DATADIR_ELUT + "MSV000081520.elut", sep='\t')
elut12_df = elut12_df.set_index('Unnamed: 0')
elut12_t10_df = elut12_df[elut12_df.sum(axis=1) > 10]
elut12_t10_rwn_df = elut12_t10_df.div(elut12_t10_df.max(axis=1), axis=0)
elut12_t10_rsm_df = elut12_t10_df.div(elut12_t10_df.sum(axis=1), axis=0)
elut12_t10_qtn_df = qnorm.quantile_normalize(elut12_t10_df, axis=0)



elut_list = [elut1_t10_df, elut2_t10_df, elut3_t10_df, elut4_t10_df,
             elut5_t10_df, elut6_t10_df, elut7_t10_df, elut8_t10_df,
             elut9_t10_df, elut10_t10_df, elut11_t10_df, elut12_t10_df]
'''
elut_list = [elut1_t10_rwn_df, elut2_t10_rwn_df, elut3_t10_rwn_df,
             elut4_t10_rwn_df, elut5_t10_rwn_df]
elut_list = [elut1_t10_rwn_df, elut2_t10_rwn_df, elut3_t10_rwn_df,
             elut4_t10_rwn_df, elut5_t10_rwn_df, elut4_t10_rwn_df,
             elut5_t10_rwn_df, elut6_t10_rwn_df, elut7_t10_rwn_df,
             elut8_t10_rwn_df, elut9_t10_rwn_df, elut10_t10_rwn_df,
             elut11_t10_rwn_df, elut12_t10_rwn_df]
elut_list = [elut1_t10_qtn_df, elut2_t10_qtn_df, elut3_t10_qtn_df,
             elut4_t10_qtn_df, elut5_t10_qtn_df, elut4_t10_qtn_df,
             elut5_t10_qtn_df, elut6_t10_qtn_df, elut7_t10_qtn_df,
             elut8_t10_qtn_df, elut9_t10_qtn_df, elut10_t10_qtn_df,
             elut11_t10_qtn_df, elut12_t10_qtn_df]
elut_list = [elut1_t10_rsm_df, elut2_t10_rsm_df, elut3_t10_rsm_df,
             elut4_t10_rsm_df, elut5_t10_rsm_df, elut4_t10_rsm_df,
             elut5_t10_rsm_df, elut6_t10_rsm_df, elut7_t10_rsm_df,
             elut8_t10_rsm_df, elut9_t10_rsm_df, elut10_t10_rsm_df,
             elut11_t10_rsm_df, elut12_t10_rsm_df]
'''
print("Elution data obtained. Now preparing ...")



# Obtain set of all unique proteins across all elution data
elut_proteins = set()
for ed in elut_list:
    elut_proteins = elut_proteins.union(ed.index)


# Only keep protein pairs contained within the elution dataframe
trainingelut_pos_ppis = []
for i in training_pos_ppis:
    if len(i.intersection(elut_proteins)) == 2:
        trainingelut_pos_ppis.append(i)

trainingelut_neg_ppis = []
for i in training_neg_ppis:
    if len(i.intersection(elut_proteins)) == 2:
        trainingelut_neg_ppis.append(i)




# Shuffle training ppis, split into training and validation sets
random.shuffle(trainingelut_pos_ppis)
random.shuffle(trainingelut_neg_ppis)

train_pos_ppis = trainingelut_pos_ppis[:(int)(len(trainingelut_pos_ppis)/2)]
valid_pos_ppis = trainingelut_pos_ppis[(int)(len(trainingelut_pos_ppis)/2):]
train_neg_ppis = trainingelut_neg_ppis[:(int)(len(trainingelut_neg_ppis)/2)]
valid_neg_ppis = trainingelut_neg_ppis[(int)(len(trainingelut_pos_ppis)/2):]


# Only keep protein pairs contained within elution dataframe
testelut_pos_ppis = []
for i in test_pos_ppis:
    if len(i.intersection(elut_proteins)) == 2:
        testelut_pos_ppis.append(i)

testelut_neg_ppis = []
for i in test_neg_ppis:
    if len(i.intersection(elut_proteins)) == 2:
        testelut_neg_ppis.append(i)


def plot_loss(iteration, loss, color, xaxis, title, filename=None):
    plt.plot(iteration, loss, color=color)
    plt.grid()
    plt.title(title)
    plt.xlabel(xaxis)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    plt.cla()






# Wrap elution pair data into PyTorch dataset
class elutionPairDataset(Dataset):
    def __init__(self, elutdf_list, pos_ppis, neg_ppis, transform=False, input_size=128):
        self.elut_df_list = elutdf_list
        self.ppis = []
        self.labels = []
        self.ppis_elut_ids = []

        for elut_id, elut_df in enumerate(self.elut_df_list):
            # miles: Loading the elut dataframe as a set vastly reduces runtime of dataset initialization
            elut_df_index = set(elut_df.index)

            pos_ppis_elut = [(tuple(pppi.intersection(elut_df_index)), pppi) for pppi in pos_ppis if len(pppi.intersection(elut_df_index)) == 2]
            neg_ppis_elut = [(tuple(nppi.intersection(elut_df_index)), nppi) for nppi in neg_ppis if len(nppi.intersection(elut_df_index)) == 2]

            self.ppis = self.ppis + pos_ppis_elut + neg_ppis_elut
            self.labels = self.labels + [0]*len(pos_ppis_elut) + [1]*len(neg_ppis_elut)
            self.ppis_elut_ids = self.ppis_elut_ids + [elut_id]*len(pos_ppis_elut) + [elut_id]*len(neg_ppis_elut)

        self.transform = transform
        self.input_size = input_size

    def __getitem__(self, index):
        pair = self.ppis[index]
        elut_id = self.ppis_elut_ids[index]
        elut_df = self.elut_df_list[elut_id]

        prot0 = pair[0][0]
        prot1 = pair[0][1]

        elut0 = (prot0, torch.from_numpy(elut_df.T[list(pair[1])[0]].values).float())
        elut1 = (prot1, torch.from_numpy(elut_df.T[list(pair[1])[1]].values).float())

        if self.transform:
            elut0 = (prot0, nn.functional.pad(elut0[1], (self.input_size - elut0[1].size()[0], 0)))
            elut1 = (prot1, nn.functional.pad(elut1[1], (self.input_size - elut1[1].size()[0], 0)))

        elut0 = (elut0[0], elut0[1].unsqueeze(0))
        elut1 = (elut1[0], elut1[1].unsqueeze(0))

        return elut0, elut1, self.labels[index]

    def __len__(self):
        return len(self.ppis)



# Instantiate the training dataset for the model
train_siamese_dataset = elutionPairDataset(elutdf_list=elut_list,
                                           pos_ppis = train_pos_ppis,
                                           neg_ppis = train_neg_ppis,
                                           transform=True)
print("Training set assembled")
print(f"  Number of samples: {len(train_siamese_dataset)}")

# Instantiate the validation dataset for the model
valid_siamese_dataset = elutionPairDataset(elutdf_list=elut_list,
                                           pos_ppis = valid_pos_ppis,
                                           neg_ppis = valid_neg_ppis,
                                           transform=True)

subset_indices = torch.randperm(len(valid_siamese_dataset))[:SUBSET_SIZE]
subset_valid_siamese_dataset = Subset(valid_siamese_dataset, subset_indices)
print("Validation set assembled")
print(f"  Number of samples: {len(valid_siamese_dataset)}")

# Instantiate the test dataset for the model
test_siamese_dataset = elutionPairDataset(elutdf_list=elut_list,
                                          pos_ppis = test_pos_ppis,
                                          neg_ppis = test_neg_ppis,
                                          transform=True)

subset_indices = torch.randperm(len(test_siamese_dataset))[:SUBSET_SIZE]
subset_test_siamese_dataset = Subset(test_siamese_dataset, subset_indices)
print("Test set assembled")
print(f"  Number of samples: {len(test_siamese_dataset)}")



# Instantiate a dataloader for visualization
vis_dataloader = DataLoader(train_siamese_dataset,
                            shuffle=True,
                            drop_last=True,
                            num_workers=NUM_THREAD,
                            batch_size=8)

# One batch is a list containing 2x8 images, indexes 0 and 1, and the label
# If label is 1, NOT the same object in images
# If label is 0, SAME object in both images
#concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
#imshow(torchvision.utils.make_grid(concatenated))


# Define the siamese neural network
class siameseNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(siameseNet, self).__init__()

        # Elution tensor input: (batch_size, channels, num_fractions)

        # CONVOLUTION OUTPUT DIMENSIONS:
        #   Size: (num_fractions - kernel_size + 2*padding / stride) + 1)

        self.cnn1 = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=4,
                          kernel_size=4, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=3, stride=2),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=4, out_channels=8,
                          kernel_size=5, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
        )

        self.rnn = nn.LSTM(input_size=8, hidden_size=hidden_size, num_layers = 1,
                           bidirectional=True, batch_first=True)


        self.cnn2 = nn.Sequential(
                nn.ConvTranspose1d(in_channels=2*hidden_size, out_channels=8,
                                   kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose1d(in_channels=8, out_channels=4,
                                   kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose1d(in_channels=4, out_channels=1,
                                   kernel_size=3, stride=1, padding=1),

        )
        self.fc = nn.Sequential(

                nn.Linear(31, 64),
                nn.ReLU(inplace=True),

                nn.Linear(64, 32),
                nn.ReLU(inplace=True),

                nn.Linear(32, 2)
        )

    # Function called on both images, x, to determine their similarity
    def forward_once(self, x, pcc):
        # Apply convolutional encoding
        y = self.cnn1(x)

        # Prepare for recurrent layers
        y = y.permute(0, 2, 1)

        # Apply transformer layer
        #y = self.tns(y)

        # Apply bidirectional recurrency
        y, _ = self.rnn(y)

        # Apply transformer layer
        #y = self.tns(y)

        # Prepare for convolutional decoding
        y = y.permute(0, 2, 1)

        # Apply convolutional decoding
        y = self.cnn2(y)

        # Flatten output to work with fully connected layer
        y = y.view(y.size()[0], -1)

        # Prepare Pearson input for network
        pcc = pcc.view(-1, 1).expand(y.size(0), 1)
        y = torch.cat((y, pcc), dim=1)

        # Apply fully connected layer
        # Potentiall add Pearson correlation coefficient here
        y = self.fc(y)
        return y

    # Main forward function
    def forward(self, input1, input2, pcc):
        output1 = self.forward_once(input1, pcc)
        output2 = self.forward_once(input2, pcc)

        return output1, output2

# Define contrastive loss function, using Euclidean distance
class contrastiveLossEuclidean(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(contrastiveLossEuclidean, self).__init__()
        self.margin = margin

    # Contrastive loss calculation
    #   CL = mean( (1-y)*d^2 + y*
    def forward(self, output1, output2, label):
        euclidean_dist = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))

        
        return loss_contrastive

# Define SimCLR contrastive loss, based on cosine similarity
class contrastiveLossCosineSimilarity(nn.Module):
    def __init__(self, margin=0.2, tau=1.0):
        super(contrastiveLossCosineSimilarity, self).__init__()
        self.margin = margin
        self.tau = tau

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        cos_sim_t = cos_sim / self.tau

        loss_contrastive = torch.mean((1-label) * torch.pow(cos_sim_t, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cos_sim_t, min=0.0), 2))
        return loss_contrastive

print("Network defined. Wrapping elution datasets in DataLoaders ...")
# Load training dataset
train_dataloader = DataLoader(train_siamese_dataset,
                              shuffle=True,
                              drop_last=True,
                              num_workers=2,
                              batch_size=BATCH_SIZE)

# Load validation and test datasets
if __FAST_VALID_TEST__:
    valid_dataloader = DataLoader(subset_valid_siamese_dataset,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=2,
                                  batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(subset_test_siamese_dataset,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=2,
                                 batch_size=BATCH_SIZE)
else:
    valid_dataloader = DataLoader(valid_siamese_dataset,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=2,
                                  batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_siamese_dataset,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=2,
                                 batch_size=BATCH_SIZE)

# Instantiate network
trainNet = True
net = siameseNet().cuda()
if len(sys.argv) != 1:
    model_to_load = sys.argv[1]
    net.load_state_dict(torch.load(model_to_load))
    trainNet = False
    print(f"Skipping training. Performing benchmark on {model_to_load}")
    net.eval()


# Choose loss function to use
#
criterion = contrastiveLossEuclidean()

#criterion = contrastiveLossCosineSimilarity(tau=TEMPERATURE)
#criterion = nn.CosineEmbeddingLoss(margin=0.8, reduction='sum')


# Choose optimizer algorithm
# miles: Notes
#   - Adam
#     - Easy, optimizes multiple parameters separately from one another
#     - Literature shows that it generalizes far worse than SGD longterm
#
#   - SGD
#     - Difficult, requires lots of hyper-parameter tuning to get working
#     - Potentially generalizes far better. Should eventually use this one.
#     - What doesn't work?
#       > Without momentum
#     - Which hyper-parameters (alone) tend to
#       improve perforance?
#       > Smaller learning rate
#       > Adding momentum
#       > Learning rate schedulers?
#
#

#optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Zero the gradients
optimizer.zero_grad()

counter = []
loss_hist = []
valid_counter = []
valid_loss_hist = []
epoch_hist = []
avg_test_loss_hist = []
iteration_num = 0
valid_iteration_num = 0

min_avg_valid_loss = 1e9
min_avg_test_loss = 1e9

# Reset weights
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)


#net.apply(weights_init)

# Training loop
#net.train()
if trainNet:
    print("Instantiating model training with following architecture:")
    print(net)

    print("\nWith following hyperparameters")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learn rate: {LEARN_RATE}")

    for epoch in range(NUM_EPOCHS):
        print("\n=====================================")
        print(f"Epoch: {epoch+1} of {NUM_EPOCHS}")
        print("=====================================\n")
    
        # Iterate over batches
        net.train()
        num_batches = len(train_dataloader)
        for i, (elut0, elut1, label) in enumerate(train_dataloader, 0):
            pccList = []

            # Obtain protein IDs
            prot0, prot1 = elut0[0], elut1[0]

            # Obtain Pearson correlation between elut0, elut1
            for ppi0, ppi1 in zip(elut0[1], elut1[1]):
                pccList.append(scipy.stats.pearsonr(ppi0[0], ppi1[0])[0])

            pcc = torch.tensor(pccList, dtype=torch.float32).cuda()

            # Send elution data, labels to CUDA
            elut0, elut1, label = elut0[1].cuda(), elut1[1].cuda(), label.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass two elution traces into network and obtain two outputs
            output1, output2 = net(elut0, elut1, pcc)

            # Pass outputs, label to the contrastive loss function
            loss_train_contrastive = criterion(output1, output2, label)

            # Perform backpropagation
            loss_train_contrastive.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)

            # Optimize
            optimizer.step()

            if i % SAMPLE_RATE == 0:
                print(f"  Batch [{i} / {num_batches}] Training Loss: {loss_train_contrastive.item()}")
                iteration_num += SAMPLE_RATE

                counter.append(iteration_num)
                loss_hist.append(loss_train_contrastive.item())

        # Validation
        net.eval()
        valid_loss = 0.0
        # Iterate over batches
        num_batches_valid = len(valid_dataloader)
        for valid_i, (valid_elut0, valid_elut1, valid_label) in enumerate(valid_dataloader, 0):
            pccListValid = []

            # Obtain protein IDs
            prot0, prot1 = valid_elut0[0], valid_elut1[0]
            for valid_ppi0, valid_ppi1 in zip(valid_elut0[1], valid_elut1[1]):
                pccListValid.append(scipy.stats.pearsonr(valid_ppi0[0], valid_ppi1[0])[0])

            pcc = torch.tensor(pccListValid, dtype=torch.float32).cuda()

            # Send elution data, labels to CUDA
            valid_elut0, valid_elut1, valid_label = valid_elut0[1].cuda(), valid_elut1[1].cuda(), valid_label.cuda()

            # Pass two elution traces into network and obtain two outputs
            valid_output1, valid_output2 = net(valid_elut0, valid_elut1, pcc)

            # Pass outputs, label to the contrastive loss function
            valid_loss_contrastive = criterion(valid_output1, valid_output2, valid_label)
            valid_loss += valid_loss_contrastive.item()

            if valid_i % SAMPLE_RATE == 0:
                print(f"  Batch [{valid_i} / {num_batches_valid}] Validation Loss: {valid_loss_contrastive.item()}")
                valid_iteration_num += SAMPLE_RATE

                valid_counter.append(valid_iteration_num)
                valid_loss_hist.append(valid_loss_contrastive.item())
        avg_valid_loss = valid_loss / len(valid_dataloader)
        if avg_valid_loss < min_avg_valid_loss:
            min_avg_valid_loss = avg_valid_loss

        # Get learning rate before and after stepping the scheduler
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_valid_loss)
        after_lr = optimizer.param_groups[0]["lr"]

        # Alert user if learning rate has changed
        if before_lr != after_lr:
            print(f"  Decreasing learning rate from {before_lr} to {after_lr}")


        with torch.no_grad():
            test_loss = 0.0
            for test_i, (test_elut0, test_elut1, test_label) in enumerate(test_dataloader, 0):
                pccListTest = []
                # Obtain protein IDs
                prot0, prot1 = elut0[0], elut1[0]
                for test_ppi0, test_ppi1 in zip(test_elut0[1], test_elut1[1]):
                    pccListTest.append(scipy.stats.pearsonr(test_ppi0[0], test_ppi1[0])[0])

                pcc = torch.tensor(pccListTest, dtype=torch.float32).cuda()

                # Send test elution data, labels to CUDA
                test_elut0, test_elut1, test_label = test_elut0[1].cuda(), test_elut1[1].cuda(), test_label.cuda()

                # Padd two elution traces into network and obtain two outputs
                output1, output2 = net(test_elut0, test_elut1, pcc)

                # Pass outputs, label to the contrastive loss function
                loss_test_contrastive = criterion(output1, output2, test_label)
                test_loss += loss_test_contrastive

        # Calculate average loss on test dataset
        avg_test_loss = test_loss / len(test_dataloader)
        if avg_test_loss < min_avg_test_loss:
            min_avg_test_loss = avg_test_loss
        epoch_hist.append(epoch+1)
        avg_test_loss_hist.append(avg_test_loss)
        print(f"\nEnd of epoch summary")
        print(f"  Average validation loss: {avg_valid_loss}")
        print(f"  Average testing loss: {avg_test_loss}")
        if avg_test_loss < 0.00001:
            break
        plot_loss(counter, loss_hist, title="Contrastive Loss - Train",
                  xaxis="Batches", color='blue', filename=f"train_{epoch+1}.png")
        plot_loss(valid_counter, valid_loss_hist, title="Contrastive Loss - Validation",
                  xaxis="Batches", color = 'orange', filename=f"valid_{epoch+1}.png")
        if epoch != 0:
            plot_loss(epoch_hist, avg_test_loss_hist, title="Avg Cont Loss - Testing", 
                      color='red', xaxis="Epochs", filename=f"test_{epoch+1}.png")

# Save the model weights
torch.save(net.state_dict(), "./net_SGD.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TEST EUCLIDEAN
# Test only positive ppis
test_pos_elution_pair_dataset = elutionPairDataset(elutdf_list=[elut1_t10_qtn_df],
                                                   pos_ppis=test_pos_ppis,
                                                   neg_ppis=[],
                                                   transform=True)
subset_indices = torch.randperm(len(test_pos_elution_pair_dataset))[:SUBSET_SIZE]
subset_test_pos_elution_pair_dataset = Subset(test_pos_elution_pair_dataset, subset_indices)

if __FAST_VALID_TEST__:
    test_pos_dataloader = DataLoader(subset_test_pos_elution_pair_dataset,
                                     num_workers=1,
                                     batch_size=1,
                                     shuffle=True)
else:
    test_pos_dataloader = DataLoader(test_pos_elution_pair_dataset,
                                     num_workers=1,
                                     batch_size=1,
                                     shuffle=True)

# Construct a list of euclidean distances between two positive protein pairs
# We want these to be smaller, as the network should recognize their similarity
pos_ppi_euclidean_dist_list = []
print("Calculating Euclidean distances for positive pairwise interactions")
for i, (pos_elut0, pos_elut1, pos_label) in enumerate(test_pos_dataloader):
    pccListPos = []

    # Obtain protein IDs
    prot0, prot1 = pos_elut0[0], pos_elut1[0]

    for pos_ppi0, pos_ppi1 in zip(pos_elut0[1], pos_elut1[1]):
         pccListPos.append(scipy.stats.pearsonr(pos_ppi0[0], pos_ppi1[0])[0])
    pcc = torch.tensor(pccListPos, dtype=torch.float32).cuda()

    pos_elut0, pos_elut1 = pos_elut0[1], pos_elut1[1]

    output1, output2 = net(pos_elut0.cuda(), pos_elut1.cuda(), pcc)

    euclidean_dist = F.pairwise_distance(output1, output2)
    pos_ppi_euclidean_dist_list.append(euclidean_dist.item())

# Test only negative ppis
test_neg_elution_pair_dataset = elutionPairDataset(elutdf_list=[elut1_t10_qtn_df],
                                                   pos_ppis=[],
                                                   neg_ppis=test_neg_ppis,
                                                   transform=True)
subset_indices = torch.randperm(len(test_neg_elution_pair_dataset))[:SUBSET_SIZE]
subset_test_neg_elution_pair_dataset = Subset(test_neg_elution_pair_dataset, subset_indices)

if __FAST_VALID_TEST__:
    test_neg_dataloader = DataLoader(subset_test_neg_elution_pair_dataset,
                                     num_workers=1,
                                     batch_size=1,
                                     shuffle=True)
else:
    test_neg_dataloader = DataLoader(test_neg_elution_pair_dataset,
                                     num_workers=1,
                                     batch_size=1,
                                     shuffle=True)

# Construct a list of euclidean distances between two negative protein pairs
# We want these to be larger, as the network should recognize their dissimilarity
neg_ppi_euclidean_dist_list = []
print("Calculating Euclidean distances for negative pairwise interactions")
for i, (neg_elut0, neg_elut1, neg_label) in enumerate(test_neg_dataloader):
    pccListNeg = []

    # Obtain protein IDs
    prot0, prot1 = neg_elut0[0], neg_elut1[0]
    for neg_ppi0, neg_ppi1 in zip(neg_elut0[1], neg_elut1[1]):
         pccListNeg.append(scipy.stats.pearsonr(neg_ppi0[0], neg_ppi1[0])[0])
    pcc = torch.tensor(pccListNeg, dtype=torch.float32).cuda()
    neg_elut0, neg_elut1 = neg_elut0[1], neg_elut1[1]
    output1, output2 = net(neg_elut0.cuda(), neg_elut1.cuda(), pcc)
    euclidean_dist = F.pairwise_distance(output1, output2)
    neg_ppi_euclidean_dist_list.append(euclidean_dist.item())

    if i % 10 == 0:
        print(f"{i * 100 / len(test_neg_dataloader):.2f} %", end='\r')


# Plot the PDF of the euclidean distance between positive and negative protein pairs
x = np.linspace(-5,20,1000)
mean_pos = np.mean(pos_ppi_euclidean_dist_list)
mean_neg = np.mean(neg_ppi_euclidean_dist_list)
diff_means_pos_neg_pdf = abs(mean_neg - mean_pos)
y = norm.pdf(x, loc=mean_neg, scale=np.std(neg_ppi_euclidean_dist_list))
y2 = norm.pdf(x, loc=mean_pos, scale=np.std(pos_ppi_euclidean_dist_list))

pylab.plot(x,y,label="negative_pairs")
pylab.plot(x,y2,label="positive_pairs")
pylab.annotate(f"{mean_pos}", (0.0, mean_pos))
pylab.title("PDFs of Euclidean distances after model transform")
pylab.legend()
pylab.grid()
pylab.savefig("pos_neg_pairs_EUCLIDEAN.png")
pylab.clf()
pylab.cla()


sns.histplot(neg_ppi_euclidean_dist_list, alpha=0.5,
             label='negative pairs')
sns.histplot(pos_ppi_euclidean_dist_list, alpha=0.5, bins=25,
             label='positive pairs', color='orange')
plt.title("Euclidean distance counts")
plt.legend()
plt.grid()
plt.savefig("fig_hist_EUCLIDEAN.png")
pylab.clf()
pylab.cla()


# TEST PEARSON
# Obtain Pearson correlation between protein pairs
pos_ppi_pearson_list = []
print("Calculating Pearson scores for positive pairwise interactions")
for elut0, elut1, label in test_pos_dataloader:
    prot0, prot1 = elut0[0], elut1[0]
    elut0, elut1 = elut0[1], elut1[1]
    pos_ppi_pearson_list.append(scipy.stats.pearsonr(elut0[0][0], elut1[0][0])[0])

neg_ppi_pearson_list = []
print("Calculating Pearson scores for negative pairwise interactions")
for i, (elut0, elut1, label) in enumerate(test_neg_dataloader):
    prot0, prot1 = elut0[0], elut1[0]
    elut0, elut1 = elut0[1], elut1[1]
    neg_ppi_pearson_list.append(scipy.stats.pearsonr(elut0[0][0], elut1[0][0])[0])

    if i % 10 == 0:
        print(f"{i * 100 / len(test_neg_dataloader):.2f} %", end='\r')



# Plot the PDFs of the Pearson scores
x = np.linspace(-2, 2, 1000)
y = norm.pdf(x, loc=np.mean(neg_ppi_pearson_list), scale=np.std(neg_ppi_pearson_list))
y2 = norm.pdf(x, loc=np.mean(pos_ppi_pearson_list), scale=np.std(pos_ppi_pearson_list))

plt.plot(x,y,label="pearson_negative_pairs")
plt.plot(x,y2,label="pearson_positive_pairs")
plt.title("PDFs of Pearson scores")
plt.legend()
plt.grid()
plt.savefig("pos_neg_pairs_PEARSON.png")
plt.clf()
plt.cla()

# Plot the counts of positive/negative Pearson scores
sns.histplot(neg_ppi_pearson_list, alpha=0.5,
             label='negative pairs')
sns.histplot(pos_ppi_pearson_list, alpha=0.5, bins=50,
             label='positive pairs', color='orange')
plt.title("Pearson score counts")
plt.legend()
plt.grid()
plt.savefig("fig_hist_PEARSON.png")
pylab.clf()
pylab.cla()

# Plot the euclidean distance against the Pearson score
data_pearson_euclidean = {
        'Euclidean Distance': pos_ppi_euclidean_dist_list + neg_ppi_euclidean_dist_list,
        'Pearson Score': pos_ppi_pearson_list + neg_ppi_pearson_list,
        'Label': ['Positive'] * len(pos_ppi_euclidean_dist_list) +
                 ['Negative'] * len(neg_ppi_euclidean_dist_list)
}

df_pearson_euclidean = pd.DataFrame(data_pearson_euclidean).reset_index()
df_pearson_euclidean = df_pearson_euclidean.melt(id_vars=['Label', 'Euclidean Distance', 'Pearson Score'])

# Plot the pearson scores for positive PPIs against the euclidean score
#   Euclidean distance should be small, preferably as close to 0 as possible
scat1 = sns.scatterplot(pos_ppi_euclidean_dist_list, pos_ppi_pearson_list, s=10)
fig_scat1 = scat1.get_figure()
fig_scat1.savefig("pearson_vs_euc_scatter.png")
fig_scat1.clf()

# Plot a contour graph of the positive PPIs
#   We want to make the glob near (0, 0) larger
kde1 = sns.kdeplot(data=df_pearson_euclidean, x='Euclidean Distance', y='Pearson Score',
                   common_norm=False, hue='Label', levels=15)
fig_kde1 = kde1.get_figure()
fig_kde1.savefig("pearson_vs_euc_kde.png")
fig_kde1.clf()


