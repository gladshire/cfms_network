# Try noise reduction


# TODO: Create random sampling PR curve function for plotting precision-recall every epoch
# TODO: Try thresholding positive PPIs for Pearson > 0.1

# Co-fractionation Mass Spectrometry Siamese Network
# Written by Dr. Kevin Drew and Miles Woodcock-Girard
# For Drew Lab at University of Illinois at Chicago

# Goal: Develop some learned transformation that, when applied to the 1-D elution trace
#       vectors for proteins A and B, returns vectors with a lower Euclidean Distance if
#       A and B co-complex, and a higher Euclidean Distance if A and B do not co-complex.
#       More specifically, to achieve better performance than Pearson correlation.

# Data comes in form of .elut files, TSVs of protein elution traces from CF-MS experiments
# Example:
'''
          f1   f2   f3   f4   f5   f6   f7   f8   f9
PROT_ID1  0.0  0.0  0.0  2.0  5.0  2.0  0.0  0.0  0.0
PROT_ID2  2.0  3.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0
PROT_ID3  0.0  0.0  0.0  3.0  4.0  3.0  0.0  0.0  0.0

'''

# One sample consists of a pair of protein elution traces from the same .elut file
#
# Example:  ('PROT_ID1', [0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0])
#           ('PROT_ID2', [2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# 
# And a binary {0, 1} label where, for the purposes of math:
#
#   1: 'PROT_ID1' and 'PROT_ID2' do NOT co-complex (NEGATIVE / -)
#   0: 'PROT_ID1' and 'PROT_ID2' DO co-complex (POSITIVE / +)
#

# Want to split proteins from complexes into datasets as partners. All proteins from a given complex must
# be in the same set
# How many proteins shared between test and training set? C3, C2, C1 splits
#   C1 = when both proteins are in TRAIN and TEST
#   C2 = when one of the proteins is in TRAIN and TEST
#   C3 = when both proteins are in either TRAIN or TEST

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

# Use random subset samples of test/validation sets for speed during debugging
__FAST_VALID_TEST__ = False
PARAMETER_FILENAME = "jul29.pt"

# Program parameters
SEED = 5123
NUM_THREAD = 4
SUBSET_SIZE = 1000 # For if __FAST_VALID_TEST__ is True
SAMPLE_RATE = 10   # How many batches between loss samples (for plotting loss curve)

# Database directories
DATADIR_ELUT = "data/elut/"
DATADIR_PPIS = "data/ppi/"

# Training parameters
NUM_EPOCHS = 25
BATCH_SIZE = 128
LEARN_RATE = 1e-3
MOMENTUM = 0
EARLY_THRESHOLD = 0.01 # Loss value below which early stopping will occur

# Loss function parameters
TEMPERATURE = 1.0 # For cosine distance contrastive loss
SENSITIVITY = 3   # For changing behavior confidence function. Higher values push conf. towards lower ED
MARGIN = 2.0      # Minimum Euclidean separation for negative PPIs
MAX_ED = 3.75     # Euclidean distance threshold for 0% confidence 

# Set up manual seeding for random number generators
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


print("Retrieving data from files ...")

# Parse complex files to obtain accurate PPI labels for training data
# Read in all training complexes
training_complexes = []
training_complex_file = DATADIR_PPIS + "intact_complex_merge_20230309.train.txt"
f = open(training_complex_file)
for line in f.readlines():
    training_complexes.append(set(line.split()))
f.close()

# Read in positive PPIs from training data
training_pos_ppis = []
training_pos_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.train_ppis.txt"
f = open(training_pos_ppis_file)
for line in f.readlines():
    training_pos_ppis.append(set(line.split()))
f.close()

# Read in negative PPIs from training data
training_neg_ppis = []
training_neg_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.neg_train_ppis.txt"
f = open(training_neg_ppis_file)
for line in f.readlines():
    training_neg_ppis.append(set(line.split()))
f.close()


# Read in positive PPIs from test data
test_pos_ppis = []
test_pos_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.test_ppis.txt"
f = open(test_pos_ppis_file)
for line in f.readlines():
    test_pos_ppis.append(set(line.split()))
f.close()

# Read in negative PPIs from test data
test_neg_ppis = []
test_neg_ppis_file = DATADIR_PPIS + "intact_complex_merge_20230309.neg_test_ppis.txt"
f = open(test_neg_ppis_file)
for line in f.readlines():
    test_neg_ppis.append(set(line.split()))
f.close()


# Define list of .elut training data filenames
elut_data = [DATADIR_ELUT + "HEK293_EDTA_minus_SEC_control_20220626.elut",
             DATADIR_ELUT + "HEK293_EDTA_plus_SEC_treatment_20220626_trimmed.elut",
             DATADIR_ELUT + "Anna_HEK_urea_SEC_0M_050817_20220314b_trimmed.elut",
             DATADIR_ELUT + "Anna_HEK_urea_SEC_0p5M_052317_20220315_reviewed_trimmed.elut",
             DATADIR_ELUT + "Hs_helaN_1010_ACC.prot_count_mFDRpsm001.elut",
             DATADIR_ELUT + "Hs_HCW_1.elut",
             DATADIR_ELUT + "Hs_HCW_2.elut",
             DATADIR_ELUT + "Hs_HCW_3.elut",
             DATADIR_ELUT + "Hs_HCW_4.elut",
             DATADIR_ELUT + "Hs_HCW_5.elut",
             DATADIR_ELUT + "Hs_HCW_6.elut",
             DATADIR_ELUT + "Hs_HCW_7.elut",
             DATADIR_ELUT + "Hs_HCW_8.elut",
             DATADIR_ELUT + "Hs_HCW_9.elut",
             DATADIR_ELUT + "Hs_IEX_1.elut",
             DATADIR_ELUT + "Hs_IEX_2.elut",
             DATADIR_ELUT + "U2OS_cells_SEC_Kirkwood_2013_rep1.elut",
             DATADIR_ELUT + "U2OS_cells_SEC_Kirkwood_2013_rep2.elut",
             DATADIR_ELUT + "U2OS_cells_SEC_Kirkwood_2013_rep3.elut",
             DATADIR_ELUT + "HEK_293_T_cells_SEC_Mallam_2019_C1.elut",
             DATADIR_ELUT + "HEK_293_T_cells_SEC_Mallam_2019_C2.elut",
             DATADIR_ELUT + "NTera2_embryonal_carcinoma_stem_cells_SEC_Moutaoufik_2019_R1.elut",
             DATADIR_ELUT + "NTera2_embryonal_carcinoma_stem_cells_SEC_Moutaoufik_2019_R2.elut",
             DATADIR_ELUT + "NTera2_embryonal_carcinoma_stem_cells_SEC_Moutaoufik_2019_2_R1.elut",
             DATADIR_ELUT + "NTera2_embryonal_carcinoma_stem_cells_SEC_Moutaoufik_2019_2_R2.elut"]


# Assemble list of preprocessed, normalized .elut dataframes
elut_list = []
for elut_file in elut_data:
    print(f"Parsing \'{elut_file}\' ...")
    elut_df = pd.read_csv(elut_file, sep='\t', index_col=0)

    #elut_df = elut_df.set_index('Unnamed: 0')
    elut_t10_df = elut_df[elut_df.sum(axis=1) >= 10]
    # Row-max normalization (all values in elution trace divided by maximum value in that trace)
    #elut_t10_rwn_df = elut_t10_df.div(elut_t10_df.max(axis=1), axis=0)

    # Row-sum normalization (all values in elution trace divided by total PSMs in that trace)
    elut_t10_rsm_df = elut_t10_df.div(elut_t10_df.sum(axis=1), axis=0)

    # Quantile normalization (quantiles of all traces in .elut file are aligned)
    #elut_t10_qtn_df = qnorm.quantile_normalize(elut_t10_df, axis=0)

    # Add normalized dataframe to list containing elution data
    elut_list.append(elut_t10_rsm_df)


print("Elution data obtained. Now preparing ...")


# Obtain set of all unique proteins across all elution data
elut_proteins = set()
for ed in elut_list:
    elut_proteins = elut_proteins.union(ed.index)

# Only keep protein pairs contained within the elution training data
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

def plot_loss(iteration, loss_lists, labels, xaxis, title, filename=None):
    if len(labels) != len(loss_lists):
        print("ERROR: Mismatch in plot lengths")
        print(len(labels))
        print(len(loss_tuple))
        exit()
    for loss_series in loss_lists:
        plt.plot(iteration, loss_series)
    plt.grid()
    plt.title(title)
    plt.xlabel(xaxis)
    plt.legend(labels)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    plt.cla()

def get_aupr(precision, recall):
    aupr = 0.0
    for i in range(1, len(precision)):
        # Get the width (difference in recall) and height (avg. of precisions)
        width = recall[i] - recall[i - 1]
        height = (precision[i] + precision[i - 1]) / 2.0
        aupr += width * height
    return aupr

# Custom function for converting Euclidean distance to confidence score, 
# where s is a parameter making the score more sensitive to smaller distances
def euclidean_to_confidence(euc_dist, max_euc_dist, s):
    #return 1 / (1 + euclidean_distance)
    confidence = (((max_euc_dist + 1) - euc_dist)**s - 1) / ((max_euc_dist + 1)**s - 1)
    return confidence


# Wrap elution pair data into PyTorch dataset
class elutionPairDataset(Dataset):
    def __init__(self, elutdf_list, pos_ppis, neg_ppis, transform=False, input_size=128, filterPearson=False):
        self.elut_df_list = elutdf_list
        self.ppis = []
        self.labels = []
        self.ppis_elut_ids = []

        for elut_id, elut_df in enumerate(self.elut_df_list):
            # miles: Loading the elut dataframe as a set vastly reduces runtime of dataset initialization
            elut_df_index = set(elut_df.index)

            pos_ppis_elut = [pppi for pppi in pos_ppis if len(pppi.intersection(elut_df_index)) == 2]
            neg_ppis_elut = [nppi for nppi in neg_ppis if len(nppi.intersection(elut_df_index)) == 2]

            if filterPearson:
                # Remove low-Pearson samples from positive PPIs
                pos_ppis_elut = [pppi for pppi in pos_ppis_elut if scipy.stats.pearsonr(
                    torch.from_numpy(elut_df.T[list(pppi)[0]].values).float(),
                    torch.from_numpy(elut_df.T[list(pppi)[1]].values).float())[0] >= 0.25]

                # Remove high-Pearson samples from negative PPIs
                neg_ppis_elut = [nppi for nppi in neg_ppis_elut if scipy.stats.pearsonr(
                    torch.from_numpy(elut_df.T[list(nppi)[0]].values).float(),
                    torch.from_numpy(elut_df.T[list(nppi)[1]].values).float())[0] <= 0.75]

            self.ppis = self.ppis + pos_ppis_elut + neg_ppis_elut
            self.labels = self.labels + [0]*len(pos_ppis_elut) + [1]*len(neg_ppis_elut)
            self.ppis_elut_ids = self.ppis_elut_ids + [elut_id]*len(pos_ppis_elut) + [elut_id]*len(neg_ppis_elut)

        self.transform = transform
        self.input_size = input_size

    def __getitem__(self, index):
        pair = list(self.ppis[index])
        elut_id = self.ppis_elut_ids[index]
        elut_df = self.elut_df_list[elut_id]

        prot0 = pair[0]
        prot1 = pair[1]

        elut0 = (prot0, torch.from_numpy(elut_df.T[pair[0]].values).float())
        elut1 = (prot1, torch.from_numpy(elut_df.T[pair[1]].values).float())

        if self.transform:
            elut0 = (prot0, nn.functional.pad(elut0[1], (self.input_size - elut0[1].size()[0], 0)))
            elut1 = (prot1, nn.functional.pad(elut1[1], (self.input_size - elut1[1].size()[0], 0)))

        elut0 = (elut0[0], elut0[1].unsqueeze(0))
        elut1 = (elut1[0], elut1[1].unsqueeze(0))

        return elut0, elut1, self.labels[index], elut_id

    def __len__(self):
        return len(self.ppis)



# Instantiate the training dataset for the model
train_siamese_dataset = elutionPairDataset(elutdf_list=elut_list,
                                           pos_ppis = train_pos_ppis,
                                           neg_ppis = train_neg_ppis,
                                           transform=True)
#                                           filterPearson=True)
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


# Define the siamese neural network
class siameseNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(siameseNet, self).__init__()

        # Elution tensor input: (batch_size, channels, num_fractions)

        # CONVOLUTION OUTPUT DIMENSIONS:
        #   Size: (num_fractions - kernel_size + 2*padding / stride) + 1)

        self.cnn1 = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=4,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(4),
                nn.ELU(inplace=True),

                nn.Conv1d(in_channels=4, out_channels=16,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(16),
                nn.ELU(inplace=True),

        )

        self.rnn = nn.LSTM(input_size=1, hidden_size=64, num_layers=1,
                           bidirectional=True, batch_first=True)

        self.tns = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=1,
                                           nhead=1, dim_feedforward=64,
                                           batch_first=True),
                num_layers=3
        )

        self.cnn2 = nn.Sequential(
                nn.ConvTranspose1d(in_channels=16, out_channels=1,
                                   kernel_size=1, stride=1, padding=1),

        )

        self.fc1 = nn.Sequential(

                nn.Linear(29, 16),
                nn.ReLU(inplace=True),

                nn.Linear(16, 8),
                nn.ReLU(inplace=True),

        )


    # Function called on both images, x, to determine their similarity
    def forward_once(self, x):
        # Apply convolutional encoding
        #print(x.shape)
        y = self.cnn1(x)

        # Prepare for recurrent layers
        #print(y.shape)
        #y = y.permute(0, 2, 1)

        # Apply bidirectional recurrency
        #y, _ = self.rnn(y)

        # Apply transformer layer
        #print(y.shape)
        #y = self.tns(y)

        # Prepare for convolutional decoding
        #print(y.shape)
        #y = y.permute(0, 2, 1)

        # Apply convolutional decoding
        #print(y.shape)
        y = self.cnn2(y)
        #print(y.shape)

        # Flatten output to work with fully connected layer
        y = y.reshape(y.size()[0], -1)

        # Prepare Pearson input for network
        #pcc = pcc.view(-1, 1).expand(y.size(0), 1)
        #y = torch.cat((y, pcc), dim=1)

        # Apply fully connected layer
        # Potentiall add Pearson correlation coefficient here
        #y = self.fc1(y)

        return y

    # Main forward function
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

# Define contrastive loss function, using Euclidean distance
class contrastiveLossEuclidean(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(contrastiveLossEuclidean, self).__init__()
        self.margin = margin

    # Contrastive loss calculation
    def forward(self, output1, output2, label):
        euclidean_dist = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1-label) * torch.pow(euclidean_dist, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))
        return loss

# Define SimCLR contrastive loss, based on cosine distance
class contrastiveLossCosineDistance(nn.Module):
    def __init__(self, margin=1.0, tau=1.0):
        super(contrastiveLossCosineDistance, self).__init__()
        self.margin = margin
        self.tau = tau

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)

        cos_dist = 1.0 - cos_sim
        cos_dist_t = cos_dist / self.tau

        loss = torch.mean((1-label) * torch.pow(cos_dist_t, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - cos_dist_t, min=0.0), 2))
        return loss


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
criterion = contrastiveLossEuclidean(margin=MARGIN)
#criterion = contrastiveLossCosineDistance(tau=TEMPERATURE)
#criterion = nn.CosineEmbeddingLoss(margin=0.8, reduction='sum')


# Choose optimizer algorithm
#optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

# Choose learning rate scheduler
# miles: ReduceLROnPlateau works great dynamically, StepLR good for exploring ruggedness
#        of loss topology
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Zero the gradients
#optimizer.zero_grad()

counter = []
loss_hist = []
epoch_hist = []
avg_test_loss_hist = []
avg_valid_loss_hist = []
avg_train_loss_hist = []
iteration_num = 0
valid_iteration_num = 0

min_avg_valid_loss = 1e9
min_avg_test_loss = 1e9



def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)                 

#net.apply(weights_init)


# Training loop
if trainNet:
    #print("Instantiating model training with following architecture:")
    #print(net)

    print("\nWith following hyperparameters")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learn rate: {LEARN_RATE}")

    for epoch in range(NUM_EPOCHS):
        print("\n================================")
        print(f"Epoch: {epoch+1} of {NUM_EPOCHS}")
        print("================================\n")
    
        # Iterate over batches
        net.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for i, (elut0, elut1, label, elut_id) in enumerate(train_dataloader, 0):
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
            output1, output2 = net(elut0, elut1)

            # Pass outputs, label to the contrastive loss function
            train_loss_contrastive = criterion(output1, output2, label)

            # Perform backpropagation
            train_loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Update training loss series for plotting
            if i % SAMPLE_RATE == 0:
                print(f"  Batch [{i} / {num_batches}] Training Loss: {train_loss_contrastive.item()}")
                iteration_num += SAMPLE_RATE

                counter.append(iteration_num)
                loss_hist.append(train_loss_contrastive.item())
  
        # Produce training loss curve figure PNG
        plot_loss(counter, [loss_hist], ["Training loss"], title="Contrastive Loss - Train",
                  xaxis="Batches", filename=f"train_{epoch+1}.png")

        # Test model on validation set 
        net.eval()
        valid_loss = 0.0
        num_batches_valid = len(valid_dataloader)
        for valid_i, (valid_elut0, valid_elut1, valid_label, valid_elut_id) in enumerate(valid_dataloader, 0):
            pccListValid = []

            # Obtain protein IDs
            prot0, prot1 = valid_elut0[0], valid_elut1[0]
            for valid_ppi0, valid_ppi1 in zip(valid_elut0[1], valid_elut1[1]):
                pccListValid.append(scipy.stats.pearsonr(valid_ppi0[0], valid_ppi1[0])[0])

            pcc = torch.tensor(pccListValid, dtype=torch.float32).cuda()

            # Send elution data, labels to CUDA
            valid_elut0, valid_elut1, valid_label = valid_elut0[1].cuda(), valid_elut1[1].cuda(), valid_label.cuda()

            # Pass two elution traces into network and obtain two outputs
            valid_output1, valid_output2 = net(valid_elut0, valid_elut1)

            # Pass outputs, label to the contrastive loss function
            valid_loss_contrastive = criterion(valid_output1, valid_output2, valid_label)

            # Add to total validation loss
            valid_loss += valid_loss_contrastive.item()

            # Update validation loss series for plotting
            if valid_i % 5 == 0 and valid_i > 0:
                avg_valid_loss = valid_loss / valid_i
                print(f"Now running model on validation set ... {valid_i * 100 / len(valid_dataloader):.4f} %  |  Avg. Loss: {avg_valid_loss}", end='\r')

        # Obtain the average validation set loss per batch
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
            for test_i, (test_elut0, test_elut1, test_label, test_elut_id) in enumerate(test_dataloader, 0):
                pccListTest = []
                # Obtain protein IDs
                prot0, prot1 = elut0[0], elut1[0]
                for test_ppi0, test_ppi1 in zip(test_elut0[1], test_elut1[1]):
                    pccListTest.append(scipy.stats.pearsonr(test_ppi0[0], test_ppi1[0])[0])

                pcc = torch.tensor(pccListTest, dtype=torch.float32).cuda()

                # Send test elution data, labels to CUDA
                test_elut0, test_elut1, test_label = test_elut0[1].cuda(), test_elut1[1].cuda(), test_label.cuda()

                # Padd two elution traces into network and obtain two outputs
                output1, output2 = net(test_elut0, test_elut1)

                # Pass outputs, label to the contrastive loss function
                loss_test_contrastive = criterion(output1, output2, test_label)

                # Add to total test loss
                test_loss += loss_test_contrastive

                if test_i % 5 == 0 and test_i > 0:
                    avg_test_loss = test_loss / test_i
                    print(f"Now running model on test set ... {test_i * 100 / len(test_dataloader):.4f} %  |  Avg. Loss {avg_test_loss}", end='\r')

        # Calculate average loss on test dataset
        avg_test_loss = test_loss / len(test_dataloader)

        # Get new minimum average test set loss for final reporting
        if avg_test_loss < min_avg_test_loss:
            min_avg_test_loss = avg_test_loss

        # Append to avg loss vs epochs series
        epoch_hist.append(epoch+1)
        avg_test_loss_hist.append(avg_test_loss)
        avg_valid_loss_hist.append(avg_valid_loss)

        # Summarize end of epoch metrics
        print(f"\nEnd of epoch summary")
        print(f"  Average validation loss: {avg_valid_loss}")
        print(f"  Average testing loss: {avg_test_loss}")

        # Plot average test set loss vs epoch number 
        if epoch != 0:
            plot_loss(epoch_hist, [avg_test_loss_hist, avg_valid_loss_hist],
                      ["Testing loss", "Validation loss"],
                      title="Average Contrastive Loss", xaxis="Epochs",
                      filename=f"epoch_{epoch+1}.png")

        # Early stopping according to user-defined threshold
        if avg_test_loss < EARLY_THRESHOLD:
            break

# Save the model weights
torch.save(net.state_dict(), PARAMETER_FILENAME)

# Run model and Pearson on positive PPIs only
test_pos_elution_pair_dataset = elutionPairDataset(elutdf_list=elut_list,
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

pos_ppi_euclidean_dist_list = []
pos_ppi_pearson_list = []
pos_ppi_conf_output_file = open("pos_ppi_conf.txt", 'w')
pos_ppi_conf_output_file.write("PPI\tEuclidean Distance\tPearson Score\tConfidence Score\n")
pos_ppi_conf_list = []
pos_ppi_lab_list = []
for i, (pos_elut0, pos_elut1, pos_label, elut_id) in enumerate(test_pos_dataloader):

    # Obtain protein IDs
    prot0, prot1 = pos_elut0[0], pos_elut1[0]

    # Obtain elution traces
    pos_elut0, pos_elut1 = pos_elut0[1], pos_elut1[1]

    # Obtain Pearson correlation
    pos_ppi_pearson_list.append(scipy.stats.pearsonr(pos_elut0[0][0], pos_elut1[0][0])[0])

    # Run model on data
    output1, output2 = net(pos_elut0.cuda(), pos_elut1.cuda())

    # Get Euclidean distance b/t model outputs
    euclidean_dist = F.pairwise_distance(output1, output2)

    # Get confidence score of similarity based on Euclidean distance
    confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)

    # Write prediction to text file, with following line-wise format
    #   prot1:prot2 euc_dist confidence label
    pos_ppi_conf_output_file.write(str(prot0[0]) + ":" + str(prot1[0]) + f"\t{euclidean_dist.item():.4f}\t{pos_ppi_pearson_list[i]:.4f}\t{confidence.item():.4f}\t{elut_id}\n")

    pos_ppi_euclidean_dist_list.append(euclidean_dist.item())
    pos_ppi_conf_list.append(confidence.item())
    pos_ppi_lab_list.append(pos_label.item())

    if i % 5 == 0:
        print(f"Calculating Euclidean distances, Pearson scores for positive pairwise interactions ... {i * 100 / len(test_pos_dataloader):.2f} %", end='\r')
pos_ppi_conf_output_file.close()
print("Calculating Euclidean distances, Pearson scores for positive pairwise interactions ... done            ")

# Run model and Pearson on negative PPIs only
test_neg_elution_pair_dataset = elutionPairDataset(elutdf_list=elut_list,
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

# Iterate over negative samples, getting Pearson scores and post-transform Euclidean distance for each PPI
neg_ppi_euclidean_dist_list = []
neg_ppi_pearson_list = []
neg_ppi_conf_output_file = open("neg_ppi_conf.txt", 'w')
neg_ppi_conf_output_file.write("PPI\tEuclidean Distance\tPearson Score\tConfidence Score\n")
neg_ppi_conf_list = []
neg_ppi_lab_list = []
for i, (neg_elut0, neg_elut1, neg_label, elut_id) in enumerate(test_neg_dataloader):

    # Obtain protein IDs
    prot0, prot1 = neg_elut0[0], neg_elut1[0]

    # Obtain elution traces
    neg_elut0, neg_elut1 = neg_elut0[1], neg_elut1[1]

    # Obtain Pearson correlation
    neg_ppi_pearson_list.append(scipy.stats.pearsonr(neg_elut0[0][0], neg_elut1[0][0])[0])

    # Run model on data
    output1, output2 = net(neg_elut0.cuda(), neg_elut1.cuda())

    # Get Euclidean distance b/t model outputs
    euclidean_dist = F.pairwise_distance(output1, output2)

    # Get confidence score of similarity based on Euclidean distance
    confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)
    neg_ppi_conf_output_file.write(str(prot0[0]) + ":" + str(prot1[0]) + f"\t{euclidean_dist.item():.4f}\t{neg_ppi_pearson_list[i]:.4f}\t{confidence.item():.4f}\t{elut_id}\n")

    neg_ppi_euclidean_dist_list.append(euclidean_dist.item())
    neg_ppi_conf_list.append(confidence.item())
    neg_ppi_lab_list.append(neg_label.item())

    if i % 5 == 0:
        print(f"Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... {i * 100 / len(test_neg_dataloader):.2f} %", end='\r')
neg_ppi_conf_output_file.close()
print("Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... done            ")

# Plot the PDF of the euclidean distance between positive and negative protein pairs
x = np.linspace(-5,5,1000)
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

# Plot the PDFs of the Pearson scores
x = np.linspace(-2, 2, 1000)
y = norm.pdf(x, loc=np.mean(neg_ppi_pearson_list), scale=np.std(neg_ppi_pearson_list))
y2 = norm.pdf(x, loc=np.mean(pos_ppi_pearson_list), scale=np.std(pos_ppi_pearson_list))

plt.plot(x,y,label="pearson_negative_pairs")
plt.plot(x,y2,label="pearson_positive_pairs")
plt.title("PDFs of Pearson scores")
plt.legend()
plt.grid()
plt.savefig(f"pos_neg_pairs_PEARSON.png")
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
plt.savefig(f"fig_hist_PEARSON.png")
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
scat1.set_xlim(-1, 12)
fig_scat1 = scat1.get_figure()
fig_scat1.savefig("pearson_vs_euc_scatter.png")
fig_scat1.clf()

# Plot a contour graph of the positive PPIs
#   We want to make the glob near (0, 0) larger
kde1 = sns.kdeplot(data=df_pearson_euclidean, x='Euclidean Distance', y='Pearson Score',
                   common_norm=False, hue='Label', levels=15)
kde1.set_xlim(-1, 12)
fig_kde1 = kde1.get_figure()
fig_kde1.savefig("pearson_vs_euc_kde.png")
fig_kde1.clf()

# Get labels, confidences, Euclidean distances, Pearson coefficients of test points
y_labels = pos_ppi_lab_list + neg_ppi_lab_list
y_confs = pos_ppi_conf_list + neg_ppi_conf_list
y_dists = pos_ppi_euclidean_dist_list + neg_ppi_euclidean_dist_list
y_pearson = pos_ppi_pearson_list + neg_ppi_pearson_list


# Invert y_labels series s.t. '1' is positive and '0' is negative
y_labels_not = [1.0 - y for y in y_labels]

# Sort inverted y_labels series according to ascending Euclidean distance
y_sort_euclidean = [y for _, y in sorted(zip(y_dists, y_labels_not), key = lambda pair: pair[0], reverse=False)]

# Sort inverted y_labels series according to descending Pearson correlation
y_sort_pearson = [y for _, y in sorted(zip(y_pearson, y_labels_not), key = lambda pair: pair[0], reverse=True)]

true_pos = sum(y_labels_not)

# Calculate precision and recall for Euclidean
precision_euclidean = []
recall_euclidean = []
true_pos_euclidean = 0

for i, y in enumerate(y_sort_euclidean, start=1):
    true_pos_euclidean += y
    curr_precision = true_pos_euclidean / i
    curr_recall = true_pos_euclidean / true_pos
    precision_euclidean.append(curr_precision)
    recall_euclidean.append(curr_recall)
    if i % 100 == 0:
        print(f"Calculating precision-recall for model ... {i * 100 / len(y_sort_euclidean):.4f} %", end='\r')
    print("Calculating precision-recall for model ... done        ")

# Calculate precision and recall for Pearson
precision_pearson = []
recall_pearson = []
true_pos_pearson = 0

for i, y in enumerate(y_sort_pearson, start=1):
    true_pos_pearson += y
    curr_precision = true_pos_pearson / i
    curr_recall = true_pos_pearson / true_pos
    precision_pearson.append(curr_precision)
    recall_pearson.append(curr_recall)
    if i % 100 == 0:
        print(f"Calculating precision-recall for Pearson ... {i * 100 / len(y_sort_pearson):.4f} %", end='\r')
    print("Calculating precision-recall for Pearson ... done        ")

    # Plot Precision-Recall curves of Euclidean distance and Pearson as PPI predictors
    #plt.plot(recall_euclidean, precision_euclidean, color='blue')
    plt.plot(recall_pearson, precision_pearson, color='orange')
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("pr_curve_{k+1}.png")
    plt.legend(["Euclidean PR", "Pearson PR"])
    plt.clf()
    plt.cla()

    # Obtain area under PR curves
    aupr_euclidean = get_aupr(precision_euclidean, recall_euclidean)
    aupr_pearson = get_aupr(precision_pearson, recall_pearson)
# Print metrics to txt file
with open("results-final_{k+1}.log", 'w') as outFile:
    outFile.write(f"Minimum Average Validation Loss: {min_avg_valid_loss:.4f}\n")
    outFile.write(f"Minimum Average Test Loss: {min_avg_test_loss:.4f}\n")
    outFile.write(f"Difference Between ED Means of Pos/Neg PPIs: {diff_means_pos_neg_pdf:.4f}\n")
    outFile.write(f"Area Under Euclidean PR Curve: {aupr_euclidean:.4f}\n")
    outFile.write(f"Area Under Pearson PR Curve: {aupr_pearson:.4f}\n")
outFile.close()

for k, elut in enumerate(elut_list):
    # Run model and Pearson on positive PPIs only
    test_pos_elution_pair_dataset = elutionPairDataset(elutdf_list=[elut],
                                                       pos_ppis=test_pos_ppis,
                                                       neg_ppis=[],
                                                       transform=True)
    print(len(test_pos_elution_pair_dataset))

    subset_indices = torch.randperm(len(test_pos_elution_pair_dataset))[:SUBSET_SIZE]
    subset_test_pos_elution_pair_dataset = Subset(test_pos_elution_pair_dataset, subset_indices)

    if __FAST_VALID_TEST__:
        test_pos_dataloader = DataLoader(subset_test_pos_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)
    else:
        test_pos_dataloader = DataLoader(test_pos_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)

    pos_ppi_euclidean_dist_list = []
    pos_ppi_pearson_list = []
    pos_ppi_conf_output_file = open(f"pos_ppi_conf_{k+1}.txt", 'w')
    pos_ppi_conf_output_file.write("PPI\tEuclidean Distance\tPearson Score\tConfidence Score\n")
    pos_ppi_conf_list = []
    pos_ppi_lab_list = []
    if len(test_pos_elution_pair_dataset) > 0:
        for i, (pos_elut0, pos_elut1, pos_label, elut_id) in enumerate(test_pos_dataloader):

            # Obtain protein IDs
            prot0, prot1 = pos_elut0[0], pos_elut1[0]

            # Obtain elution traces
            pos_elut0, pos_elut1 = pos_elut0[1], pos_elut1[1]

            # Obtain Pearson correlation
            pos_ppi_pearson_list.append(scipy.stats.pearsonr(pos_elut0[0][0], pos_elut1[0][0])[0])

            # Run model on data
            output1, output2 = net(pos_elut0.cuda(), pos_elut1.cuda())

            # Get Euclidean distance b/t model outputs
            euclidean_dist = F.pairwise_distance(output1, output2)

            # Get confidence score of similarity based on Euclidean distance
            confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)

            # Write prediction to text file, with following line-wise format
            #   prot1:prot2 euc_dist confidence label
            pos_ppi_conf_output_file.write(str(prot0[0]) + ":" + str(prot1[0]) + f"\t{euclidean_dist.item():.4f}\t{pos_ppi_pearson_list[i]:.4f}\t{confidence.item():.4f}\t{elut_id}\n")

            pos_ppi_euclidean_dist_list.append(euclidean_dist.item())
            pos_ppi_conf_list.append(confidence.item())
            pos_ppi_lab_list.append(pos_label.item())
        
            if i % 5 == 0:
                print(f"Calculating Euclidean distances, Pearson scores for positive pairwise interactions ... {i * 100 / len(test_pos_dataloader):.2f} %", end='\r')
        #pos_ppi_conf_output_file.close()
        print("Calculating Euclidean distances, Pearson scores for positive pairwise interactions ... done            ")
    else:
        print("Dataset is empty!")

    # Run model and Pearson on negative PPIs only
    test_neg_elution_pair_dataset = elutionPairDataset(elutdf_list=[elut],
                                                       pos_ppis=[],
                                                       neg_ppis=test_neg_ppis,
                                                       transform=True)
    subset_indices = torch.randperm(len(test_neg_elution_pair_dataset))[:SUBSET_SIZE]
    subset_test_neg_elution_pair_dataset = Subset(test_neg_elution_pair_dataset, subset_indices)

    if __FAST_VALID_TEST__:
        test_neg_dataloader = DataLoader(subset_test_neg_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)
    else:
        test_neg_dataloader = DataLoader(test_neg_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)

    # Iterate over negative samples, getting Pearson scores and post-transform Euclidean distance for each PPI
    neg_ppi_euclidean_dist_list = []
    neg_ppi_pearson_list = []
    neg_ppi_conf_output_file = open(f"neg_ppi_conf_{k+1}.txt", 'w')
    neg_ppi_conf_output_file.write("PPI\tEuclidean Distance\tPearson Score\tConfidence Score\n")
    neg_ppi_conf_list = []
    neg_ppi_lab_list = []
    for i, (neg_elut0, neg_elut1, neg_label, elut_id) in enumerate(test_neg_dataloader):

        # Obtain protein IDs
        prot0, prot1 = neg_elut0[0], neg_elut1[0]

        # Obtain elution traces
        neg_elut0, neg_elut1 = neg_elut0[1], neg_elut1[1]

        # Obtain Pearson correlation
        neg_ppi_pearson_list.append(scipy.stats.pearsonr(neg_elut0[0][0], neg_elut1[0][0])[0])

        # Run model on data
        output1, output2 = net(neg_elut0.cuda(), neg_elut1.cuda())

        # Get Euclidean distance b/t model outputs
        euclidean_dist = F.pairwise_distance(output1, output2)

        # Get confidence score of similarity based on Euclidean distance
        confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)
        neg_ppi_conf_output_file.write(str(prot0[0]) + ":" + str(prot1[0]) + f"\t{euclidean_dist.item():.4f}\t{neg_ppi_pearson_list[i]:.4f}\t{confidence.item():.4f}\t{elut_id}\n")

        neg_ppi_euclidean_dist_list.append(euclidean_dist.item())
        neg_ppi_conf_list.append(confidence.item())
        neg_ppi_lab_list.append(neg_label.item())

        if i % 5 == 0:
            print(f"Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... {i * 100 / len(test_neg_dataloader):.2f} %", end='\r')
    neg_ppi_conf_output_file.close()
    print("Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... done            ")

    # Plot the PDF of the euclidean distance between positive and negative protein pairs
    x = np.linspace(-5,5,1000)
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
    pylab.savefig(f"pos_neg_pairs_EUCLIDEAN_{k+1}.png")
    pylab.clf()
    pylab.cla()


    sns.histplot(neg_ppi_euclidean_dist_list, alpha=0.5,
                 label='negative pairs')
    sns.histplot(pos_ppi_euclidean_dist_list, alpha=0.5, bins=25,
                 label='positive pairs', color='orange')
    plt.title("Euclidean distance counts")
    plt.legend()
    plt.grid()
    plt.savefig(f"fig_hist_EUCLIDEAN_{k+1}.png")
    pylab.clf()
    pylab.cla()

    # Plot the PDFs of the Pearson scores
    x = np.linspace(-2, 2, 1000)
    y = norm.pdf(x, loc=np.mean(neg_ppi_pearson_list), scale=np.std(neg_ppi_pearson_list))
    y2 = norm.pdf(x, loc=np.mean(pos_ppi_pearson_list), scale=np.std(pos_ppi_pearson_list))
    plt.plot(x,y,label="pearson_negative_pairs")
    plt.plot(x,y2,label="pearson_positive_pairs")
    plt.title("PDFs of Pearson scores")
    plt.legend()
    plt.grid()
    plt.savefig(f"pos_neg_pairs_PEARSON_{k+1}.png")
    plt.clf()
    plt.cla()

    # Plot the counts of positive/negative Pearson scores
    sns.histplot(neg_ppi_pearson_list, alpha=0.5,
                 label='negative pairs')
    sns.histplot(pos_ppi_pearson_list, alpha=0.5, bins=50,
                 label='positive pairs', color='orange')
    plt.title(f"Pearson score counts")
    plt.legend()
    plt.grid()
    plt.savefig(f"fig_hist_PEARSON_{k+1}.png")
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
    scat1.set_xlim(-1, 12)
    fig_scat1 = scat1.get_figure()
    fig_scat1.savefig(f"pearson_ed_pos_scatter_{k+1}.png")
    fig_scat1.clf()

    # Plot a contour graph of the positive PPIs
    #   We want to make the glob near (0, 0) larger
    kde1 = sns.kdeplot(data=df_pearson_euclidean, x='Euclidean Distance', y='Pearson Score',
                       common_norm=False, hue='Label', levels=15)
    kde1.set_xlim(-1, 12)
    fig_kde1 = kde1.get_figure()
    fig_kde1.savefig(f"pearson_ed_contour_{k+1}.png")
    fig_kde1.clf()

    # Get labels, confidences, Euclidean distances, Pearson coefficients of test points
    y_labels = pos_ppi_lab_list + neg_ppi_lab_list
    y_confs = pos_ppi_conf_list + neg_ppi_conf_list
    y_dists = pos_ppi_euclidean_dist_list + neg_ppi_euclidean_dist_list
    y_pearson = pos_ppi_pearson_list + neg_ppi_pearson_list


    # Invert y_labels series s.t. '1' is positive and '0' is negative
    y_labels_not = [1.0 - y for y in y_labels]

    # Sort inverted y_labels series according to ascending Euclidean distance
    y_sort_euclidean = [y for _, y in sorted(zip(y_dists, y_labels_not), key = lambda pair: pair[0], reverse=False)]

    # Sort inverted y_labels series according to descending Pearson correlation
    y_sort_pearson = [y for _, y in sorted(zip(y_pearson, y_labels_not), key = lambda pair: pair[0], reverse=True)]

    true_pos = sum(y_labels_not)

    # Calculate precision and recall for Euclidean
    precision_euclidean = []
    recall_euclidean = []
    true_pos_euclidean = 0

    for i, y in enumerate(y_sort_euclidean, start=1):
        true_pos_euclidean += y
        curr_precision = true_pos_euclidean / i
        curr_recall = true_pos_euclidean / true_pos
        precision_euclidean.append(curr_precision)
        recall_euclidean.append(curr_recall)
        if i % 100 == 0:
            print(f"Calculating precision-recall for model ... {i * 100 / len(y_sort_euclidean):.4f} %", end='\r')
    print("Calculating precision-recall for model ... done      ")


    # Calculate precision and recall for Pearson
    precision_pearson = []
    recall_pearson = []
    true_pos_pearson = 0

    for i, y in enumerate(y_sort_pearson, start=1):
        true_pos_pearson += y
        curr_precision = true_pos_pearson / i
        curr_recall = true_pos_pearson / true_pos
        precision_pearson.append(curr_precision)
        recall_pearson.append(curr_recall)
        if i % 100 == 0:
            print(f"Calculating precision-recall for Pearson ... {i * 100 / len(y_sort_pearson):.4f} %", end='\r')
    print("Calculating precision-recall for Pearson ... done        ")

    # Plot Precision-Recall curves of Euclidean distance and Pearson as PPI predictors
    plt.plot(recall_euclidean, precision_euclidean, color='blue')
    plt.plot(recall_pearson, precision_pearson, color='orange')
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"pr_curve_{k+1}.png")
    plt.legend(["Euclidean PR", "Pearson PR"])
    plt.clf()
    plt.cla()

    # Obtain area under PR curves
    aupr_euclidean = get_aupr(precision_euclidean, recall_euclidean)
    aupr_pearson = get_aupr(precision_pearson, recall_pearson)
    # Print metrics to txt file
    with open(f"results-final_{k+1}.log", 'w') as outFile:
        outFile.write(f"Minimum Average Validation Loss: {min_avg_valid_loss:.4f}\n")
        outFile.write(f"Minimum Average Test Loss: {min_avg_test_loss:.4f}\n")
        outFile.write(f"Difference Between ED Means of Pos/Neg PPIs: {diff_means_pos_neg_pdf:.4f}\n")
        outFile.write(f"Area Under Euclidean PR Curve: {aupr_euclidean:.4f}\n")
        outFile.write(f"Area Under Pearson PR Curve: {aupr_pearson:.4f}\n")
        outFile.close()
