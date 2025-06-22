# TODO: Create random sampling PR curve function for plotting precision-recall every epoch
# TODO: Try to obtain 2284 Train set pos PPIs size from HEK293_EDTA_minus.elut file
#  - Which is the network missing?

# Co-fractionation Mass Spectrometry Siamese Network
# Written by Dr. Kevin Drew and Miles Woodcock-Girard
# For Drew Lab at University of Illinois at Chicago
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
import seaborn as sns

import curses
import sys
import os

# Use random subset samples for speeding up run on test sets during debugging
__FAST_TEST__ = False
PARAMETER_FILENAME = "cfms_ppi_network_bdlstm_adam_1e-4_256.pt"
FIGURES_DIRECTORY = f"{PARAMETER_FILENAME.split(".")[0]}_figs"
if not os.path.exists(FIGURES_DIRECTORY):
    os.makedirs(FIGURES_DIRECTORY)

# Program parameters
SEED = 5123
NUM_THREAD = 4
SUBSET_SIZE = 1000 # For if __FAST_VALID_TEST__ is True
SAMPLE_RATE = 50   # How many batches between loss samples (for printing to terminal, plotting loss curve)

# Database directories
DATADIR_ELUT = "data/elut/"
DATADIR_PPIS = "data/ppi/"

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARN_RATE = 1e-4
MOMENTUM = 0.9
EARLY_THRESHOLD = 0.01 # Loss value below which early stopping will occur

# Loss function parameters
TEMPERATURE = 1.0 # For cosine distance contrastive loss
SENSITIVITY = 3   # For changing behavior confidence function. Higher values push conf. towards lower ED
MARGIN = 1.0      # Minimum Euclidean separation for negative PPIs
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

    if not os.path.exists(elut_file):
        print(f"File '{elut_file}' not found. Skipping ...")
        continue

    print(f"Parsing '{elut_file}' ...")
    elut_df = pd.read_csv(elut_file, sep='\t', index_col=0)

    #elut_df = elut_df.set_index('Unnamed: 0')
    elut_t10_df = elut_df[elut_df.sum(axis=1) >= 10]

    # Row-sum normalization (all values in elution trace divided by total PSMs in that trace)
    elut_t10_rsm_df = elut_t10_df.div(elut_t10_df.sum(axis=1), axis=0)

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

train_pos_ppis = trainingelut_pos_ppis[:(int)(len(trainingelut_pos_ppis)-(len(trainingelut_pos_ppis)/2))]
valid_pos_ppis = trainingelut_pos_ppis[(int)(len(trainingelut_pos_ppis)-(len(trainingelut_pos_ppis)/2)):]

train_neg_ppis = trainingelut_neg_ppis[:(int)(len(trainingelut_neg_ppis)-(len(trainingelut_neg_ppis)/2))]
valid_neg_ppis = trainingelut_neg_ppis[(int)(len(trainingelut_neg_ppis)-(len(trainingelut_neg_ppis)/2)):]


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

        numPos = 0
        numNeg = 0

        for elut_id, elut_df in enumerate(self.elut_df_list):
            # miles: Loading the elut dataframe as a set vastly reduces runtime of dataset initialization
            elut_df_index = set(elut_df.index)

            pos_ppis_elut = [pppi for pppi in pos_ppis if len(pppi.intersection(elut_df_index)) == 2]
            neg_ppis_elut = [nppi for nppi in neg_ppis if len(nppi.intersection(elut_df_index)) == 2]

            numPos += len(pos_ppis_elut)
            numNeg += len(neg_ppis_elut)

            if filterPearson:
                # Remove low-Pearson samples from positive PPIs
                pos_ppis_elut = [pppi for pppi in pos_ppis_elut if scipy.stats.pearsonr(
                    torch.from_numpy(elut_df.T[list(pppi)[0]].values.copy()).float(),
                    torch.from_numpy(elut_df.T[list(pppi)[1]].values.copy()).float())[0] >= 0.4]

                # Remove high-Pearson samples from negative PPIs
                neg_ppis_elut = [nppi for nppi in neg_ppis_elut if scipy.stats.pearsonr(
                    torch.from_numpy(elut_df.T[list(nppi)[0]].values.copy()).float(),
                    torch.from_numpy(elut_df.T[list(nppi)[1]].values.copy()).float())[0] <= 0.6]

            self.ppis = self.ppis + pos_ppis_elut + neg_ppis_elut
            self.labels = self.labels + [0]*len(pos_ppis_elut) + [1]*len(neg_ppis_elut)
            self.ppis_elut_ids = self.ppis_elut_ids + [elut_id]*len(pos_ppis_elut) + [elut_id]*len(neg_ppis_elut)

        print(f"Total number of pairs: {len(self.ppis)}")
        print(f"  Number of positive pairs: {numPos}")
        print(f"  Number of negative pairs: {numNeg}")

        self.transform = transform
        self.input_size = input_size

    def __getitem__(self, index):
        pair = list(self.ppis[index])
        elut_id = self.ppis_elut_ids[index]
        elut_df = self.elut_df_list[elut_id]

        prot0 = pair[0]
        prot1 = pair[1]

        elut0 = (prot0, torch.from_numpy(elut_df.T[pair[0]].values.copy()).float())
        elut1 = (prot1, torch.from_numpy(elut_df.T[pair[1]].values.copy()).float())

        if self.transform:
            elut0 = (prot0, nn.functional.pad(elut0[1], (self.input_size - elut0[1].size()[0], 0)))
            elut1 = (prot1, nn.functional.pad(elut1[1], (self.input_size - elut1[1].size()[0], 0)))

            if torch.isnan(elut0[1]).any():
                print("NaN found after padding input0")

            if torch.isnan(elut1[1]).any():
                print("NaN found after padding input1")

        elut0 = (elut0[0], elut0[1].unsqueeze(0))
        elut1 = (elut1[0], elut1[1].unsqueeze(0))

        return elut0, elut1, self.labels[index], elut_id

    def __len__(self):
        return len(self.ppis)

# =======================================================================================
#                            INSTANTIATE MODEL DATASETS
# =======================================================================================

# Instantiate the training dataset for the model
print("Assembling training set ...")
train_siamese_dataset = elutionPairDataset(elutdf_list=elut_list,
                                           pos_ppis = train_pos_ppis,
                                           neg_ppis = train_neg_ppis,
                                           transform=True)

# Instantiate the validation dataset for the model
print("Assembling validation set ...")
valid_siamese_dataset = elutionPairDataset(elutdf_list=elut_list,
                                           pos_ppis = valid_pos_ppis,
                                           neg_ppis = valid_neg_ppis,
                                           transform=True)

# Instantiate the test dataset for the model
print("Assembling test set ...")
test_siamese_dataset = elutionPairDataset(elutdf_list=elut_list,
                                          pos_ppis = test_pos_ppis,
                                          neg_ppis = test_neg_ppis,
                                          transform=True)

subset_indices = torch.randperm(len(test_siamese_dataset))[:SUBSET_SIZE]
subset_test_siamese_dataset = Subset(test_siamese_dataset, subset_indices)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


# Define the siamese neural network
class siameseNet(nn.Module):
    def __init__(self, projection_size=32, hidden_size=64, rnn_layers=1, lstm_layers=3):
        super(siameseNet, self).__init__()

        # Input: (B, 1, 128)
        self.pos_encoder = PositionalEncoding(d_model=hidden_size)

        # Transformer layers
        #   - Overfits easily, probably too complex
        self.tns = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size,
                                           nhead=1, dim_feedforward=32,
                                           batch_first=True),
                num_layers=1
        )

        self.rnn = nn.RNN(input_size=1,
                          hidden_size=hidden_size,
                          num_layers=rnn_layers,
                          batch_first=True)

        self.LSTM = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            bidirectional=True,
                            batch_first=True)
        
        self.input_projection = nn.Linear(1, projection_size)

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Sequential(

                nn.Linear(29, 16),
                nn.ReLU(inplace=True),

                nn.Linear(16, 8),
                nn.ReLU(inplace=True),

        )


    # Function called on both images, x, to determine their similarity
    def forward_once(self, x):
        # x shape: [batch, 1, 128] -> [batch, 128, 1]
        x = x.permute(0, 2, 1)

        # Apply input projection
        #x = self.input_projection(x)

        # Apply positional encoding
        #x = self.pos_encoder(x)

        # Apply transformer/rnn layer
        #x = self.tns(x)
        #x, _ = self.rnn(x)
        x, _ = self.LSTM(x)

        # Reshape
        x = x.permute(0, 2, 1) # [batch, features, seq_len]
        x = self.pooling(x).squeeze(-1) # [batch, features]

        #x, _ = x.max(dim=1)
        return x

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
valid_dataloader = DataLoader(valid_siamese_dataset,
                              shuffle=True,
                              drop_last=True,
                              num_workers=2,
                              batch_size=BATCH_SIZE)

if __FAST_TEST__:
    test_dataloader = DataLoader(subset_test_siamese_dataset,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=2,
                                 batch_size=BATCH_SIZE)
else:
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
optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
#optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

# Choose learning rate scheduler
# miles: ReduceLROnPlateau works great dynamically, StepLR good for exploring ruggedness
#        of loss topology
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Zero the gradients
optimizer.zero_grad()

counter = []
train_loss_hist = []
valid_loss_hist = []
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
        valid_loss = 0
        num_batches = len(train_dataloader)

        nb_train = len(train_dataloader)
        nb_valid = len(valid_dataloader)

        #print(f"Train: {nb_train}")
        #print(f"Valid: {nb_valid}")

        valid_dataloader_iter = iter(valid_dataloader)

        for i, (elut0_train, elut1_train, label_train, elut_id_train) in enumerate(train_dataloader, 0):

            try:
                (elut0_valid, elut1_valid, label_valid, elut_id_valid) = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                (elut0_valid, elut1_valid, label_valid, elut_id_valid) = next(valid_dataloader_iter)

            pccs_train = []
            pccs_valid = []

            # Obtain protein IDs
            prot0_train, prot1_train = elut0_train[0], elut1_train[0]
            prot0_valid, prot1_valid = elut0_valid[0], elut1_valid[1]

            # Obtain Pearson correlation between elut0, elut1
            for ppi0_train, ppi1_train in zip(elut0_train[1], elut1_train[1]):
                pccs_train.append(scipy.stats.pearsonr(ppi0_train[0], ppi1_train[0])[0])

            for ppi0_valid, ppi1_valid in zip(elut0_valid[1], elut1_valid[1]):
                pccs_valid.append(scipy.stats.pearsonr(ppi0_valid[0], ppi1_valid[0])[0])

            # Send elution data, labels to CUDA
            elut0_train, elut1_train, label_train = elut0_train[1].cuda(), elut1_train[1].cuda(), label_train.cuda()
            elut0_valid, elut1_valid, label_valid = elut0_valid[1].cuda(), elut1_valid[1].cuda(), label_valid.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass two elution traces into network and obtain two outputs
            output1_train, output2_train = net(elut0_train, elut1_train)
            net.eval()
            with torch.no_grad():
                output1_valid, output2_valid = net(elut0_valid, elut1_valid)
            net.train()
    
            # Pass outputs, label to the contrastive loss function
            train_loss_contrastive = criterion(output1_train, output2_train, label_train)
            valid_loss_contrastive = criterion(output1_valid, output2_valid, label_valid)

            # Perform backpropagation
            train_loss_contrastive.backward()

            # Perform gradient clipping
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            # Optimize
            optimizer.step()

            # Update training loss series for plotting
            if i % SAMPLE_RATE == 0:
                print(f"  Batch [{i} / {num_batches}]")
                print(f"      Training Loss: {train_loss_contrastive.item():.4f}")
                print(f"    Validation Loss: {valid_loss_contrastive.item():.4f}")
                iteration_num += SAMPLE_RATE

                counter.append(iteration_num)
                train_loss_hist.append(train_loss_contrastive.item())
                valid_loss_hist.append(valid_loss_contrastive.item())
  
        # Produce training loss curve figure PNG
        plot_loss(counter, [train_loss_hist, valid_loss_hist], ["Training loss", "Validation Loss"], title="Contrastive Loss",
                  xaxis="Batches", filename=f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_train_{epoch+1}.png")

        # Test model on test set
        net.eval()
        with torch.no_grad():
            test_loss = 0.0
            for test_i, (test_elut0, test_elut1, test_label, test_elut_id) in enumerate(test_dataloader, 0):
                # Obtain protein IDs
                test_prot0, test_prot1 = test_elut0[0], test_elut1[0]
                #for test_ppi0, test_ppi1 in zip(test_elut0[1], test_elut1[1]):
                #    pccListTest.append(scipy.stats.pearsonr(test_ppi0[0], test_ppi1[0])[0])

                #pcc = torch.tensor(pccListTest, dtype=torch.float32).cuda()

                # Send test elution data, labels to CUDA
                test_elut0, test_elut1, test_label = test_elut0[1].cuda(), test_elut1[1].cuda(), test_label.cuda()

                # Padd two elution traces into network and obtain two outputs
                test_output1, test_output2 = net(test_elut0, test_elut1)

                # Pass outputs, label to the contrastive loss function
                loss_test_contrastive = criterion(test_output1, test_output2, test_label)

                # Add to total test loss
                test_loss += loss_test_contrastive.item()

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

        # Summarize end of epoch metrics
        print(f"\nEnd of epoch summary")
        print(f"  Average testing loss: {avg_test_loss}")

        # Plot average test set loss vs epoch number 
        if epoch != 0:
            plot_loss(epoch_hist, [avg_test_loss_hist],
                      ["Testing loss"],
                      title="Average Contrastive Loss", xaxis="Epochs",
                      filename=f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_epoch_{epoch+1}.png")

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

if __FAST_TEST__:
    test_pos_dataloader = DataLoader(subset_test_pos_elution_pair_dataset,
                                     num_workers=1,
                                     batch_size=1,
                                     shuffle=True)
else:
    test_pos_dataloader = DataLoader(test_pos_elution_pair_dataset,
                                     num_workers=1,
                                     batch_size=1,
                                     shuffle=True)

def count_open_fds():
        return len(os.listdir(f'/proc/{os.getpid()}/fd'))

pos_ppi_scores_list = []
for i, (pos_elut0, pos_elut1, pos_label, elut_id) in enumerate(test_pos_dataloader):
    #print(f"Iteration {i}, Open FDs: {count_open_fds()}")

    # Obtain protein IDs
    prot0, prot1 = pos_elut0[0], pos_elut1[0]

    # Obtain elution traces
    pos_elut0, pos_elut1 = pos_elut0[1], pos_elut1[1]

    # Obtain Pearson correlation
    pearson_corr = scipy.stats.pearsonr(pos_elut0[0][0], pos_elut1[0][0])[0]

    # Run model on data
    output1, output2 = net(pos_elut0.cuda(), pos_elut1.cuda())

    # Get Euclidean distance b/t model outputs
    euclidean_dist = F.pairwise_distance(output1, output2)

    # Get confidence score of similarity based on Euclidean distance
    confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)

    pos_ppi_scores_list.append({"ID1": str(prot0[0]),
                                "ID2": str(prot1[0]),
                                "Experiment": elut_id.item(),
                                "Euclidean": euclidean_dist.item(),
                                "Pearson": pearson_corr,
                                "Confidence": confidence.item(),
                                "Label": pos_label.item()})

    if i % 5 == 0:
        print(f"Calculating Euclidean distances, Pearson scores for positive pairwise interactions ... {i * 100 / len(test_pos_dataloader):.2f} %", end='\r')
print("Calculating Euclidean distances, Pearson scores for positive pairwise interactions ... done            ")
pos_scores_df = pd.DataFrame(pos_ppi_scores_list)


# Run model and Pearson on negative PPIs only
test_neg_elution_pair_dataset = elutionPairDataset(elutdf_list=elut_list,
                                                   pos_ppis=[],
                                                   neg_ppis=test_neg_ppis,
                                                   transform=True)
subset_indices = torch.randperm(len(test_neg_elution_pair_dataset))[:SUBSET_SIZE]
subset_test_neg_elution_pair_dataset = Subset(test_neg_elution_pair_dataset, subset_indices)

if __FAST_TEST__:
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
neg_ppi_scores_list = []
for i, (neg_elut0, neg_elut1, neg_label, elut_id) in enumerate(test_neg_dataloader):
    #print(f"Iteration {i}, Open FDs: {count_open_fds()}")

    # Obtain protein IDs
    prot0, prot1 = neg_elut0[0], neg_elut1[0]

    # Obtain elution traces
    neg_elut0, neg_elut1 = neg_elut0[1], neg_elut1[1]

    # Obtain Pearson correlation
    #neg_ppi_pearson_list.append(scipy.stats.pearsonr(neg_elut0[0][0], neg_elut1[0][0])[0])
    pearson_corr = scipy.stats.pearsonr(neg_elut0[0][0], neg_elut1[0][0])[0]

    # Run model on data
    output1, output2 = net(neg_elut0.cuda(), neg_elut1.cuda())

    # Get Euclidean distance b/t model outputs
    euclidean_dist = F.pairwise_distance(output1, output2)

    # Get confidence score of similarity based on Euclidean distance
    confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)
    #neg_ppi_conf_output_file.write(str(prot0[0]) + ":" + str(prot1[0]) + f"\t{euclidean_dist.item():.4f}\t{neg_ppi_pearson_list[i]:.4f}\t{confidence.item():.4f}\t{elut_id}\n")

    neg_ppi_scores_list.append({"ID1": str(prot0[0]),
                                "ID2": str(prot1[0]),
                                "Experiment": elut_id.item(),
                                "Euclidean": euclidean_dist.item(),
                                "Pearson": pearson_corr,
                                "Confidence": confidence.item(),
                                "Label": neg_label.item()})


    if i % 5 == 0:
        print(f"Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... {i * 100 / len(test_neg_dataloader):.2f} %", end='\r')
print("Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... done            ")
neg_scores_df = pd.DataFrame(neg_ppi_scores_list)


# Plot the PDF of the euclidean distance between positive and negative protein pairs
x = np.linspace(-5,5,1000)
mean_pos = np.mean(pos_scores_df[['Euclidean']].values)
mean_neg = np.mean(neg_scores_df[['Euclidean']].values)
diff_means_pos_neg_pdf = abs(mean_neg - mean_pos)
y = norm.pdf(x, loc=mean_neg, scale=np.std(neg_scores_df[['Euclidean']].values))
y2 = norm.pdf(x, loc=mean_pos, scale=np.std(pos_scores_df[['Euclidean']].values))

pylab.plot(x,y,label="negative_pairs")
pylab.plot(x,y2,label="positive_pairs")
pylab.annotate(f"{mean_pos}", (0.0, mean_pos))
pylab.title("PDFs of Euclidean distances after model transform")
pylab.legend()
pylab.grid()
pylab.savefig(f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_pos_neg_pairs_EUCLIDEAN.png")
pylab.clf()
pylab.cla()


sns.histplot(neg_scores_df[['Euclidean']].values, alpha=0.5,
             label='negative pairs')
sns.histplot(pos_scores_df[['Euclidean']].values, alpha=0.5, bins=25,
             label='positive pairs', color='orange')
plt.title("Euclidean distance counts")
plt.legend()
plt.grid()
plt.savefig(f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_fig_hist_EUCLIDEAN.png")
pylab.clf()
pylab.cla()

# Plot the PDFs of the Pearson scores
x = np.linspace(-2, 2, 1000)
y = norm.pdf(x, loc=np.mean(neg_scores_df[['Pearson']].values), scale=np.std(neg_scores_df[['Pearson']].values))
y2 = norm.pdf(x, loc=np.mean(pos_scores_df[['Pearson']].values), scale=np.std(pos_scores_df[['Pearson']].values))

plt.plot(x,y,label="pearson_negative_pairs")
plt.plot(x,y2,label="pearson_positive_pairs")
plt.title("PDFs of Pearson scores")
plt.legend()
plt.grid()
plt.savefig(f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_pos_neg_pairs_PEARSON.png")
plt.clf()
plt.cla()

# Plot the counts of positive/negative Pearson scores
sns.histplot(neg_scores_df[['Pearson']].values, alpha=0.5,
             label='negative pairs')
sns.histplot(pos_scores_df[['Pearson']].values, alpha=0.5, bins=50,
             label='positive pairs', color='orange')
plt.title("Pearson score counts")
plt.legend()
plt.grid()
plt.savefig(f"{PARAMETER_FILENAME.split(".")[0]}_fig_hist_PEARSON.png")
pylab.clf()
pylab.cla()

# Plot the euclidean distance against the Pearson score
scores_df = pd.DataFrame(pos_ppi_scores_list + neg_ppi_scores_list)

scores_df.to_csv(f"{PARAMETER_FILENAME.split(".")[0]}_test_scores.csv")
# Plot the pearson scores for positive PPIs against the euclidean score
#   Euclidean distance should be small, preferably as close to 0 as possible
scat1 = sns.scatterplot(data=scores_df, x="Euclidean", y="Pearson",
                        hue="Label", s=10)
scat1.set_xlim(-1, 12)
fig_scat1 = scat1.get_figure()
fig_scat1.savefig(f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_pearson_vs_euc_scatter.png")
fig_scat1.clf()

# Plot a contour graph of the positive PPIs
#   We want to make the glob near (0, 0) larger
kde1 = sns.kdeplot(data=scores_df, x="Euclidean", y="Pearson",
                   common_norm=False, hue="Label", levels=15)
kde1.set_xlim(-1, 12)
fig_kde1 = kde1.get_figure()
fig_kde1.savefig(f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_pearson_vs_euc_kde.png")
fig_kde1.clf()

# Get labels, confidences, Euclidean distances, Pearson coefficients of test points
y_labels = pos_scores_df['Label'].tolist() + neg_scores_df['Label'].tolist()
y_confs = pos_scores_df['Confidence'].tolist() + neg_scores_df['Confidence'].tolist()
y_dists = pos_scores_df['Euclidean'].tolist() + neg_scores_df['Euclidean'].tolist()
y_pearson = pos_scores_df['Pearson'].tolist() + neg_scores_df['Pearson'].tolist()


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
plt.plot(recall_euclidean, precision_euclidean, color='blue')
plt.plot(recall_pearson, precision_pearson, color='orange')
plt.title("PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(["Euclidean PR", "Pearson PR"])
plt.savefig(f"{FIGURES_DIRECTORY}/{PARAMETER_FILENAME.split(".")[0]}_pr_curve.png")
plt.clf()
plt.cla()

# Obtain area under PR curves
aupr_euclidean = get_aupr(precision_euclidean, recall_euclidean)
aupr_pearson = get_aupr(precision_pearson, recall_pearson)
# Print metrics to txt file
with open(f"{PARAMETER_FILENAME.split(".")[0]}_results-final.log", 'w') as outFile:
    outFile.write(f"Minimum Average Validation Loss: {min_avg_valid_loss:.4f}\n")
    outFile.write(f"Minimum Average Test Loss: {min_avg_test_loss:.4f}\n")
    outFile.write(f"Difference Between ED Means of Pos/Neg PPIs: {diff_means_pos_neg_pdf:.4f}\n")
    outFile.write(f"Area Under Euclidean PR Curve: {aupr_euclidean:.4f}\n")
    outFile.write(f"Area Under Pearson PR Curve: {aupr_pearson:.4f}\n")

for k, elut in enumerate(elut_list):

    elut_data_filename = elut_data[k].split(".")[0]
    curr_figs_directory = f"{FIGURES_DIRECTORY}/{elut_data_filename.split(".")[0]}"

    if not os.path.exists(curr_figs_directory):
        os.makedirs(curr_figs_directory)

    # Run model and Pearson on positive PPIs only
    test_pos_elution_pair_dataset = elutionPairDataset(elutdf_list=[elut],
                                                       pos_ppis=test_pos_ppis,
                                                       neg_ppis=[],
                                                       transform=True)

    subset_indices = torch.randperm(len(test_pos_elution_pair_dataset))[:SUBSET_SIZE]
    subset_test_pos_elution_pair_dataset = Subset(test_pos_elution_pair_dataset, subset_indices)

    if __FAST_TEST__:
        test_pos_dataloader = DataLoader(subset_test_pos_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)
    else:
        test_pos_dataloader = DataLoader(test_pos_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)

    pos_ppi_scores_list = []
    if len(test_pos_elution_pair_dataset) > 0:
        for i, (pos_elut0, pos_elut1, pos_label, elut_id) in enumerate(test_pos_dataloader):

            # Obtain protein IDs
            prot0, prot1 = pos_elut0[0], pos_elut1[0]

            # Obtain elution traces
            pos_elut0, pos_elut1 = pos_elut0[1], pos_elut1[1]

            # Obtain Pearson correlation
            pearson_corr = scipy.stats.pearsonr(pos_elut0[0][0], pos_elut1[0][0])[0]

            # Run model on data
            output1, output2 = net(pos_elut0.cuda(), pos_elut1.cuda())

            # Get Euclidean distance b/t model outputs
            euclidean_dist = F.pairwise_distance(output1, output2)

            # Get confidence score of similarity based on Euclidean distance
            confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)

            pos_ppi_scores_list.append({"ID1": str(prot0[0]),
                                        "ID2": str(prot1[0]),
                                        "Experiment": elut_id.item(),
                                        "Euclidean": euclidean_dist.item(),
                                        "Pearson": pearson_corr,
                                        "Confidence": confidence.item(),
                                        "Label": pos_label.item()})


        
            if i % 5 == 0:
                print(f"Calculating Euclidean distances, Pearson scores for positive pairwise interactions ... {i * 100 / len(test_pos_dataloader):.2f} %", end='\r')
    else:
        print("Dataset is empty!")
    pos_scores_df = pd.DataFrame(pos_ppi_scores_list)

    # Run model and Pearson on negative PPIs only
    test_neg_elution_pair_dataset = elutionPairDataset(elutdf_list=[elut],
                                                       pos_ppis=[],
                                                       neg_ppis=test_neg_ppis,
                                                       transform=True)
    subset_indices = torch.randperm(len(test_neg_elution_pair_dataset))[:SUBSET_SIZE]
    subset_test_neg_elution_pair_dataset = Subset(test_neg_elution_pair_dataset, subset_indices)

    if __FAST_TEST__:
        test_neg_dataloader = DataLoader(subset_test_neg_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)
    else:
        test_neg_dataloader = DataLoader(test_neg_elution_pair_dataset,
                                         num_workers=1,
                                         batch_size=1)

    # Iterate over negative samples, getting Pearson scores and post-transform Euclidean distance for each PPI
    neg_ppi_scores_list = []
    for i, (neg_elut0, neg_elut1, neg_label, elut_id) in enumerate(test_neg_dataloader):

        # Obtain protein IDs
        prot0, prot1 = neg_elut0[0], neg_elut1[0]

        # Obtain elution traces
        neg_elut0, neg_elut1 = neg_elut0[1], neg_elut1[1]

        # Obtain Pearson correlation
        pearson_corr = scipy.stats.pearsonr(neg_elut0[0][0], neg_elut1[0][0])[0]

        # Run model on data
        output1, output2 = net(neg_elut0.cuda(), neg_elut1.cuda())

        # Get Euclidean distance b/t model outputs
        euclidean_dist = F.pairwise_distance(output1, output2)

        # Get confidence score of similarity based on Euclidean distance
        confidence = euclidean_to_confidence(euclidean_dist, MAX_ED, SENSITIVITY)
        #neg_ppi_conf_output_file.write(str(prot0[0]) + ":" + str(prot1[0]) + f"\t{euclidean_dist.item():.4f}\t{neg_ppi_pearson_list[i]:.4f}\t{confidence.item():.4f}\t{elut_id}\n")

        neg_ppi_scores_list.append({"ID1": str(prot0[0]),
                                    "ID2": str(prot1[0]),
                                    "Experiment": elut_id.item(),
                                    "Euclidean": euclidean_dist.item(),
                                    "Pearson": pearson_corr,
                                    "Confidence": confidence.item(),
                                    "Label": neg_label.item()})


        if i % 5 == 0:
            print(f"Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... {i * 100 / len(test_neg_dataloader):.2f} %", end='\r')
    neg_scores_df = pd.DataFrame(neg_ppi_scores_list)
    print("Calculating Euclidean distances, Pearson scores for negative pairwise interactions ... done            ")

    # Plot the PDF of the euclidean distance between positive and negative protein pairs
    x = np.linspace(-5,5,1000)
    mean_pos = np.mean(pos_scores_df['Euclidean'].tolist())
    mean_neg = np.mean(neg_scores_df['Euclidean'].tolist())
    diff_means_pos_neg_pdf = abs(mean_neg - mean_pos)
    y = norm.pdf(x, loc=mean_neg, scale=np.std(neg_scores_df['Euclidean'].tolist()))
    y2 = norm.pdf(x, loc=mean_pos, scale=np.std(pos_scores_df['Euclidean'].tolist()))

    pylab.plot(x,y,label="negative_pairs")
    pylab.plot(x,y2,label="positive_pairs")
    pylab.annotate(f"{mean_pos}", (0.0, mean_pos))
    pylab.title("PDFs of Euclidean distances after model transform")
    pylab.legend()
    pylab.grid()
    pylab.savefig(f"{curr_figs_directory}/{PARAMETER_FILENAME.split(".")[0]}_pos_neg_pairs_EUCLIDEAN_{elut_data_filename}.png")
    pylab.clf()
    pylab.cla()


    sns.histplot(neg_scores_df['Euclidean'].tolist(), alpha=0.5,
                 label='negative pairs')
    sns.histplot(pos_scores_df['Euclidean'].tolist(), alpha=0.5, bins=25,
                 label='positive pairs', color='orange')
    plt.title("Euclidean distance counts")
    plt.legend()
    plt.grid()
    plt.savefig(f"{curr_figs_directory}/f{PARAMETER_FILENAME.split(".")[0]}_ig_hist_EUCLIDEAN_{elut_data_filename}.png")
    pylab.clf()
    pylab.cla()

    # Plot the PDFs of the Pearson scores
    x = np.linspace(-2, 2, 1000)
    y = norm.pdf(x, loc=np.mean(neg_scores_df['Pearson'].tolist()), scale=np.std(neg_scores_df['Pearson'].tolist()))
    y2 = norm.pdf(x, loc=np.mean(pos_scores_df['Pearson'].tolist()), scale=np.std(neg_scores_df['Pearson'].tolist()))
    plt.plot(x,y,label="pearson_negative_pairs")
    plt.plot(x,y2,label="pearson_positive_pairs")
    plt.title("PDFs of Pearson scores")
    plt.legend()
    plt.grid()
    plt.savefig(f"{curr_figs_directory}/{PARAMETER_FILENAME.split(".")[0]}_pos_neg_pairs_PEARSON_{elut_data_filename}.png")
    plt.clf()
    plt.cla()

    # Plot the counts of positive/negative Pearson scores
    sns.histplot(neg_scores_df['Pearson'].tolist(), alpha=0.5,
                 label='negative pairs')
    sns.histplot(pos_scores_df['Pearson'].tolist(), alpha=0.5, bins=50,
                 label='positive pairs', color='orange')
    plt.title(f"Pearson score counts")
    plt.legend()
    plt.grid()
    plt.savefig(f"{curr_figs_directory}/{PARAMETER_FILENAME.split(".")[0]}_fig_hist_PEARSON_{elut_data_filename}.png")
    pylab.clf()
    pylab.cla()

    # Plot the euclidean distance against the Pearson score
    scores_df = pd.DataFrame(pos_ppi_scores_list + neg_ppi_scores_list)

    #scores_df.to_csv(f"{PARAMETER_FILENAME.split(".")[0]}_test_scores.csv")


    # Plot the pearson scores for positive PPIs against the euclidean score
    #   Euclidean distance should be small, preferably as close to 0 as possible
    scat1 = sns.scatterplot(data=scores_df, x="Euclidean", y="Pearson",
                            hue="Label", s=10)
    scat1.set_xlim(-1, 12)
    fig_scat1 = scat1.get_figure()
    fig_scat1.savefig(f"{curr_figs_directory}/{PARAMETER_FILENAME.split(".")[0]}_pearson_ed_pos_scatter_{elut_data_filename}.png")
    fig_scat1.clf()

    # Plot a contour graph of the positive PPIs
    #   We want to make the glob near (0, 0) larger
    kde1 = sns.kdeplot(data=scores_df, x='Euclidean', y='Pearson',
                       common_norm=False, hue='Label', levels=15)
    kde1.set_xlim(-1, 12)
    fig_kde1 = kde1.get_figure()
    fig_kde1.savefig(f"{curr_figs_directory}/{PARAMETER_FILENAME.split(".")[0]}_pearson_ed_contour_{elut_data_filename}.png")
    fig_kde1.clf()

    # Get labels, confidences, Euclidean distances, Pearson coefficients of test points
    y_labels = pos_scores_df['Label'].tolist() + neg_scores_df['Label'].tolist()
    y_confs = pos_scores_df['Confidence'].tolist() + neg_scores_df['Confidence'].tolist()
    y_dists = pos_scores_df['Euclidean'].tolist() + neg_scores_df['Euclidean'].tolist()
    y_pearson = pos_scores_df['Pearson'].tolist() + neg_scores_df['Pearson'].tolist()


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
    plt.savefig(f"{curr_figs_directory}/{PARAMETER_FILENAME.split(".")[0]}_pr_curve_{elut_data_filename}.png")
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
