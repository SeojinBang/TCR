from __future__ import print_function
import os
import sys
import csv
import time
sys.path.append('../')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import define_dataloader,load_embedding
from utils import str2bool,check_model_name,timeSince,get_performance_batchiter,print_performance,write_blackbox_output_batchiter
import data_io_tf
from pathlib import Path

PRINT_EVERY_EPOCH = 1
SIZE_HIDDEN1_CNN = 32
SIZE_HIDDEN2_CNN = 16
SIZE_KERNEL1 = 3
SIZE_KERNEL2 = 3

class Net(nn.Module):
    def __init__(self, embedding, pep_length, tcr_length):
        
        super(Net, self).__init__()

        ## embedding layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino,self.embedding_dim,padding_idx=self.num_amino-1).from_pretrained(torch.FloatTensor(embedding),freeze = False)

        ## peptide encoding layer
        self.size_hidden1_cnn = SIZE_HIDDEN1_CNN
        self.size_hidden2_cnn = SIZE_HIDDEN2_CNN
        self.size_kernel1 = SIZE_KERNEL1
        self.size_kernel2 = SIZE_KERNEL2
        self.size_padding = (self.size_kernel1-1)/2
        self.encode_pep = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(self.embedding_dim, self.size_hidden1_cnn, kernel_size=self.size_kernel1),
            nn.BatchNorm1d(self.size_hidden1_cnn),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel1, stride=1, padding=self.size_padding),
            nn.Conv1d(self.size_hidden1_cnn, self.size_hidden2_cnn, kernel_size=self.size_kernel2),
            nn.BatchNorm1d(self.size_hidden2_cnn),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel2)
            )
        
        ## trc encoding layer
        self.encode_tcr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(self.embedding_dim, self.size_hidden1_cnn, kernel_size=self.size_kernel1),
            nn.BatchNorm1d(self.size_hidden1_cnn),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel1, stride=1, padding=self.size_padding),
            nn.Conv1d(self.size_hidden1_cnn, self.size_hidden2_cnn, kernel_size=self.size_kernel2),
            nn.BatchNorm1d(self.size_hidden2_cnn),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel2)
            )

        ## dense layer at the end
        self.net_pep_dim = self.size_hidden2_cnn * ((pep_length-self.size_kernel1+1-self.size_kernel2+1)/self.size_kernel2)
        self.net_tcr_dim = self.size_hidden2_cnn * ((tcr_length-self.size_kernel1+1-self.size_kernel2+1)/self.size_kernel2)
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.net_pep_dim+self.net_tcr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.LogSoftmax(1)
            )
        
    def forward(self, pep, tcr):

        pep = self.embedding(pep)
        tcr = self.embedding(tcr)
        pep = self.encode_pep(pep.transpose(1,2))
        tcr = self.encode_tcr(tcr.transpose(1,2))
        peptcr = torch.cat((pep, tcr), -1)#50, 8, 2
        peptcr = peptcr.view(-1, 1, peptcr.size(-1) * peptcr.size(-2)).squeeze(-2)
        peptcr = self.net(peptcr)
        
        return peptcr

def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()     

    for batch in train_loader:

        X_pep, X_tcr, y = batch.X_pep.to(device), batch.X_tcr.to(device), batch.y.to(device)
        optimizer.zero_grad()
        yhat = model(X_pep, X_tcr)
        loss = F.cross_entropy(yhat, y)
        loss.backward()
        optimizer.step()

    if epoch % PRINT_EVERY_EPOCH == 1:
        print('[TRAIN] Epoch {} Loss {:.4f}'.format(epoch, loss.item()))

def main():

    parser = argparse.ArgumentParser(description='Prediction of TCR binding to peptide-MHC complexes')

    parser.add_argument('--infile', type=str, help='input file')
    parser.add_argument('--indepfile', type=str, default=None, help='independent test file')
    parser.add_argument('--indepfile2', type=str, default=None)
    parser.add_argument('--blosum', type=str, default='data/BLOSUM50', help='file with BLOSUM matrix')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N', help='batch size')
    parser.add_argument('--model_name', type=str, default='original.ckpt', help = 'if train is True, model name to be saved, otherwise model name to be loaded')
    parser.add_argument('--epoch', type = int, default=200, metavar='N', help='number of epoch to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--cuda', type = str2bool, default=True, help = 'enable cuda')
    parser.add_argument('--seed', type=int, default=7405, help='random seed')
    parser.add_argument('--mode', default = 'train', type=str, help = 'train or test')
    
    args = parser.parse_args()

    ## cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    args.cuda = (args.cuda and torch.cuda.is_available()) 
    device = torch.device('cuda' if args.cuda else 'cpu')

    ## set random seed
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed) if args.cuda else None
      
    ## read data
    X_pep, X_tcr, y = data_io_tf.read_pTCR(args.infile)
    y = np.array(y)

    ## read indep data
    if args.indepfile is not None:
        X_indep_pep, X_indep_tcr, y_indep = data_io_tf.read_pTCR(args.indepfile)
        y_indep = np.array(y_indep)

    if args.indepfile2 is not None:
        X_indep2_pep, X_indep2_tcr, y_indep2 = data_io_tf.read_pTCR(args.indepfile2)
        y_indep2 = np.array(y_indep2)

    # embedding matrix
    embedding = load_embedding(args.blosum)
    
    ## split data
    n_total = len(y)
    n_train = int(round(n_total * 0.8))
    n_valid = int(round(n_total * 0.1))
    n_test = n_total - n_train - n_valid
    idx_shuffled = np.arange(n_total); np.random.shuffle(idx_shuffled)
    idx_train, idx_valid, idx_test = idx_shuffled[:n_train], idx_shuffled[n_train:(n_train+n_valid)], idx_shuffled[(n_train+n_valid):]

    ## define dataloader
    train_loader = define_dataloader(X_pep[idx_train], X_tcr[idx_train], y[idx_train], None,
                                     None, None,
                                     batch_size=args.batch_size, device=device)
    valid_loader = define_dataloader(X_pep[idx_valid], X_tcr[idx_valid], y[idx_valid], None,
                                maxlen_pep=train_loader['pep_length'], maxlen_tcr=train_loader['tcr_length'],
                                batch_size=args.batch_size, device=device)
    test_loader = define_dataloader(X_pep[idx_test], X_tcr[idx_test], y[idx_test], None,
                                maxlen_pep=train_loader['pep_length'], maxlen_tcr=train_loader['tcr_length'],
                                batch_size=args.batch_size, device=device)
    if args.indepfile is not None:
        indep_loader = define_dataloader(X_indep_pep, X_indep_tcr, y_indep, None,
                                maxlen_pep=train_loader['pep_length'], maxlen_tcr=train_loader['tcr_length'],
                                batch_size=args.batch_size, device=device)
    if args.indepfile2 is not None:
        indep_loader2 = define_dataloader(X_indep2_pep, X_indep2_tcr, y_indep2, None,
                                maxlen_pep=train_loader['pep_length'], maxlen_tcr=train_loader['tcr_length'],
                                batch_size=args.batch_size, device=device)
                                     
    ## define model
    model = Net(embedding, train_loader['pep_length'], train_loader['tcr_length']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if 'models' not in os.listdir('.'):
        os.mkdir('models')
    if 'result' not in os.listdir('.'):
        os.mkdir('result')

    ## fit model        
    if args.mode == 'train' : 
            
        model_name = check_model_name(args.model_name)
        model_name = check_model_name(model_name, './models')
        model_name = args.model_name

        wf_open = open('result/'+os.path.splitext(os.path.basename(args.infile))[0]+'_'+os.path.splitext(os.path.basename(args.model_name))[0]+'_valid_K3.csv', 'w')
        wf_colnames = ['loss', 'accuracy', 'precision1', 'precision0', 'recall1', 'recall0',
                       'f1macro','f1micro', 'auc']
        wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')

        t0 = time.time()
        for epoch in range(1, args.epoch + 1):
            
            train(args, model, device, train_loader['loader'], optimizer, epoch)

            ## evaluate performance
            perf_train = get_performance_batchiter(train_loader['loader'], model, device)
            perf_valid = get_performance_batchiter(valid_loader['loader'], model, device)

            ## print performance
            print('Epoch {} TimeSince {}\n'.format(epoch, timeSince(t0)))
            print('[TRAIN] {} ----------------'.format(epoch))
            print_performance(perf_train)
            print('[VALID] {} ----------------'.format(epoch))
            print_performance(perf_valid, writeif=True, wf=wf)

        ## evaluate and print test-set performance 
        print('[TEST ] {} ----------------'.format(epoch))
        perf_test = get_performance_batchiter(test_loader['loader'], model, device)
        print_performance(perf_test)

        ## evaluate and print independent-test-set performance
        if args.indepfile is not None:
            print('[INDEP] {} ----------------'.format(epoch)) 
            perf_indep = get_performance_batchiter(indep_loader['loader'], model, device)
            print_performance(perf_indep)

            ## write blackbox output
            wf_bb_open = open('data/blackboxpred_' + os.path.basename(args.indepfile), 'w')
            wf_bb = csv.writer(wf_bb_open, delimiter='\t')
            #wf_bb = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')
            write_blackbox_output_batchiter(indep_loader, model, wf_bb, device)

            wf_bb_open1 = open('data/blackboxpredscore_' + os.path.basename(args.indepfile), 'w')
            wf_bb1 = csv.writer(wf_bb_open1, delimiter='\t')
            write_blackbox_output_batchiter(indep_loader, model, wf_bb1, device, ifscore=True)

        if args.indepfile2 is not None:
            print('[INDEP2] {} ----------------'.format(epoch)) 
            perf_indep2 = get_performance_batchiter(indep_loader2['loader'], model, device)
            print_performance(perf_indep2)

            ## write blackbox output
            wf_bb_open2 = open('data/blackboxpred_' + os.path.basename(args.indepfile2), 'w')
            wf_bb2 = csv.writer(wf_bb_open2, delimiter='\t')
            write_blackbox_output_batchiter(indep_loader2, model, wf_bb2, device)

            wf_bb_open3 = open('data/blackboxpredscore_' + os.path.basename(args.indepfile2), 'w')
            wf_bb3 = csv.writer(wf_bb_open3, delimiter='\t')
            write_blackbox_output_batchiter(indep_loader2, model, wf_bb3, device, ifscore=True)

        model_name = './models' + model_name
        torch.save(model.state_dict(), model_name)
            
    elif args.mode == 'test' : 
        
        model_name = args.model_name

        assert model_name in os.listdir('./models')
        
        model_name = './models' + model_name
        model.load_state_dict(torch.load(model_name))
        
        test(args, model, device, test_loader, outfile = True, outmode = 'test') # test accuracy
        
    else :
        
        print('\nError: "--mode train" or "--mode test" expected')
        
if __name__ == '__main__':
    main()  


