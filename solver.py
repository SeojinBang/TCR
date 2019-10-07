from __future__ import print_function
import os
import sys
import csv
import time
import math
sys.path.append('../')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from data_loader import define_dataloader,load_embedding
from utils import str2bool,check_model_name,timeSince,get_performance_fidelity_batchiter,print_performance,idxtobool,cuda,write_explain_batchiter
import data_io_tf
from pathlib import Path

PRINT_EVERY_EPOCH = 100000
SIZE_HIDDEN1_CNN = 32
SIZE_HIDDEN2_CNN = 16
SIZE_KERNEL1 = 3
SIZE_KERNEL2 = 3

class VIBI(nn.Module):
    def __init__(self, embedding, pep_length, tcr_length, **kwargs):
        
        super(VIBI, self).__init__()

        ## parameter
        self.tau = 0.1
        self.K = kwargs['K']
        self.chunk_size = kwargs['chunk_size']
        self.num_sample = kwargs['num_sample']
        self.cuda = kwargs['cuda']
        self.pep_length = pep_length
        self.tcr_length = tcr_length
        
        ## embedding layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino,
                                      self.embedding_dim,padding_idx = self.num_amino-1).\
                                      from_pretrained(torch.FloatTensor(embedding),freeze = False)

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

        ## dense layer in the approximator
        self.net_pep_dim = self.size_hidden2_cnn * ((pep_length-self.size_kernel1+1-self.size_kernel2+1)/self.size_kernel2)
        self.net_tcr_dim = self.size_hidden2_cnn * ((tcr_length-self.size_kernel1+1-self.size_kernel2+1)/self.size_kernel2)
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.net_pep_dim+self.net_tcr_dim, 32),
            nn.ReLU(),
            #nn.Linear(32, 16),
            #nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(1)
            )

        ## layers in the explainer
        self.explainer_pep = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.net_pep_dim+self.net_tcr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, pep_length),
            nn.LogSoftmax(-1)
            )
        self.explainer_tcr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.net_pep_dim+self.net_tcr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, tcr_length),
            nn.LogSoftmax(-1)
            )
        
    def forward(self, pep, tcr, num_sample=1):

        assert num_sample > 0

        # Explainer
        p_pep, p_tcr = self.explainer(pep, tcr)

        # Information Bottleneck
        z_pep, z_tcr, z_fixed_pep, z_fixed_tcr = self.reparameterize(p_pep, p_tcr, self.tau, self.K, num_sample)
        pep = self.embedding(pep).transpose(1,2)
        tcr = self.embedding(tcr).transpose(1,2)
        size_pep = torch.Size([pep.size(0), num_sample, pep.size(1), pep.size(2)])
        size_tcr = torch.Size([tcr.size(0), num_sample, tcr.size(1), tcr.size(2)])
        t_pep = torch.mul(pep.unsqueeze(1).expand(size_pep), z_pep.unsqueeze(-2).expand(size_pep))
        t_tcr = torch.mul(tcr.unsqueeze(1).expand(size_tcr), z_tcr.unsqueeze(-2).expand(size_tcr))
        t_fixed_pep = torch.mul(pep, z_fixed_pep.unsqueeze(1).expand_as(pep))
        t_fixed_tcr = torch.mul(tcr, z_fixed_tcr.unsqueeze(1).expand_as(tcr))
        
        # Approximator
        score = self.approximator(t_pep.view(-1, self.embedding_dim, self.pep_length),
                                  t_tcr.view(-1, self.embedding_dim, self.tcr_length), num_sample)
        score_fixed = self.approximator(t_fixed_pep, t_fixed_tcr)
        
        return score, p_pep, p_tcr, z_pep, z_tcr, score_fixed

    def explainer(self, pep, tcr):

        pep = self.embedding(pep)
        tcr = self.embedding(tcr)
        pep = self.encode_pep(pep.transpose(1,2))
        tcr = self.encode_tcr(tcr.transpose(1,2))
        peptcr = torch.cat((pep, tcr), -1)
        peptcr = peptcr.view(-1, 1, peptcr.size(-1) * peptcr.size(-2)).squeeze(-2)
        p_pep = self.explainer_pep(peptcr)
        p_tcr = self.explainer_tcr(peptcr)
        
        return p_pep, p_tcr

    def approximator(self, pep, tcr, num_sample=1):
            
        pep = self.encode_pep(pep)
        tcr = self.encode_tcr(tcr)
        peptcr = torch.cat((pep, tcr), -1)#50, 8, 2
        peptcr = peptcr.view(-1, 1, peptcr.size(-1) * peptcr.size(-2)).squeeze(-2)
        peptcr = self.net(peptcr)

        if num_sample > 1:
            peptcr = peptcr.view(-1, num_sample, peptcr.size(-1)).mean(1)
        
        return peptcr

    def reparameterize(self, p_pep, p_tcr, tau, k, num_sample):

        # sampling
        batch_size = p_pep.size(0)
        len_pep = p_pep.size(1) # batch_size * len_pep
        len_tcr = p_tcr.size(1) # batch_size * len_tcr
        p_pep_ = p_pep.view(batch_size, 1, 1, len_pep).expand(batch_size, num_sample, k, len_pep)
        p_tcr_ = p_tcr.view(batch_size, 1, 1, len_tcr).expand(batch_size, num_sample, k, len_tcr)
        C_pep = RelaxedOneHotCategorical(tau, p_pep_)
        C_tcr = RelaxedOneHotCategorical(tau, p_tcr_)
        Z_pep, _ = torch.max(C_pep.sample(), -2) # batch_size, num_sample, len_pep
        Z_tcr, _ = torch.max(C_tcr.sample(), -2) # batch_size, num_sample, len_tcr
        
        # without sampling
        _, Z_fixed_pep = p_pep.topk(k, dim = -1) # batch_size, k
        _, Z_fixed_tcr = p_tcr.topk(k, dim = -1) # batch_size, k
        size_pep = p_pep.size()
        size_tcr = p_tcr.size()
        Z_fixed_pep = idxtobool(Z_fixed_pep, size_pep, self.cuda)
        Z_fixed_tcr = idxtobool(Z_fixed_tcr, size_tcr, self.cuda)

        return Z_pep, Z_tcr, Z_fixed_pep, Z_fixed_tcr

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

def prior(var_size):

    p = torch.ones(var_size[1])/var_size[1]
    p = p.view(1, var_size[1])
    p_prior = p.expand(var_size)

    return p_prior

def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()

def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()
    class_criterion = nn.CrossEntropyLoss(reduction = 'sum')
    info_criterion = nn.KLDivLoss(reduction = 'sum')

    for batch in train_loader:

        X_pep, X_tcr, y = batch.X_pep.to(device), batch.X_tcr.to(device), batch.y.to(device)
        optimizer.zero_grad()
        score, p_pep, p_tcr, z_pep, z_tcr, score_fixed = model(X_pep, X_tcr, args.num_sample)

        # prior distribution
        p_pep_prior = cuda(prior(var_size=p_pep.size()), args.cuda)
        p_tcr_prior = cuda(prior(var_size=p_tcr.size()), args.cuda)

        # define loss
        class_loss = class_criterion(score, y).div(math.log(2)) / args.batch_size
        #info_loss = args.K * (info_criterion(p_pep, p_pep_prior)+info_criterion(p_tcr, p_tcr_prior)) / args.batch_size
        info_loss = (info_criterion(p_pep, p_pep_prior)+info_criterion(p_tcr, p_tcr_prior)) / args.batch_size
        total_loss = class_loss + args.beta* info_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    #print("peptide_prob")
    #print(torch.exp(p_pep)[:3])

    if epoch % PRINT_EVERY_EPOCH == 0:
        print('[TRAIN] Epoch {} Total Loss {:.4f} = {:.4f} + beta {:.4f}'.format(epoch,
                                                                                 total_loss.item(),
                                                                                 class_loss.item(),
                                                                                 info_loss.item()))

def main():

    parser = argparse.ArgumentParser(description='Prediction of TCR binding to peptide-MHC complexes')

    parser.add_argument('--K', type=int, default=10,
                        help='number of cognitive chunk')
    parser.add_argument('--chunk_size', type=int, default=1,
                        help='chunk size')
    parser.add_argument('--num_sample', type=int, default=10,
                        help='the number of samples when perform multi-shot prediction')
    parser.add_argument('--beta', type = float, default=0.1,
                        help = 'beta for balance between information loss and prediction loss')
    parser.add_argument('--infile', type=str, help='input file')
    parser.add_argument('--indepfile', type=str, default=None,
                        help='independent test file')
    parser.add_argument('--blosum', type=str, default='data/BLOSUM50',
                        help='file with BLOSUM matrix')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='batch size')
    parser.add_argument('--model_name', type=str, default='vibi.ckpt',
                        help = 'if train is True, model name to be saved, otherwise model name to be loaded')
    parser.add_argument('--epoch', type = int, default=250, metavar='N',
                        help='number of epoch to train')
    parser.add_argument('--lr', type=float, default=0.00000001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--cuda', type = str2bool, default=True,
                        help = 'enable cuda')
    parser.add_argument('--seed', type=int, default=7405,
                        help='random seed')
    parser.add_argument('--mode', default = 'train', type=str,
                        help = 'train or test')
    
    args = parser.parse_args()

    if args.mode is 'test':
        assert args.indepfile is not None, '--indepfile is missing!'
        
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

    # embedding matrix
    embedding = load_embedding(args.blosum)
    
    ## split data
    n_total = len(y)
    n_train = int(round(n_total * 0.8))
    n_valid = int(round(n_total * 0.1))
    n_test = n_total - n_train - n_valid
    idx_shuffled = np.arange(n_total); np.random.shuffle(idx_shuffled)
    idx_train, idx_valid, idx_test = idx_shuffled[:n_train], \
                                     idx_shuffled[n_train:(n_train+n_valid)], \
                                     idx_shuffled[(n_train+n_valid):]

    ## define dataloader
    train_loader = define_dataloader(X_pep[idx_train], X_tcr[idx_train], y[idx_train], None,
                                     None, None,
                                     batch_size=args.batch_size, device=device)
    valid_loader = define_dataloader(X_pep[idx_valid], X_tcr[idx_valid], y[idx_valid], None,
                                     maxlen_pep=train_loader['pep_length'],
                                     maxlen_tcr=train_loader['tcr_length'],
                                     batch_size=args.batch_size, device=device)
    test_loader = define_dataloader(X_pep[idx_test], X_tcr[idx_test], y[idx_test], None,
                                    maxlen_pep=train_loader['pep_length'],
                                    maxlen_tcr=train_loader['tcr_length'],
                                    batch_size=args.batch_size, device=device)
    ## read indep data
    if args.indepfile is not None:
        X_indep_pep, X_indep_tcr, y_indep = data_io_tf.read_pTCR(args.indepfile)
        y_indep = np.array(y_indep)
        indep_loader = define_dataloader(X_indep_pep, X_indep_tcr, y_indep, None,
                                         maxlen_pep=train_loader['pep_length'],
                                         maxlen_tcr=train_loader['tcr_length'],
                                         batch_size=args.batch_size, device=device)                                     

    ## define model
    model = VIBI(embedding, train_loader['pep_length'], train_loader['tcr_length'],
                 K=args.K, chunk_size=args.chunk_size,
                 num_sample=args.num_sample, cuda=args.cuda).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if 'models' not in os.listdir('.'):
        os.mkdir('models')
    if 'result' not in os.listdir('.'):
        os.mkdir('result')

    ## fit model        
    if args.mode == 'train' : 
            
        #model_name = check_model_name(args.model_name)
        #model_name = check_model_name(model_name, './models')

        wf_open_fixed = open('result/'+os.path.splitext(os.path.basename(args.infile))[0]+\
                             '_'+os.path.splitext(os.path.basename(args.model_name))[0]+'_valid_fixed.csv', 'w')
        wf_open_cont = open('result/'+os.path.splitext(os.path.basename(args.infile))[0]+\
                            '_'+os.path.splitext(os.path.basename(args.model_name))[0]+'_valid_cont.csv', 'w')
        #wf = csv.writer(wf_open)
        wf_colnames = ['total_loss', 'class_loss', 'info_loss',
                       'accuracy', 'precision1', 'precision0', 'recall1', 'recall0',
                       'f1macro','f1micro', 'auc']
        wf_fixed = csv.DictWriter(wf_open_fixed, wf_colnames, delimiter='\t')
        wf_cont = csv.DictWriter(wf_open_cont, wf_colnames, delimiter='\t')

        t0 = time.time()
        for epoch in range(1, args.epoch + 1):
            
            train(args, model, device, train_loader['loader'], optimizer, epoch)

            print('Epoch {} TimeSince {}\n'.format(epoch, timeSince(t0)))
            
            perf_train_fixed, perf_train_cont = get_performance_fidelity_batchiter(train_loader['loader'],
                                                                                   model, prior, args, device)
            perf_valid_fixed, perf_valid_cont = get_performance_fidelity_batchiter(valid_loader['loader'],
                                                                                   model, prior, args, device)
            print('[TRAIN] {} ----------------'.format(epoch))
            print_performance(perf_train_fixed)
            print_performance(perf_train_cont)
            print('[VALID] {} ----------------'.format(epoch))
            print_performance(perf_valid_fixed, writeif=True, wf=wf_fixed)
            print_performance(perf_valid_cont, writeif=True, wf=wf_cont)
            

        print('[TEST ] {} ----------------'.format(epoch))
        perf_test_fixed, perf_test_cont = get_performance_fidelity_batchiter(test_loader['loader'],
                                                                             model, prior, args, device)
        print_performance(perf_test_fixed)
        print_performance(perf_test_cont)
        
        if args.indepfile is not None:
            print('[INDEP] {} ----------------'.format(epoch)) 
            perf_indep_fixed, perf_indep_cont = get_performance_fidelity_batchiter(indep_loader['loader'],
                                                                                   model, prior, args, device)
            print_performance(perf_indep_fixed)
            print_performance(perf_indep_cont)

            wf_vibi_open = open('result/vibi_' + os.path.basename(args.indepfile), 'w')
            wf_vibi = csv.writer(wf_vibi_open, delimiter='\t')
            write_explain_batchiter(indep_loader, model, wf_vibi, device)

        model_name = './models/' + model_name
        torch.save(model.state_dict(), model_name)
           
    elif args.mode == 'test' : 
        
        model_name = args.model_name

        assert model_name in os.listdir('./models')
        
        model_name = './models/' + model_name
        model.load_state_dict(torch.load(model_name))

        print('[INDEP] {} ----------------')
        perf_indep_fixed, perf_indep_cont = get_performance_fidelity_batchiter(indep_loader['loader'],
                                                                               model, prior, args, device)
        print_performance(perf_indep_fixed)
        print_performance(perf_indep_cont)

        wf_vibi_open = open('result/vibi_' + os.path.basename(args.indepfile), 'w')
        wf_vibi = csv.writer(wf_vibi_open, delimiter='\t')
        write_explain_batchiter(indep_loader, model, wf_vibi, device)
        
    else :
        
        print('\nError: "--mode train" or "--mode test" expected')
        
if __name__ == '__main__':
    main()  


