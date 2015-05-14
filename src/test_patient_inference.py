#This code infers the test set distribution given the topics learned by the training set.
#To do this, we sample using: test set "record_counts", test set "gammas, deltas, etc." and training set "diag_counts, ord_counts,etc"

from __future__ import division; import cython,random,operator,argparse,math,re,json,pylab;pylab.ion(); import numpy as np; import scipy.stats as stats; import time; import scipy.special as special; import cPickle as pickle; 
import pyximport; pyximport.install(); import samplers as ss;

## passing in a bunch of arguments. This one has an extra argument which asks which iteration from the learning do you want to use.
parser = argparse.ArgumentParser(description='Learning phenotypes for unseen patients.')
parser.add_argument('which_train', type=str,
                    help='which of the saved training set iterations would you like to use, you should have a set of them in the results directory.  I usually pick the last maximum one so something like "5770max"')
parser.add_argument('--P', type=int,default=5,
                    help='the number of phenotypes (default is 5)')
parser.add_argument('--num_it', type=int,default=1000,
                    help='the number of iterations to run the sampler (default is 1000)')
parser.add_argument('--alpha', type=float,default=0.1,
                    help='the alpha (prior on patient-phenotype distribution) parameter. default is 0.1')
parser.add_argument('--mu', type=float,default=0.1,
                    help='the mu (prior on diagnosis code-phenotype distribution) parameter. default is 0.1')
parser.add_argument('--nu', type=float,default=0.1,
                    help='the nu (prior on words-phenotype distribution) parameter. default is 0.1')
parser.add_argument('--xi', type=float,default=0.1,
                    help='the xi (prior on medications-phenotype distribution) parameter. default is 0.1')
parser.add_argument('--pi', type=float,default=0.1,
                    help='the pi (prior on laboratory-phenotype distribution) parameter. default is 0.1')

args=parser.parse_args()
P=args.P; num_iteration=args.num_it; alpha=args.alpha; mu=args.mu; nu=args.nu; xi=args.xi; pi=args.pi; which_train=args.which_train

# Load vocabularies
with open('../data/examples/diag_vocab.txt') as f:
    icd9_vocab = np.array(f.read().split('\n'))
    #remove last in list because it is always empty space
icd9_vocab=np.delete(icd9_vocab,-1)
with open('../data/examples/word_vocab.txt') as f:
    term_vocab = np.array(f.read().split('\n'))
term_vocab=np.delete(term_vocab,-1)
with open('../data/examples/med_vocab.txt') as f:
    med_vocab = np.array(f.read().split('\n'))
med_vocab=np.delete(med_vocab,-1)
with open('../data/examples/lab_vocab.txt') as f:
    lab_vocab = np.array(f.read().split('\n'))
lab_vocab=np.delete(lab_vocab,-1)


#Priors and initializations
card_I = len(icd9_vocab); card_N = len(term_vocab); card_O = len(med_vocab); card_M = L = len(lab_vocab) 
vs = []; ws = []; xs = []; ys = []

with open('../data/examples/test_diag_counts.txt') as f:
    for i, line in enumerate(f):
        data = [int(x.split(':')[0]) for x in line.strip('\n').split(',')[1:]]
        #count of the icd9 is always 1 because we are only looking at one inpatient stay, we don't need the count
        vs.append(np.array(data,np.int))

with open('../data/examples/test_word_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        ws.append(np.array(lodata,np.int))    
with open('../data/examples/test_med_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        xs.append(np.array(lodata,np.int))

with open('../data/examples/test_lab_counts.txt') as f:
    for i, line in enumerate(f):
        data=[x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            lab_id=each.split(':')[0]
            for i in range(len(each.split(';'))):
                lodata.append(int(lab_id))
        ys.append(np.array(lodata,np.int))
        
# Number of Records
R = len(ys)

# Initialize latent variables
tgammas = [np.random.randint(P,size=x) for x in [len(y) for y in vs]]
tdeltas = [np.random.randint(P,size=x) for x in [len(y) for y in ws]]
tepsilons = [np.random.randint(P,size=x) for x in [len(y) for y in xs]]
tzetas = [np.random.randint(P,size=x) for x in [len(y) for y in ys]]

#initialize the count variables
def init_vars(passign,dtype,obs):
    for i,record in enumerate(passign):
        for assi in range(len(record)):
            phenotype=record[assi]
            val=obs[i][assi]
            dtype[val,phenotype]+=1
            
record_counts=np.zeros((R,P))
tvars=[tgammas,tdeltas,tepsilons,tzetas]
for teach in tvars:
    for i,record in enumerate(teach):
        for assi in range(len(record)):
            record_counts[i,record[assi]]+=1

#Read in the trained phenotype assignments
train_gammas=pickle.load(open('../results/gammas_'+str(P)+'_iteration_'+which_train+'.pickle'))
train_deltas=pickle.load(open('../results/deltas_'+str(P)+'_iteration_'+which_train+'.pickle'))
train_epsilons=pickle.load(open('../results/epsilons_'+str(P)+'_iteration_'+which_train+'.pickle'))
train_zetas=pickle.load(open('../results/zetas_'+str(P)+'_iteration_'+which_train+'.pickle'))

train_vs = []; train_ws = []; train_xs = []; train_ys = []

with open('../data/examples/diag_counts.txt') as f:
    for i, line in enumerate(f):
        data = [int(x.split(':')[0]) for x in line.strip('\n').split(',')[1:]] #count of the icd9 is always 1, we don't need the count, and first one is the hadmid
        train_vs.append(np.array(data,np.int))

with open('../data/examples/word_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        train_ws.append(np.array(lodata,np.int))
    
with open('../data/examples/med_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        train_xs.append(np.array(lodata,np.int))

with open('../data/examples/lab_counts.txt') as f:
    for i, line in enumerate(f):
        data=[x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            lab_id=each.split(':')[0]
            for i in range(len(each.split(';'))):
                lodata.append(int(lab_id))
        train_ys.append(np.array(lodata,np.int))

diag_counts=np.zeros((card_I,P))
init_vars(train_gammas,diag_counts,train_vs)
diag_pheno=np.sum(diag_counts,0)

doc_counts=np.zeros((card_N,P))
init_vars(train_deltas,doc_counts,train_ws)
doc_pheno=np.sum(doc_counts,0)

ord_counts=np.zeros((card_O,P))
init_vars(train_epsilons,ord_counts,train_xs)
ord_pheno=np.sum(ord_counts,0)

lab_counts=np.zeros((card_M,P))
init_vars(train_zetas,lab_counts,train_ys)
lab_pheno=np.sum(lab_counts,0)

#Define the collapsed log-likelihood
def c_joint_ll(diag_counts,doc_counts,ord_counts,lab_counts):
    ll=0
    ll+=special.gammaln(alpha*P)-P*special.gammaln(alpha)
    ll+=np.sum(special.gammaln(alpha+record_counts))-np.sum(special.gammaln(np.sum((record_counts+alpha),1)))
    ll+=P*(special.gammaln(mu*card_I)-card_I*special.gammaln(mu)+
        special.gammaln(nu*card_N)-card_N*special.gammaln(nu)+
        special.gammaln(xi*card_O)-card_O*special.gammaln(xi)+
        special.gammaln(pi*card_M)-card_M*special.gammaln(pi))
    ll+=np.sum(special.gammaln(mu+diag_counts))-np.sum(special.gammaln(mu*card_I+np.sum(diag_counts,0)))
    ll+=np.sum(special.gammaln(nu+doc_counts))-np.sum(special.gammaln(nu*card_N+np.sum(doc_counts,0)))
    ll+=np.sum(special.gammaln(xi+ord_counts))-np.sum(special.gammaln(xi*card_O+np.sum(ord_counts,0)))
    ll+=np.sum(special.gammaln(pi+lab_counts))-np.sum(special.gammaln(pi*card_M+np.sum(lab_counts,0)))
    return ll


def save_assignments(iteration):
    pickle.dump(tgammas,open('../results/TESTgammas_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))
    pickle.dump(tdeltas,open('../results/TESTdeltas_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))
    pickle.dump(tepsilons,open('../results/TESTepsilons_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))
    pickle.dump(tzetas,open('../results/TESTzetas_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))
    #this also saves record counts, which tells us the phenotype distribution for the test patients
    pickle.dump(record_counts,open('../results/TESTrecord_counts_matrix'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))

#initialize a list of log likelihood calculations
ll_trail = []

p_for_cy=np.zeros(P)
for iteration in range(0,num_iteration):
    print iteration
    for rec_i in range(R):
        rcounts=record_counts[rec_i,:]
        grand_num=np.random.rand(len(tgammas[rec_i]))
        ss.test_sample(P,alpha,tgammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts,grand_num,p_for_cy)
        drand_num=np.random.rand(len(tdeltas[rec_i]))
        ss.test_sample(P,alpha,tdeltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts,drand_num,p_for_cy)
        erand_num=np.random.rand(len(tepsilons[rec_i]))
        ss.test_sample(P,alpha,tepsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts,erand_num,p_for_cy)
        zrand_num=np.random.rand(len(tzetas[rec_i]))
        ss.test_sample(P,alpha,tzetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts,zrand_num,p_for_cy)
    if iteration>0 and iteration%(num_iteration/5)==0:
        save_assignments(iteration)
    if iteration%5==0:
        ll_trail.append(c_joint_ll(diag_counts,doc_counts,ord_counts,lab_counts))
        with open('../results/TEST_log_likelihood_list_'+str(P)+'.data', 'a') as f:
            f.write(str(ll_trail[-1])+'\n')
        if iteration>(num_iteration/5):
            if float(ll_trail[-1])>=float(max(ll_trail)):
                save_assignments(str(iteration)+'max')
                
