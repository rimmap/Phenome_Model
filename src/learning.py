#This is code for the phenome_model inference on the training set.
#There are some MIMIC-ICU specific things in there (for example each diagnosis code only shows up once since we are looking only at one inpatient stay per patient)

#do all the imports, samplers is the cython code.
from __future__ import division; import cython, argparse,pylab;import numpy as np;
import scipy.stats as stats; import time; import scipy.special as special; import cPickle as pickle
import pyximport; pyximport.install(); import samplers as ss

## passing in a bunch of arguments.
parser = argparse.ArgumentParser(description='Learning phenotypes for the phenome model.')
parser.add_argument('--P', type=int,default=5,
                    help='the number of phenotypes (default is 5)')
parser.add_argument('--num_it', type=int,default=7000,
                    help='the number of iterations to run the sampler (default is 7000)')
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
P=args.P; num_iteration=args.num_it; alpha=args.alpha; mu=args.mu; nu=args.nu; xi=args.xi; pi=args.pi

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

with open('../data/examples/diag_counts.txt') as f:
    for i, line in enumerate(f):
        data = [int(x.split(':')[0]) for x in line.strip('\n').split(',')[1:]]
        #count of the icd9 is always 1 because we are only looking at one inpatient stay, we don't need the count
        vs.append(np.array(data,np.int))

with open('../data/examples/word_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        ws.append(np.array(lodata,np.int))
    
with open('../data/examples/med_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        xs.append(np.array(lodata,np.int))

with open('../data/examples/lab_counts.txt') as f:
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
gammas = [np.random.randint(P,size=x) for x in [len(y) for y in vs]]
deltas = [np.random.randint(P,size=x) for x in [len(y) for y in ws]]
epsilons = [np.random.randint(P,size=x) for x in [len(y) for y in xs]]
zetas = [np.random.randint(P,size=x) for x in [len(y) for y in ys]]

#initialize the count variables
def init_vars(passign,dtype,obs):
    for i,record in enumerate(passign):
        for assi in range(len(record)):
            phenotype=record[assi]
            val=obs[i][assi]
            dtype[val,phenotype]+=1
            record_counts[i,phenotype]+=1

#fill in different important variables
record_counts=np.zeros((R,P))
diag_counts=np.zeros((card_I,P))
init_vars(gammas,diag_counts,vs)
diag_pheno=np.sum(diag_counts,0)

doc_counts=np.zeros((card_N,P))
init_vars(deltas,doc_counts,ws)
doc_pheno=np.sum(doc_counts,0)

ord_counts=np.zeros((card_O,P))
init_vars(epsilons,ord_counts,xs)
ord_pheno=np.sum(ord_counts,0)

lab_counts=np.zeros((card_M,P))
init_vars(zetas,lab_counts,ys)
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

#Save the phenotype assignments.
def save_assignments(iteration):
    pickle.dump(gammas,open('../results/gammas_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))
    pickle.dump(deltas,open('../results/deltas_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))
    pickle.dump(epsilons,open('../results/epsilons_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))
    pickle.dump(zetas,open('../results/zetas_'+str(P)+'_iteration_'+str(iteration)+'.pickle','w'))

#initialize a list of log likelihood calculations
ll_trail = []

p_for_cy=np.zeros(P)
for iteration in range(0,num_iteration):
    print iteration
    starttime=time.time() #it's nice to know how long each iteration takes.
    for rec_i in range(R):
        rcounts=record_counts[rec_i,:]
        grand_num=np.random.rand(len(gammas[rec_i]))
        ss.sample_assign(P,alpha,gammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts,grand_num,p_for_cy)
        drand_num=np.random.rand(len(deltas[rec_i]))
        ss.sample_assign(P,alpha,deltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts,drand_num,p_for_cy)
        erand_num=np.random.rand(len(epsilons[rec_i]))
        ss.sample_assign(P,alpha,epsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts,erand_num,p_for_cy)
        zrand_num=np.random.rand(len(zetas[rec_i]))
        ss.sample_assign(P,alpha,zetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts,zrand_num,p_for_cy)
    print "done sampling", (time.time()-starttime)/60.
    if iteration>0 and iteration%(num_iteration/5)==0:
        save_assignments(iteration)
    #calculate the log_likelihood every 5 iterations
    if iteration%5==0:
        ll_trail.append(c_joint_ll(diag_counts,doc_counts,ord_counts,lab_counts))
        with open('../results/log_likelihood_list_'+str(P)+'.data', 'a') as f:
            f.write(str(ll_trail[-1])+'\n')
        if iteration>(num_iteration/10): #once it has burned in, start saving maxima
            if float(ll_trail[-1])>=float(max(ll_trail)): #if this is a maximum, save the phenotypes and phenotype assignments.
                save_assignments(str(iteration)+'max') 
                with open('../results/learned_phenotypes_'+str(P)+'_iteration_'+str(iteration)+'.txt','w') as g:
                    g.write(str(iteration)+'\n'+str(ll_trail[-1])+'\n')
                    for maxptype in range(P):
                        g.write("\nPHENOTYPE "+str(maxptype)+'\n')
                        dsort = np.argsort(diag_counts[:,maxptype])[::-1]
                        g.write("DIAGNOSES\n")
                        g.write(','.join(icd9_vocab[dsort[:20]])+'\n')
                        tsort = np.argsort(doc_counts[:,maxptype])[::-1]
                        g.write("TERMS\n")
                        g.write(','.join(term_vocab[tsort[:20]])+'\n')
                        msort = np.argsort(ord_counts[:,maxptype])[::-1]
                        g.write("MEDS\n")
                        g.write(','.join(med_vocab[msort[:20]])+'\n')
                        lsort = np.argsort(lab_counts[:,maxptype])[::-1]
                        g.write("LABS\n")
                        g.write(','.join(lab_vocab[lsort[:20]])+'\n')
