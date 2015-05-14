#This code uses the Iain Murray method to estimate the likelihood of the held out data.
from __future__ import division; import cython,argparse,random,math,re,json,pylab;pylab.ion(); import numpy as np; import scipy.stats as stats; import time; import scipy.special as special; import cPickle as pickle; import pyximport; pyximport.install(); import samplers as ss;


## passing in a bunch of arguments.
parser = argparse.ArgumentParser(description='Calculating held-out likelihood for the phenome model.')
parser.add_argument('which_train', type=str,
                    help='which of the saved training set iterations would you like to use, you should have a set of them in the results directory.  I usually pick the last maximum one so something like "5770max"')
parser.add_argument('which_test', type=str,
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
P=args.P; num_iteration=args.num_it; alpha=args.alpha; mu=args.mu; nu=args.nu; xi=args.xi; pi=args.pi; which_train=args.which_train; which_test=args.which_test

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

#Output file for all of the log probabilites we learn at each iteration
outfile=open('../results/log_probs'+str(P)+'.txt','w')

#Priors and initializations
card_I = len(icd9_vocab); card_N = len(term_vocab); card_O = len(med_vocab); card_M = L = len(lab_vocab) 
vs = []; ws = []; xs = []; ys = [] #these are test data

with open('../data/examples/test_diag_counts.txt') as f:
    for i, line in enumerate(f):
        data = [int(x.split(':')[0]) for x in line.strip('\n').split(',')[1:]] #count of the icd9 is always 1, we don't need the count, and first one is the hadmid
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

train_vs = []; train_ws = []; train_xs = []; train_ys = [] #training samples

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
     
# Number of Records
R = len(ys)

# Initialize latent variables
train_gammas=pickle.load(open('../results/gammas_'+str(P)+'_iteration_'+which_train+'.pickle'))
train_deltas=pickle.load(open('../results/deltas_'+str(P)+'_iteration_'+which_train+'.pickle'))
train_epsilons=pickle.load(open('../results/epsilons_'+str(P)+'_iteration_'+which_train+'.pickle'))
train_zetas=pickle.load(open('../results/zetas_'+str(P)+'_iteration_'+which_train+'.pickle'))

tgammas=pickle.load(open('../results/TESTgammas_'+str(P)+'_iteration_'+which_test+'.pickle'))
tdeltas=pickle.load(open('../results/TESTdeltas_'+str(P)+'_iteration_'+which_test+'.pickle'))
tepsilons=pickle.load(open('../results/TESTepsilons_'+str(P)+'_iteration_'+which_test+'.pickle'))
tzetas=pickle.load(open('../results/TESTzetas_'+str(P)+'_iteration_'+which_test+'.pickle'))

#initialize the count variables
def init_vars(passign,dtype,obs):
    for i,record in enumerate(passign):
        for assi in range(len(record)):
            phenotype=record[assi]
            val=obs[i][assi]
            dtype[val,phenotype]+=1

record_counts=np.zeros((R,P)) #testing
tvars=[tgammas,tdeltas,tepsilons,tzetas] #testing
for teach in tvars:
    for i,record in enumerate(teach):
        for assi in range(len(record)):
            record_counts[i,record[assi]]+=1 

diag_counts=np.zeros((card_I,P))
test_diag_counts=np.zeros((card_I,P))
init_vars(tgammas,test_diag_counts,vs)
init_vars(train_gammas,diag_counts,train_vs)
diag_p=(diag_counts+mu)/np.sum(diag_counts+mu,0)
diag_pheno=np.sum(diag_counts,0)

doc_counts=np.zeros((card_N,P))
test_doc_counts=np.zeros((card_N,P))
init_vars(tdeltas,test_doc_counts,ws)
init_vars(train_deltas,doc_counts,train_ws)
doc_p=(doc_counts+nu)/np.sum(doc_counts+nu,0)
doc_pheno=np.sum(doc_counts,0)

ord_counts=np.zeros((card_O,P))
test_ord_counts=np.zeros((card_O,P))
init_vars(tepsilons,test_ord_counts,xs)
init_vars(train_epsilons,ord_counts,train_xs)
ord_p=(ord_counts+xi)/np.sum(ord_counts+xi,0)
ord_pheno=np.sum(ord_counts,0)

lab_counts=np.zeros((card_M,P))
test_lab_counts=np.zeros((card_M,P))
init_vars(tzetas,test_lab_counts,ys)
init_vars(train_zetas,lab_counts,train_ys)
lab_p=(lab_counts+pi)/np.sum(lab_counts+pi,0)
lab_pheno=np.sum(lab_counts,0)

#Define the collapsed log-likelihood
def c_joint_ll(record_counts,diag_p,test_diag_counts,doc_p,test_doc_counts,ord_p,test_ord_counts,lab_p,test_lab_counts):
    ll=0
    ll+=R*(special.gammaln(alpha*P)-P*special.gammaln(alpha))
    ll+=np.sum(special.gammaln(alpha+record_counts))-np.sum(special.gammaln(np.sum((record_counts+alpha),1)))
    ll+=np.sum(test_diag_counts*np.log(diag_p))
    ll+=np.sum(test_doc_counts*np.log(doc_p))
    ll+=np.sum(test_ord_counts*np.log(ord_p))
    ll+=np.sum(test_lab_counts*np.log(lab_p))
    return ll


##### This is how to calculate perplexity #####
p_for_cy=np.zeros(P)
#first run the sampler to identify h_star
for rec_i in range(R):
    rcounts=record_counts[rec_i,:]
    grand_num=np.random.rand(len(tgammas[rec_i])) #test: gammas, vs, test #train: diag_counts, diag_pheno
    ss.sample_max(P,alpha,tgammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts,grand_num,p_for_cy)
    drand_num=np.random.rand(len(tdeltas[rec_i]))
    ss.sample_max(P,alpha,tdeltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts,drand_num,p_for_cy)
    erand_num=np.random.rand(len(tepsilons[rec_i]))
    ss.sample_max(P,alpha,tepsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts,erand_num,p_for_cy)
    zrand_num=np.random.rand(len(tzetas[rec_i]))
    ss.sample_max(P,alpha,tzetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts,zrand_num,p_for_cy)
#calculate Pvh here since you have h* right here
pvh=c_joint_ll(record_counts,diag_p,test_diag_counts,doc_p,test_doc_counts,ord_p,test_ord_counts,lab_p,test_lab_counts)
hstar_rcounts=record_counts.copy()
hstar_gammas=[x.copy() for x in tgammas]
hstar_deltas=[x.copy() for x in tdeltas]
hstar_epsilons=[x.copy() for x in tepsilons]
hstar_zetas=[x.copy() for x in tzetas]
bigS=1000
littles=random.randint(1,bigS)
to_average=[]
#use reverse gibbs sampler to identify middle element.
for rec_i in reversed(range(R)):
    rcounts=record_counts[rec_i,:]
    zrand_num=np.random.rand(len(tzetas[rec_i]))
    ss.test_backwards_sample(P,alpha,tzetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts,zrand_num,p_for_cy)
    erand_num=np.random.rand(len(tepsilons[rec_i]))
    ss.test_backwards_sample(P,alpha,tepsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts,erand_num,p_for_cy)
    drand_num=np.random.rand(len(tdeltas[rec_i]))
    ss.test_backwards_sample(P,alpha,tdeltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts,drand_num,p_for_cy)
    grand_num=np.random.rand(len(tgammas[rec_i]))
    ss.test_backwards_sample(P,alpha,tgammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts,grand_num,p_for_cy)
print "got middle element and littles is", littles
##calculate the log(T(h_littles->h*)
hstar_prob=np.float64(0.)
for rec_i in range(R):
    rcounts_fake=record_counts.copy()[rec_i,:]
    grand_num=np.random.rand(len(tgammas[rec_i]))
    hstar_prob+=ss.the_T(P,alpha,tgammas[rec_i],hstar_gammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts_fake,grand_num,p_for_cy)
    drand_num=np.random.rand(len(tdeltas[rec_i]))
    hstar_prob+=ss.the_T(P,alpha,tdeltas[rec_i],hstar_deltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts_fake,drand_num,p_for_cy)
    erand_num=np.random.rand(len(tepsilons[rec_i]))
    hstar_prob+=ss.the_T(P,alpha,tepsilons[rec_i],hstar_epsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts_fake,erand_num,p_for_cy)
    zrand_num=np.random.rand(len(tzetas[rec_i]))
    hstar_prob+=ss.the_T(P,alpha,tzetas[rec_i],hstar_zetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts_fake,zrand_num,p_for_cy)
to_average.append(hstar_prob)
outfile.write(str(hstar_prob)+'\n')
##use reverse gibbs to go from littles to 1
print "going down under"
for i in reversed(range(littles)):
    start=time.clock()
    print i
    for rec_i in reversed(range(R)):
        rcounts=record_counts[rec_i,:]
        zrand_num=np.random.rand(len(tzetas[rec_i]))
        ss.test_backwards_sample(P,alpha,tzetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts,zrand_num,p_for_cy)
        erand_num=np.random.rand(len(tepsilons[rec_i]))
        ss.test_backwards_sample(P,alpha,tepsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts,erand_num,p_for_cy)
        drand_num=np.random.rand(len(tdeltas[rec_i]))
        ss.test_backwards_sample(P,alpha,tdeltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts,drand_num,p_for_cy)
        grand_num=np.random.rand(len(tgammas[rec_i]))
        ss.test_backwards_sample(P,alpha,tgammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts,grand_num,p_for_cy)
    #after each iteration, calculate log(T->h*) and add it to to_sum
    hstar_prob=np.float64(0.)
    for rec_i in range(R):
        rcounts_fake=record_counts.copy()[rec_i,:]
        grand_num=np.random.rand(len(tgammas[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tgammas[rec_i],hstar_gammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts_fake,grand_num,p_for_cy)
        drand_num=np.random.rand(len(tdeltas[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tdeltas[rec_i],hstar_deltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts_fake,drand_num,p_for_cy)
        erand_num=np.random.rand(len(tepsilons[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tepsilons[rec_i],hstar_epsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts_fake,erand_num,p_for_cy)
        zrand_num=np.random.rand(len(tzetas[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tzetas[rec_i],hstar_zetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts_fake,zrand_num,p_for_cy)
    to_average.append(hstar_prob)
    outfile.write(str(hstar_prob)+'\n')
    elapsed=(time.clock()-start)/60
    print elapsed
#use normal gibbs
print "going back up"
for i in range(littles+1,bigS):
    print i
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
    #after each iteration, calculate log(T->h*) and add it to to_sum
    hstar_prob=np.float64(0.)
    for rec_i in range(R):
        rcounts_fake=record_counts.copy()[rec_i,:]
        grand_num=np.random.rand(len(tgammas[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tgammas[rec_i],hstar_gammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts_fake,grand_num,p_for_cy)
        drand_num=np.random.rand(len(tdeltas[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tdeltas[rec_i],hstar_deltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts_fake,drand_num,p_for_cy)
        erand_num=np.random.rand(len(tepsilons[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tepsilons[rec_i],hstar_epsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts_fake,erand_num,p_for_cy)
        zrand_num=np.random.rand(len(tzetas[rec_i]))
        hstar_prob+=ss.the_T(P,alpha,tzetas[rec_i],hstar_zetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts_fake,zrand_num,p_for_cy)
    to_average.append(hstar_prob)
    outfile.write(str(hstar_prob)+'\n')
#divide pvh by the average of "to_average" - there is a trick here to deal with the teeny tiny numbers, and you need this arbitrary precision package called mpmath
import mpmath

to_avg=[mpmath.mpf(x) for x in to_average]
the_min=min(to_avg)
to_avg_min=[x-min(to_avg) for x in to_avg]
to_avg_exp=[mpmath.exp(x) for x in to_avg_min]
the_sum=mpmath.mpf(0)
for each in to_avg_exp:
    the_sum+=each
logged=mpmath.log(the_sum/1000)
denom=logged+the_min
perplexity=pvh-denom #subtract because we are in log space
print perplexity,"the held out likelihood"
outfile.write('\n'+str(perplexity)+'\n')         
outfile.close()
