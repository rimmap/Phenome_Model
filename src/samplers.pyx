import time
import numpy as np
cimport numpy as np
cimport cython

###the normal, basic sampling code ####
@cython.wraparound(False)
@cython.boundscheck(False)
def sample_assign(np.int64_t P,
                  np.float64_t alpha,
                  np.ndarray[np.int64_t, ndim=1] passign, #this is really gammas[rec_i]
                  np.ndarray[np.int64_t, ndim=1] obs, #this is really vs[rec_i]
                  np.ndarray[np.float64_t, ndim=2] dtype, #this is really diag_counts
                  np.ndarray[np.float64_t, ndim=1] dtype_pheno, #this is really diag_pheno
                  np.float64_t prior, #this i really mu
                  np.ndarray[np.float64_t, ndim=1] rcounts, #this is really record_counts[rec_i]
                  np.ndarray[np.float64_t, ndim=1] rand_nums, # this is really grand_num
                  np.ndarray[np.float64_t, ndim=1] probs): #this is really p_for_cy
    cdef Py_ssize_t i, j, k
    cdef np.float64_t mysum, cursor,V_prior,location
    cdef np.int64_t phenotype, val, new_topic
    V_prior=dtype.shape[0]*prior
    # For each data element
    for i in range(len(passign)):
        phenotype=passign[i]
        val = obs[i]
        # Subtract counts
        rcounts[phenotype] -= 1
        dtype[val,phenotype] -= 1
        dtype_pheno[phenotype] -= 1
        # Calculate probabilities
        mysum = 0
        for j in range(P): 
            probs[j] = (alpha + rcounts[j])*(prior + dtype[val,j])/(V_prior + dtype_pheno[j])
            mysum += probs[j]
        # Sample
        location = rand_nums[i]*mysum
        cursor = 0
        for k in range(P):
            cursor += probs[k]
            if cursor >= location:
                new_topic = k
                break
        # Add counts
        passign[i] = new_topic
        rcounts[new_topic]+=1
        dtype[val,new_topic] += 1
        dtype_pheno[new_topic] += 1

### sampling the test set documents (dont change the topics) #### 
@cython.wraparound(False)
@cython.boundscheck(False)
def test_sample(np.int64_t P,
                  np.float64_t alpha,
                  np.ndarray[np.int64_t, ndim=1] passign, #this is really gammas[rec_i]
                  np.ndarray[np.int64_t, ndim=1] obs, #this is really vs[rec_i]
                  np.ndarray[np.float64_t, ndim=2] dtype, #this is really diag_counts
                  np.ndarray[np.float64_t, ndim=1] dtype_pheno, #this is really diag_pheno
                  np.float64_t prior, #this i really mu
                  np.ndarray[np.float64_t, ndim=1] rcounts, #this is really record_counts[rec_i]
                  np.ndarray[np.float64_t, ndim=1] rand_nums, # this is really grand_num
                  np.ndarray[np.float64_t, ndim=1] probs): #this is really p_for_cy
    cdef Py_ssize_t i, j, k
    cdef np.float64_t mysum, cursor,V_prior,location
    cdef np.int64_t phenotype, val, new_topic
    V_prior=dtype.shape[0]*prior
    # For each data element
    for i in range(len(passign)):
        phenotype=passign[i]
        val = obs[i]
        # Subtract counts
        rcounts[phenotype] -= 1
        # Calculate probabilities
        mysum = 0
        for j in range(P): 
            probs[j] = (alpha + rcounts[j])*(prior + dtype[val,j])/(V_prior + dtype_pheno[j])
            mysum+=probs[j]
        #Sample
        location = rand_nums[i]*mysum
        cursor = 0
        for k in range(P):
            cursor += probs[k]
            if cursor >= location:
                new_topic = k
                break
        # Add counts
        passign[i] = new_topic
        rcounts[new_topic]+=1



######## CODE FOR THE HELD-OUT LIKELIHOOD CALCULATION ######

### this runs the sampler but takes the max (this gets h* for the held-out likelihood)
@cython.wraparound(False)
@cython.boundscheck(False)
def sample_max(np.int64_t P,
                  np.float64_t alpha,
                  np.ndarray[np.int64_t, ndim=1] passign, #this is really gammas[rec_i]
                  np.ndarray[np.int64_t, ndim=1] obs, #this is really vs[rec_i]
                  np.ndarray[np.float64_t, ndim=2] dtype, #this is really diag_counts
                  np.ndarray[np.float64_t, ndim=1] dtype_pheno, #this is really diag_pheno
                  np.float64_t prior, #this i really mu
                  np.ndarray[np.float64_t, ndim=1] rcounts, #this is really record_counts[rec_i]
                  np.ndarray[np.float64_t, ndim=1] rand_nums, # this is really grand_num
                  np.ndarray[np.float64_t, ndim=1] probs): #this is really p_for_cy
    cdef Py_ssize_t i, j, k
    cdef np.float64_t mysum, cursor,V_prior,location, mymax
    cdef np.int64_t phenotype, val, new_topic
    V_prior=dtype.shape[0]*prior
    # For each data element
    for i in range(len(passign)):
        phenotype=passign[i]
        val = obs[i]
        # Subtract counts
        rcounts[phenotype] -= 1
        # Calculate probabilities
        mymax=0
        for j in range(P): 
            probs[j] = (alpha + rcounts[j])*(prior + dtype[val,j])/(V_prior + dtype_pheno[j])
            if probs[j]>mymax:
                mymax=probs[j]
                new_topic=j
        # Add counts
        passign[i] = new_topic
        rcounts[new_topic]+=1
        
### sampling the test documents...but backwards! ###
@cython.wraparound(False)
@cython.boundscheck(False)
def test_backwards_sample(np.int64_t P,
                  np.float64_t alpha,
                  np.ndarray[np.int64_t, ndim=1] passign, #this is really gammas[rec_i]
                  np.ndarray[np.int64_t, ndim=1] obs, #this is really vs[rec_i]
                  np.ndarray[np.float64_t, ndim=2] dtype, #this is really diag_counts
                  np.ndarray[np.float64_t, ndim=1] dtype_pheno, #this is really diag_pheno
                  np.float64_t prior, #this i really mu
                  np.ndarray[np.float64_t, ndim=1] rcounts, #this is really record_counts[rec_i]
                  np.ndarray[np.float64_t, ndim=1] rand_nums, # this is really grand_num
                  np.ndarray[np.float64_t, ndim=1] probs): #this is really p_for_cy
    cdef Py_ssize_t i, j, k
    cdef np.float64_t mysum, cursor,V_prior,location
    cdef np.int64_t phenotype, val, new_topic
    V_prior=dtype.shape[0]*prior
    # For each data element
    for i in range(len(passign)-1,-1,-1):
        phenotype=passign[i]
        val = obs[i]
        # Subtract counts
        rcounts[phenotype] -= 1
        # Calculate probabilities
        mysum = 0
        for j in range(P): 
            probs[j] = (alpha + rcounts[j])*(prior + dtype[val,j])/(V_prior + dtype_pheno[j])
            mysum+=probs[j]
        #Sample
        location = rand_nums[i]*mysum
        cursor = 0
        for k in range(P):
            cursor += probs[k]
            if cursor >= location:
                new_topic = k
                break
        # Add counts
        passign[i] = new_topic
        rcounts[new_topic]+=1

### this is the T function for the Iaian Murray "Figure 1" MC
@cython.wraparound(False)
@cython.boundscheck(False)
def the_T(np.int64_t P,
          np.float64_t alpha,
          np.ndarray[np.int64_t, ndim=1] passign, #this is really gammas[rec_i]
          np.ndarray[np.int64_t, ndim=1] hstar_passign, #this is really gammas[rec_i] of hstar
          np.ndarray[np.int64_t, ndim=1] obs, #this is really vs[rec_i]
          np.ndarray[np.float64_t, ndim=2] dtype, #this is really diag_counts
          np.ndarray[np.float64_t, ndim=1] dtype_pheno, #this is really diag_pheno
          np.float64_t prior, #this is really mu
          np.ndarray[np.float64_t, ndim=1] rcounts_fake, #this keeps track of my h* changes
          np.ndarray[np.float64_t, ndim=1] rand_nums, # this is really grand_num
          np.ndarray[np.float64_t, ndim=1] probs): #this is really p_for_cy
    cdef Py_ssize_t i, j
    cdef np.float64_t mysum, V_prior,hstar_prob
    cdef np.int64_t phenotype, val, hstar_phenotype
    V_prior=dtype.shape[0]*prior
    hstar_prob=0
    # For each data element
    for i in range(len(passign)):
        phenotype=passign[i]
        hstar_phenotype=hstar_passign[i]
        val = obs[i]
        # Subtract counts
        rcounts_fake[phenotype] -= 1
        # Calculate probabilities
        mysum = 0
        for j in range(P): 
            probs[j] = (alpha + rcounts_fake[j])*(prior + dtype[val,j])/(V_prior + dtype_pheno[j])
            mysum+=probs[j]
        #Pick the h* phenotype-assignment
        hstar_prob+=np.log((probs[hstar_phenotype])/mysum)
        # Add counts
        rcounts_fake[hstar_phenotype]+=1
    return hstar_prob
