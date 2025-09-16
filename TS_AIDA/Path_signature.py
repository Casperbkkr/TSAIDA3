import numpy as np
import roughpy as rp
import gc as gc


def path_signature(X_t, K):
    dX_t = np.diff(X_t, axis=0)
    signature_terms = np.ones(shape=dX_t.shape)
    terms_out = [signature_terms]

    for k in range(1, K):
        previous = terms_out[-1]

        integrals = path_integral(previous, dX_t, k, d=X_t.shape[1])
        terms_out.append(integrals)
        terms = [integrals[-1,:] for integrals in terms_out]
        A = np.concatenate(terms, axis=0)

    return A

def path_integral(Z, dX_t, k, d):
    level_out_dim = d**(k)

    if k==0:
        dX_t_i = dX_t
        Z_i = Z
    else:
        dX_t_i = np.repeat(dX_t, [level_out_dim for i in range(dX_t.shape[1])], axis=1)
        Z_i = np.tile(Z, (1,d))

    integral = np.cumsum(Z_i * dX_t_i, axis=0)

    return integral

def sig(X, K):
    if len(X.shape)>2:
        out_d = sum([X.shape[2]**i for i in range(1,K+1)])
    else:
        out_d = 1

    output = np.zeros(shape=[X.shape[0], out_d])
    for t in range(X.shape[0]):
        A = path_signature(X[t, :,:], K)
        output[t, :] = A
    return output

def sig_rp(X, K, interval, indices, context):
    times = indices
    #context = rp.get_context(width=X.shape[0], depth=K, coeffs=rp.DPReal)
    stream = rp.LieIncrementStream.from_increments(X, indices=times, ctx=context)

    out = stream.signature(interval)
    times, stream = None, None

    return out

def sig_rp2(X, K, interval, indices, context):
    if len(X.shape)>2:
        out_d = sum([X.shape[2]**i for i in range(1,K+1)])+1
    else:
        out_d = 1

    output = np.zeros(shape=[X.shape[0], out_d])

    for t in range(X.shape[0]):
        A = sig_rp(X[t,:,:], K, interval, indices, context)
        B = np.array(A)
        output[t, :] = B
    out_d, A, B  = None, None, None
    gc.collect()
    return output
