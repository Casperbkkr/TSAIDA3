import numpy as np
import math



from TS_AIDA.Relative_contrast import rel_contrast

def Eu_norm(X, Ys, p=2):
	Xs = np.repeat(X, Ys.shape[0], axis=0)
	dif = (np.abs(Xs - Ys)) ** p
	return np.sum(dif, axis=1)


def DistanceProfile(seq, coll, p=2):
	seq = seq[np.newaxis, :]
	A = np.repeat(seq, coll.shape[0], axis=0)
	dif = (np.abs(A - coll)) ** p
	S = np.sum(dif, axis=1)
	if len(A.shape) == 3:
		S = np.sum(S, axis=1)
	sorted = np.sort(S)
	del S, dif, A, seq
	return sorted



def Isolation(Z_n, alpha=1):
	Z_top =  Z_n[1:] - Z_n[0:-1]
	Z_bot = Z_n[1:] - Z_n[0]
	div = (Z_top/Z_bot)[1:]
	div = np.nan_to_num(div, copy=True, nan=0.0)
	mean = 1 + np.sum(div)
	var = np.sum(div*(1-div))
	if math.isnan(var):
		pause=1
	Z_top, Z_bot, div = None, None, None
	return mean, var

def DFT(Z, thresh):

	Z_hat = np.fft.fftn(Z, axes=(1,))
	Z_hat = np.abs(Z_hat)
	Z_hat_avg = Z_hat.mean(axis=0)
	K = int(Z_hat.shape[1]*thresh)
	Z_out = np.zeros(shape=[Z.shape[0],K, Z.shape[-1]])
	for d in range(Z_hat.shape[-1]):
		Z_hat_K = np.argpartition(Z_hat_avg[:,d], K, axis=0)[-K:]
		select = np.repeat(Z_hat_K[np.newaxis, :], Z.shape[0], axis=0)
		Z_out[:,:,d] = Z_hat[:, Z_hat_K, d]
	return Z_out, Z_hat

def Score(Z_i, Z, p=1):
	profile = DistanceProfile(Z_i, Z, p=p)
	cont = rel_contrast(profile)
	mean, var = Isolation(profile)
	if math.isnan(var):
		pause=1
	profile =None
	return mean, var, cont

