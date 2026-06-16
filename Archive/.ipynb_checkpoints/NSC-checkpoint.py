import numpy as np
import numba as nb

class NSC: #Natural Stochastic Component
    '''Natural Stochastic Component (relations, dist, c, t, ent) '''
    def __init__(self, c=1, t=1, s=1):
        
        self.c = c #capacity
        self.t = t #temperature
        self.s = s #entropy

        self.q = self.c * self.t #quantity

    def init_rel(self, relations, dist, rgen=None):
        self.relations = relations
        self.d = dist

class Diffusor():

    def __init__(self, nsc_l):
        self.time = 0
        self.dist = np.zeros((len(nsc_l), len(nsc_l)))
        for i in range(len(nsc_l)):
            for k in range(len(nsc_l[i].relations)):
                self.dist[i, nsc_l.index(nsc_l[i].relations[k])] = nsc_l[i].d[k]
        self.dist = np.triu(self.dist)
        self.mask = self.dist > 0

        ctsq = np.stack([(nsc.c, nsc.t, nsc.s, nsc.q) for nsc in nsc_l], axis=1)
        #self.c, self.t, self.s, self.q = (ctsq[i,:,np.newaxis]*np.ones(self.mask.shape) for i in range(ctsq.shape[0]))
        self.c, self.t, self.s, self.q = (ctsq[i] for i in range(ctsq.shape[0]))

    def update(self, default_step = 0.1):
        m = self.mask
        dist = self.dist
        delT, delQ, deltime = self.delta(
            self.c, self.t, self.s, self.q,
            dist, m, default_step
        )
        #print(delT, delQ, deltime)
        self.s += np.log((self.t + delT)/self.t)
        self.t += delT
        self.q += delQ
        self.time += deltime
    
    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def delta(c, t, s, q, dist, m, default_step=0.1):
        n = t.shape[0]
        a = np.zeros(m.shape)
        b = np.zeros(m.shape)
        for i in range(n):
            for k in range(n):
                if (m[i,k]):
                    a[i,k] = 0.5*(1/t[i] - 1/t[k])**2 / ((1/t[i])**2/c[i] + (1/t[k])**2/c[k]) #delS_eq
                    a[i,k] = 0.5 * dist[i,k]**2 * (1/t[i]) * np.exp(-a[i,k]) #diffusivity
                    a[i,k] = -(a[i,k] * (q[i] - q[k]) / dist[i,k]) * default_step #delQ
        delQ = np.sum(a, axis=1) - np.sum(a, axis=0)
#        delT = delQ/c
        #b = np.sum(b, axis=0) + np.sum(b, axis=1)
        stability_scaling = np.max(-delQ / q)
        if (stability_scaling > 0.9999):
            a /= stability_scaling
            delQ = np.sum(a, axis=0) - np.sum(a, axis=1)
        delT = delQ / c
        return delT, delQ, (default_step / stability_scaling)