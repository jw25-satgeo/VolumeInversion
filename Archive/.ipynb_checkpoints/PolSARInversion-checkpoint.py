import numpy as np
import numba as nb
import scipy
from abc import ABC, abstractmethod

class AdamLearner(ABC):
    
    '''An Adam-based parameter inversion basis for RVoG-like algorithms.'''
    def init_learning(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.params)  # Initialize first moment vector
        self.v = np.zeros_like(self.params)  # Initialize second moment vector
        self.t = 0.0     # Initialize timestep

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def param_correction(self):
        pass

    @abstractmethod
    def data_modelled(self, params):
        pass

    @abstractmethod
    def data_captured(self):
        pass
    
    #def finite_diff(self, epsilon, p_err):
    #    delta = np.zeros_like(self.params);
    #    grad = np.zeros_like(self.params)
    #    for i in range(delta.shape[0]):
    #        delta[i] = epsilon[i]
    #        grad[i] = (np.nanmean(np.abs(self.data_modelled(self.params+delta) - self.data_captured())**2,axis=0) - p_err)/epsilon[i]
    #        delta[i] = 0
    #    return grad

    def finite_diff(self, epsilon, p_err):
        # batch : disturbed parameters (dim0: number of disturbances -- dim1: number of parameters)
        #     e.g. for each disturbance, disturb a given parameter by epsilon. n_params = n_disturbances.
        n_params = self.params.shape[0]
        batch = np.repeat(self.params[np.newaxis, ...], n_params, axis=0)
        for i in range(n_params):
            batch[i, i] += epsilon[i]

        # (dims : batch, param, i, j) -{model}-> (dims : batch, pol, i, j) -{nanmean_ax1}-> (dims : batch, i, j)
        modelled = np.array([self.data_modelled(disturbed_param) for disturbed_param in batch])
        err = np.nanmean(np.abs(modelled - self.data_captured())**2, axis=1)
        grad = (err - p_err) / epsilon[:,np.newaxis,np.newaxis]
        return grad

    def converge(self, iterations=100, epsilon=1e-4, error_record=True, error_record_interval=10):
        if self.m is None or self.v is None: self.init_learning()
        if (not isinstance(epsilon, np.ndarray)):
            if(isinstance(epsilon, float)): epsilon = np.full(len(self.params), epsilon)
            else: raise ValueError("epsilon must be float or array.")
        elif (len(epsilon) != len(self.params)): raise ValueError("length of epsilon must either be one or equal to length of params.")
        # -- begin convergence
        error_history = np.zeros(int(iterations / error_record_interval))
        for i in range(iterations):
            p_err = np.nanmean(np.abs(self.data_captured() - self.data_modelled(self.params))**2, axis=0)
            grad = self.finite_diff(epsilon, p_err)
            if (error_record and i%error_record_interval==0):
                error_history[i // error_record_interval] = (np.nanmean(np.nanmean(np.nanmean(p_err, axis=0))))
                print("[##########"[:10*(i)//iterations+1]+"          ]"[10*(i)//iterations:]+str(np.round((i)*100 / iterations,3))+"% - "+str(i)+"/"+str(iterations))
            self.t += 1 # prevent divide by zero (1 - beta1**0 = 0)
            self.params = self.update(self.m, self.v, self.t, self.beta1, self.beta2, self.lr, self.epsilon, self.params, grad)
            self.param_correction()
        return (error_history) if error_record else None

    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def update(m, v, t, beta1, beta2, lr, epsilon, params, grads):
        for i in range(params.shape[0]):
            grad = grads[i]

            # Update biased first moment estimate
            m[i] = beta1 * m[i] + (1 - beta1) * grad

            # Update biased second moment estimate
            v[i] = beta2 * v[i] + (1 - beta2) * grad**2

            # Compute bias-corrected first moment estimate
            m_hat = m[i] / (1 - beta1**t)

            # Compute bias-corrected second moment estimate
            v_hat = v[i] / (1 - beta2**t)

            # Update parameters
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        return params

class RVoW(AdamLearner):
    '''Adaptation of RVoG-like inversion to sea ice
    (src: custom)

    Models the interferometric coherence at a known incidence angle and incidence angle offset
    '''
    def __init__(self, coherences, extinction_func, incidence_angle, k_z, wavelength = 0.12):
        self.coherences = coherences
        self.wavelength = wavelength
        if (np.array(incidence_angle).shape != coherences[0].shape):
            self.incidence_angle = np.full((coherences.shape[1:]), incidence_angle)
        else: self.incidence_angle = incidence_angle
        
        self.k_z = k_z
        self.k_z[np.isinf(self.k_z)] = np.nan

        self.init_params()
        self.init_learning()
        self.init_vol_coh(extinction_func)
        self.find_icesurf_phase()

    def init_vol_coh(self, extinction_func, hgt_min=0.1, hgt_max=1, hgt_bins=20, age_min=10, age_max=200, age_bins=20, k_z_bins=20, smoothing=0.2):
        '''Takes a function and integrates it over ranges to initialise a polynomial curve fit.
        extinction_func(z (depth, m), hgt(m), age, k_z) -> extinction coefficient'''
        print("Beginning coherence function mapping...")
        z = np.linspace(0, hgt_max, int(hgt_bins + hgt_min*(hgt_max-hgt_min)/hgt_bins))
        hgts = np.linspace(hgt_min, hgt_max, hgt_bins)
        ages = np.linspace(age_min, age_max, age_bins)
        mesh = np.meshgrid(z, hgts, ages, indexing='ij')
        x = extinction_func(*mesh) #  y_V = integral(np.exp(-A(z) + 1j*self.k_z*z)) / integral(np.exp(-A(z))
        k_z_max = max(np.max(self.k_z), np.min(self.k_z)+1e-8)
        k_z_r = np.linspace(np.min(self.k_z), k_z_max, k_z_bins)
        #NOTE - the phase term e^(1j * k_z * z) represents z increasing downward, while e^(-1j * k_z * z) has z increasing upwards.
        x = np.stack([np.trapezoid(np.exp(-x + 1j*KZ*mesh[0]), z, axis=0) / np.trapezoid(np.exp(-x), z, axis=0) for KZ in k_z_r], axis=-1)
        self.vol_coh = scipy.interpolate.RegularGridInterpolator((hgts,ages,k_z_r), x, bounds_error=False, fill_value=np.nan)
        print(x.shape)
        print("coherence function initialized.")

    def find_icesurf_phase(self): # -- the typical means applied to RVoG
        #DEPREC - placeholder biggest coherence picker
        ##step 1 - extract highest coherence polarimetric basis unit
        #self.icesurf_phase = np.zeros(self.coherences.shape[1:]).astype(complex)
        #tmp = np.argmax(np.abs(self.coherences), axis=0)
        #for i in range(self.coherences.shape[0]):
        #    self.icesurf_phase[tmp==i] = self.coherences[i][tmp==i]
        ##step 2 - extract corresponding signal
        #self.icesurf_phase = np.arctan2(np.imag(self.icesurf_phase), np.real(self.icesurf_phase))
        
        # Reshape for vectorized regression
        coha_r = np.real(self.coherences).reshape(self.coherences.shape[0], -1).T  # shape: (N_pixels, T)
        coha_i = np.imag(self.coherences).reshape(self.coherences.shape[0], -1).T
        mask = (~np.isnan(coha_r).any(axis=1)) & (~np.isnan(coha_i).any(axis=1)) & (np.nanvar(coha_r, axis=1) > 1e-12)
        X, Y = np.stack([coha_r[mask], np.ones_like(coha_r[mask])], axis=2), coha_i[mask][..., None]  # shape: (N_valid, T, 2)
        # Least squares solution: beta = (X^T X)^-1 X^T y
        Xt = np.transpose(X, (0, 2, 1))
        XtX, XtY = Xt @ X, Xt @ Y
        beta = np.linalg.solve(XtX, XtY).squeeze(-1)  # shape: (N_valid, 2)

        # Fill into coh_lins
        coh_lins = np.zeros((2, self.coherences.shape[1], self.coherences.shape[2]))
        flat_idx = np.where(mask)[0]
        for i, idx in enumerate(flat_idx):
            y, x = divmod(idx, self.coherences.shape[2])
            coh_lins[:, y, x] = beta[i]

        radius = 1
        a, b = coh_lins[0], coh_lins[1]
        c = 4*((a**2 + 1) * radius - b**2)
        c[c < 0] = np.nan; c = np.sqrt(c) #nullifies invalid roots
        x1 = ((-2*a*b - c) / (2 * a**2 + 2)).astype(complex); x1 += (a*x1 + b)*1j
        x2 = ((-2*a*b + c) / (2 * a**2 + 2)).astype(complex); x2 += (a*x2 + b)*1j
        mask = np.abs(x1 - radius) > np.abs(x2 - radius)
        x1[mask] = x2[mask]
        self.icesurf_phase = np.arctan2(np.imag(x1),np.real(x1))
    
    def init_params(self):
        self.params = np.zeros((3+self.coherences.shape[0], self.coherences.shape[1], self.coherences.shape[2]))
        self.params[0] = 0.1 #ice thickness (m)
        self.params[1] = 20 #ice age (days)
        self.params[2] = 0 #
        self.params[3:] = 0.5 #polarimetric surface-volume ratio

    def param_correction(self):
        self.params[3:][self.params[3:]>1] = 1
        self.params[3:][self.params[3:]<0] = 0

    def data_captured(self):
        return self.coherences

    def data_modelled(self, params):
        vcohs_input = np.array([params[0].ravel(), params[1].ravel(), self.k_z.ravel()]).T
        #print(vcohs_input.shape)
        #vcohs_input += (np.random.rand(*vcohs_input.shape)-0.5)*1e-3 #small disturbance to avoid interpolation bug
        vcohs = self.vol_coh(vcohs_input).reshape(params.shape[1:])
        #icebott_phase = self.icesurf_phase + (2*np.pi*params[0]/(self.wavelength*np.cos(self.incidence_angle)))
        cohs = np.exp(1j * self.icesurf_phase) * (vcohs + params[3:]) / (1 + params[3:])
        cohs[np.isnan(cohs)] = 0
        return cohs
        #return self.coh_modelled_jit(params, self.wavelength, self.incidence_angle, self.k_z)

    #@staticmethod
    #@nb.njit(cache=True, fastmath=True, parallel=True)
    #def coh_modelled_jit(params, wvl, ia, k_z, ip, r_int):

class IceVVHHr(AdamLearner):
    '''Discrimination of open water from sea ice in the Labrador Sea using quad-polarized synthetic aperture radar
    (src: https://www.sciencedirect.com/science/article/pii/S0034425720303187)
    
    Models the VV/HH backscatter ratio (vvhhr) at a known incidence angle using relative dielectric constant of the surface (rdie, param[0]) and standard deviation of surface orientation angle shift (stdv, param[1]), representing surface roughness, as parameters.
    
    Possible method - both surface roughness and effective dielectric constant should change with frequency, allowing comparison where multifrequency images are available.'''
    def __init__(self, vvhhr, incidence_angle):
        self.vvhhr = vvhhr
        self.ia = incidence_angle

        self.init_params()
        self.init_learning()

    def init_params(self):
        self.params = np.zeros((2, self.vvhhr.shape[-2], self.vvhhr.shape[-1]))
        self.params[0] = 3.175 #rdie
        self.params[1] = 0.2     #stdv

    def param_correction(self):
        self.params[self.params < 0] = 0

    def data_modelled(self, params):
        return self.simVVHHr(self.ia, params[0], params[1])[np.newaxis,:]

    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def simVVHHr(ia, rdie, stdv):
        F_VV = (rdie - 1) * (np.sin(ia)**2 - rdie*(1 + np.sin(ia)**2)) / (rdie * np.cos(ia) + np.sqrt(rdie - np.sin(ia)**2))
        F_HH = (np.cos(ia) - np.sqrt(rdie - np.sin(ia)**2)) / (np.cos(ia) + np.sqrt(rdie - np.sin(ia)**2))
        rough = np.sqrt(3) * stdv / np.sin(ia)
        A = 0.5*(1+np.sin(2*rough)/2*rough)*F_VV**2 + 0.5*(1-np.sin(2*rough)/2*rough)*F_HH**2 + 0.125*(1-np.sin(4*rough)/4*rough)*(F_VV-F_HH)**2
        B = 0.5*(1+np.sin(2*rough)/2*rough)*F_HH**2 + 0.5*(1-np.sin(2*rough)/2*rough)*F_VV**2 + 0.125*(1-np.sin(4*rough)/4*rough)*(F_VV-F_HH)**2
        return A/B

    def data_captured(self):
        return self.vvhhr

class DFRMoG(AdamLearner):
    def __init__(self, coherences, incidence_angle_map, incidence_angle_delta, wavelength=0.24,
                 init_learning_default=True, init_params_default=True):
        # -- physical parameters
        if (coherences[0].shape != incidence_angle_map.shape):
            raise ValueError("Mismatch in dimension between coherence[0] and incident angle map.")
        self.__coherences = coherences
        self.__incidence_angle_map = incidence_angle_map
        if ((not isinstance(incidence_angle_delta, float)) and (incidence_angle_map.shape != incidence_angle_delta.shape)):
            raise ValueError("Mismatch in dimension between incidence_angle_map and non-singular incidence_angle_delta.")
        self.__k_z = (4*np.pi*incidence_angle_delta/(wavelength*np.sin(incidence_angle_map)))
        self.__k_z[np.isinf(self.__k_z)] = np.nan
        self.__wavelength = wavelength
        self.linefit_coh()
        # -- learning parameters
        if(init_learning_default):
            self.init_learning()
        if(init_params_default):
            self.init_params()

    def init_params(self):
        self.init_params(0.5, 0.5, 1.0, None)
    
    def init_params(self, extinction_coeff, sq_variance, h_v, gvratios):
        param_extinction_coeff = np.full(self.__incidence_angle_map.shape, extinction_coeff)
        param_sq_variance = np.full(self.__incidence_angle_map.shape, sq_variance)
        param_h_v = np.full(self.__incidence_angle_map.shape, h_v)
        
        if (isinstance(gvratios, np.ndarray)):
            if (gvratios.shape == self.__coherences.shape): param_gvratios = gvratios
            else: raise ValueError("dimension mismatch between input gvratios and stored coherence.")
        elif (isinstance(gvratios, float)): param_gvratios = np.full(self.__coherences.shape, gvratios)
        else: param_gvratios = np.full(self.__coherences.shape, 0.5)
        self.params = [param_extinction_coeff, param_sq_variance, param_h_v]
        for gvr in param_gvratios:
            self.params.append(gvr)
        self.params = np.array(self.params)

    def param_correction(self):
        self.params[self.params < 0] = 0
    
    ## --  COHERENCE LINE FIT & GROUND POINT MATCH  -- ##
    ##     output: variables r_int, intersect_phase    ##
    def linefit_coh(self):
        bigcoh = np.argmax(np.abs(self.__coherences), axis=0)
        bigcoh = self.__coherences[bigcoh, np.arange(self.__coherences.shape[1])[:,None], np.arange(self.__coherences.shape[2])]
        self.__r_int = np.abs(bigcoh)

        # Reshape for vectorized regression
        coha_r = np.real(self.__coherences).reshape(self.__coherences.shape[0], -1).T  # shape: (N_pixels, T)
        coha_i = np.imag(self.__coherences).reshape(self.__coherences.shape[0], -1).T
        mask = ~np.isnan(coha_r).any(axis=1) & ~np.isnan(coha_i).any(axis=1)
        X, Y = np.stack([coha_r[mask], np.ones_like(coha_r[mask])], axis=2), coha_i[mask][..., None]  # shape: (N_valid, T, 2)
        # Least squares solution: beta = (X^T X)^-1 X^T y
        Xt = np.transpose(X, (0, 2, 1))  # (N_valid, 2, T)
        XtX, XtY = Xt @ X, Xt @ Y
        beta = np.linalg.solve(XtX, XtY).squeeze(-1)  # shape: (N_valid, 2)

        # Fill into coh_lins
        coh_lins = np.zeros((2, self.__r_int.shape[0], self.__r_int.shape[1]))
        flat_idx = np.where(mask)[0]
        for i, idx in enumerate(flat_idx):
            y, x = divmod(idx, self.__coherences.shape[2])
            coh_lins[:, y, x] = beta[i]
            
        a, b = coh_lins[0], coh_lins[1]
        c = 4*((a**2 + 1) * self.__r_int**2 - b**2)
        c[c < 0] = np.nan; c = np.sqrt(c) #nullifies invalid roots
        x1 = ((-2*a*b - c) / (2 * a**2 + 2)).astype(complex); x1 += (a*x1 + b)*1j
        x2 = ((-2*a*b + c) / (2 * a**2 + 2)).astype(complex); x2 += (a*x2 + b)*1j
        mask = np.abs(x1 - bigcoh) > np.abs(x2 - bigcoh)
        x1[mask] = x2[mask]
        self.__intersect_phase = np.arctan2(np.imag(x1),np.real(x1))
        return

    def data_modelled(self, params):
        return self.coh_modelled(params)

    def data_captured(self):
        return self.__coherences

    # takes in parameter images (2D, L x W) and ratio stack (3D, B x L x W) to form simulated coherence stack (3D)
    def coh_modelled(self, params):
        return self.coh_modelled_jit(params, self.__wavelength, self.__incidence_angle_map, self.__k_z, self.__intersect_phase, self.__r_int)
    
    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def coh_modelled_jit(params, wvl, ia, k_z, ip, r_int):
        ext, sqvr, h_v, gvratios = params[0], params[1], params[2], params[3:]
        cia = np.cos(ia)
        a = 2 * (ext - 0.5 * (4*np.pi / wvl)**2 * sqvr * cia).astype(np.complex128) + 1j*(k_z * cia).astype(np.complex128)
        veg_coh = (2*ext * np.exp(a) * h_v / cia - 1)
        a *= np.exp(2 * ext * h_v / cia) - 1
        for i in range(a.shape[0]):
            for k in range(a.shape[1]):
                if (np.abs(a[i,k]) < 1e-8): a[i,k] = 1
        veg_coh /= a
        cohs = np.zeros_like(gvratios, dtype=np.complex128)
        for i in range(cohs.shape[0]):
            cohs[i] = np.exp(1j * ip) * r_int * (veg_coh + (gvratios[i]/(gvratios[i] + 1))*(1 - veg_coh))
        return cohs

# ----- utility functions for processing ----- #

def coh(s1, s2, window_size=1): 
    wind = np.ones((window_size, window_size))/window_size**2
    edge = int(window_size / 2)
    val = np.sqrt(scipy.signal.convolve2d(np.abs(s1)**2, wind, mode='same') * scipy.signal.convolve2d(np.abs(s2)**2, wind, mode='same'))
    #val[np.abs(val) < 1e-12] = 1e-12
    val = scipy.signal.convolve2d(s1 * np.conj(s2), wind, mode='same') / (val + 1e-12)
    val[:edge] = val[-edge:] = 0#np.nan#0
    val[:,:edge] = val[:,-edge:] = 0#np.nan#0
    return val

def stackcoh(s1, s2, window_size=1):
    return np.stack([coh(s1[i], s2[i], window_size) for i in range(s1.shape[0])])