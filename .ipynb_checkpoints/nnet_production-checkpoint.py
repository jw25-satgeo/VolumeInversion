import sys, os, tqdm
import numpy as np

import torch
import torch.nn.functional as F

import nn_time
import nn_reg
import nn_tf

import util_metrics
import util_collector

def prep_reload():
    import importlib
    importlib.reload(nn_time)
    importlib.reload(nn_reg)
    importlib.reload(nn_tf)
    importlib.reload(util_metrics)
    importlib.reload(util_collector)
    
# ----- 


# ----- VMWrapper for Inference_CoreExplorer 2026.10.08

def train_VMWrapper_CoreExplorer(epochs=500, batch_size=1000):
    
    import script_CoreExplorer; importlib.reload(script_CoreExplorer)
    whitener, dwhitenr = torch.nn.Identity(), torch.nn.Identity()
    fn = 'em_resice_260922.hdf5'
    feed, floqs, vl, data_shapes = script_CoreExplorer.get_core(fn, 110000, whitener=whitener)
    incidence_n = len(floqs); 

    # condition mask for inference
    def get_condmask(x): # select & lock nC4 EM signature of 1 incidence in ...,EM_real,EM_imag feature vector
        _mask = torch.ones((x.shape[0], *data_shapes[-1]), device=x.device, dtype=torch.bool)
        k = torch.randint(incidence_n, (x.shape[0],), device=x.device)
        _mask[torch.arange(x.shape[0], device=x.device), :, k] = False # for (N,1,k,P,P)
        _mask = torch.cat([_mask.flatten(1), _mask.flatten(1)], dim=1)
        omask = torch.ones_like(x, dtype=torch.bool)
        omask[..., -_mask.shape[-1]:] = _mask[:,None,:] # transfer to full feature vector
        return omask

    model = nn_tf.r2PA_Transformer(h=16, c=1, dim=feed.shape[-1], hdim=16, layers=4, activation='sigmoid')
    wrapper = nn_reg.VMWrapper(model, feed.shape[-1], n=64, T=1).to('cuda', torch.float32)

    # -- save whitener & dewhitener for later reference
    model.whitener = whitener
    model.unwhitener = dwhitenr

    # 90:10 train:test split
    d_train, d_valid = feed[:-feed.shape[0]//10], feed[-feed.shape[0]//10:]
    # optimizer
    loss_hist, v_loss_hist = [0], [0]
    opt = torch.optim.AdamW(wrapper.parameters(), lr=0.005, fused=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.9)
    # iterated training
    t_loop = tqdm.tqdm(range(epochs))
    for i in t_loop:
        d_perm = torch.randperm(d_train.shape[0])
        for j in range(0, d_train.shape[0], batch_size):
            opt.zero_grad()
            s_train = d_train[d_perm[j:j+batch_size]]
            s_valid = d_valid[np.random.choice(np.arange(d_valid.shape[0]), size=batch_size)]
            loss    = wrapper.train(s_train)
            v_loss  = wrapper.train(s_valid, do_backprop=False).detach()
            opt.step()
            sched.step(v_loss)
            v_loss_hist.append(v_loss.item()); loss_hist.append(loss.detach().item())
            t_loop.set_description(f'epoch {i} - lr : {opt.param_groups[0]['lr']} - loss : {loss_hist[-1]}')
    
    return wrapper, opt, sched
