import multiprocessing
import concurrent.futures

# blargh.

#GCM: 96x96 if square patches are really needed
di, dj = 1152, 624
qp1, qp2 = torch.as_tensor(qpol1).unfold(1,di,di).unfold(2,dj,dj), torch.as_tensor(qpol2).unfold(1,di,di).unfold(2,dj,dj)
cohs = torch.empty(qp1.shape[1],qp1.shape[2],qp1.shape[3],qp1.shape[4],18, dtype=torch.complex64)#, device='cuda')
print (cohs.shape, qp1.shape)
tasks = []
def do_write(i,j):
    cohs[i,j] = PolarDecomp.CoherenceBuilder(qp1[:,i,j],qp2[:,i,j],types='all')[0].cpu()

with concurrent.futures.ThreadPoolExecutor(4) as executor: #max_workers=multiprocessing.cpu_count()-2
    for i in range(qp1.shape[1]):
        for j in range(qp1.shape[2]):
            tasks.append(executor.submit(do_write, i, j))
    with tqdm.tqdm(total=len(tasks)) as pbar:
        for task in concurrent.futures.as_completed(tasks):
            pbar.update(1)
            _ = task.result()
cohs = cohs.permute(0,2,1,3,4).reshape(cohs.size(0)*cohs.size(2),cohs.size(1)*cohs.size(3),cohs.size(4))