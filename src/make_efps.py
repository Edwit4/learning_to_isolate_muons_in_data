import lmdb
import numpy as np
import pandas as pd
import energyflow as ef

if __name__ == '__main__':

    kappa = [-1.,0.,0.25,0.5,1.,2.]
    beta = [0.25,0.5,1.,2.,3.,4.]
    map_size = 5*10995116278
    n_jobs = 64
    data_dir = 'data'
    csv_dir = f'{data_dir}/cms_csvs'
    pfn_dir = f'{data_dir}/pfn'

    muon_pTs = pd.read_csv(f'{csv_dir}/muon_pTs.csv', header=None).to_numpy()[:,0]
    pfn_env = lmdb.open(f'{pfn_dir}',readonly=True)
    pfn_txn = pfn_env.begin()
    pfn_reps = []
    for i in range(pfn_env.stat()['entries']):
        pfn_reps.append(np.frombuffer(pfn_txn.get(str(i).encode())).reshape(128,4))
    pfn_env.close()
    pfn_reps = np.array(pfn_reps)
    pfn_reps[:,:,0] = pfn_reps[:,:,0] / muon_pTs[:,np.newaxis]
    nonzeros = np.sum(pfn_reps[:,:,0],axis=1)!=0
    pfn_reps = pfn_reps[nonzeros]
    pfn_reps = [pfn_reps[i,:,:3][pfn_reps[i,:,0]>0] for i in range(len(pfn_reps))]

    efp_sets = [[ef.EFPSet(('d<=',7),('n<=',7),('p<=',1), measure='hadr', beta=b, kappa=k, normed=True)
        for b in beta] for k in kappa]
    num_efps = len(efp_sets[0][0].graphs(('d<=',7)))

    for i in range(len(kappa)):
        for j in range(len(beta)):

            index_1d = i*len(beta)+j

            print(f'Computing set {index_1d+1}/{len(kappa)*len(beta)}\r',end='')

            efps = np.zeros((len(nonzeros),num_efps))

            efps[nonzeros] = efp_sets[i][j].batch_compute(pfn_reps, n_jobs=n_jobs)

            efp_env = lmdb.open(
                f'{data_dir}/cms_efp_k{str(kappa[i])}_b{str(beta[j])}',
                map_size=map_size)
            efp_txn = efp_env.begin(write=True)

            for k in range(len(efps)):
                efp_txn.put(key=str(k).encode(),value=efps[k])
            efp_txn.commit()
            efp_env.close()