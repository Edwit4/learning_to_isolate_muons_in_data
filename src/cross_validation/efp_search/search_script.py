import os
import lmdb
import pickle
import argparse
import subprocess
import torch # needs to be imported before np even though it's only used in other script
import numpy as np
import pytorch_lightning as pl
from average_decision_ordering import calc_do

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('kfold_index', metavar='f', type=int, nargs=1,
        help='Index of cross validation fold')
    parser.add_argument('train_size', metavar='s', type=float, nargs=1,
        help='Proportion of total available training set to use')
    parser.add_argument('num_cones', metavar='c', type=int, nargs=1,
        help='Number of cones to use')
    parser.add_argument('num_gpus', metavar='g', type=int, nargs=1,
        help='Number of gpus to use')
    args = parser.parse_args()

    kfold_index = args.kfold_index[0] # Index of fold to evaluate
    train_size = args.train_size[0]
    num_cones = args.num_cones[0]
    num_gpus = args.num_gpus[0]

    seed = 123 
    rng = np.random.default_rng(seed=seed)
    benchmark = 0.902
    num_pairs = 50000
    max_iterations = 20

    # Set up
    pl.seed_everything(seed)
    kappa = [-1.0,0.0,0.25,0.5,1.0,2.0]
    beta = [0.25,0.5,1.0,2.0,3.0,4.0]
    kappa_betas = np.array([[[k,b] for k in kappa] for b in beta])
    kappa_betas = kappa_betas.reshape((kappa_betas.shape[0]*kappa_betas.shape[1],
        kappa_betas.shape[2]))
    efp_paths = [f'src/data/cms_efp_k{k}_b{b}' for k,b in kappa_betas]
    efp_envs = [lmdb.open(p,readonly=True,readahead=False,meminit=False,lock=False) 
        for p in efp_paths]
    efp_txns = [env.begin(buffers=True) for env in efp_envs]
    efp_cursors = [txn.cursor() for txn in efp_txns]
    num_graphs = len(np.frombuffer(efp_txns[0].get(str(0).encode())))
    pfn_version_dir = f'src/training/pfn/lightning_logs/trainsize_1.0_fold_{kfold_index}'

    pfn_test_idx = []
    pfn_test_labels = []
    pfn_test_outs = []
    pfn_test_masses= []
    pfn_gpus = np.unique([f.split(':')[1][0] 
        for f in os.listdir(pfn_version_dir) if ':' in f]).astype(np.int32)
    for j in pfn_gpus:
        pfn_test_idx.append(np.load(f'{pfn_version_dir}/test_idx_cuda:{j}.npy'))
        pfn_test_labels.append(np.load(f'{pfn_version_dir}/test_labels_cuda:{j}.npy'))
        pfn_test_outs.append(np.load(f'{pfn_version_dir}/test_outs_cuda:{j}.npy'))
        pfn_test_masses.append(np.load(f'{pfn_version_dir}/test_masses_cuda:{j}.npy'))
    pfn_test_idx = np.concatenate(pfn_test_idx).flatten()
    pfn_test_idx, unique_filter = np.unique(pfn_test_idx,return_index=True)
    pfn_test_idx = np.int32(pfn_test_idx)
    pfn_test_labels = np.concatenate(pfn_test_labels).flatten()[unique_filter]
    pfn_test_outs = np.concatenate(pfn_test_outs).flatten()[unique_filter]
    pfn_test_masses = np.concatenate(pfn_test_masses).flatten()[unique_filter]
    test_pos_idx = pfn_test_idx[pfn_test_labels==1]
    test_neg_idx = pfn_test_idx[pfn_test_labels==0]
    pfn_pos_outs = pfn_test_outs[pfn_test_labels==1]
    pfn_neg_outs = pfn_test_outs[pfn_test_labels==0]
    num_events = len(pfn_test_labels)

    # Load EFPs for test events
    efps_pos = np.empty((len(test_pos_idx), len(efp_cursors), num_graphs))
    efps_neg = np.empty((len(test_neg_idx), len(efp_cursors), num_graphs))

    for i, crsr in enumerate(efp_cursors):
        for j,idx in enumerate(test_pos_idx):
            if j%1024==0:
                print(f'cursor {i} pos event {j}        \r',end='')
            efps_pos[j,i,:] = np.frombuffer(crsr.get(str(idx).encode()))
        for j,idx in enumerate(test_neg_idx):
            if j%1024==0:
                print(f'cursor {i} neg event {j}        \r',end='')
            efps_neg[j,i,:] = np.frombuffer(crsr.get(str(idx).encode()))

    max_auc = 0.5
    selected_kappa_betas = []
    selected_graphs = []
    for iteration in range(max_iterations):
        if max_auc >= benchmark:
            break
        max_ado, max_kappa, max_beta, max_graph = 0.5, None, None, None

        version = f'numcones_{num_cones}_trainsize_{train_size}_fold_{kfold_index}_iteration_{iteration}'
        experiment_dir = 'src/training/efp_search'
        version_dir = f'{experiment_dir}/lightning_logs/{version}'

        print('')
        print(f'Currently Selected kappa, betas: {selected_kappa_betas}')
        print(f'Currently Selected graphs: {selected_graphs}')
        print('')

        # Train network on current high level selection
        # Done through subprocess as ddp just runs the entire script for every gpu,
        # but we don't want the search itself to run for every gpu
        cmd = ['python', 'src/training/efp_search/network_trainer.py', 
            str(kfold_index), str(train_size), str(num_cones), str(num_gpus), str(iteration)]
        subprocess.call(cmd)

        # Generate index pairs, randomly pairing things from our positive and negative samples
        num_pairs_chosen = 0
        rand_pairs = np.empty((num_pairs,2),dtype=np.int32)
        while num_pairs_chosen < num_pairs:
            generated_pairs = np.array([rng.choice(np.arange(len(test_neg_idx)),size=num_pairs),
                rng.choice(np.arange(len(test_pos_idx)),size=num_pairs)]).T
            unique_pairs = np.unique(generated_pairs,axis=0)
            if (num_pairs_chosen + len(unique_pairs)) > num_pairs:
                unique_pairs = unique_pairs[:num_pairs-len(unique_pairs)]
            rand_pairs[num_pairs_chosen:num_pairs_chosen+len(unique_pairs)] = unique_pairs
            num_pairs_chosen = num_pairs_chosen + len(unique_pairs)
        idx_pairs = np.int32(np.stack([test_neg_idx[rand_pairs[:,0]],
            test_pos_idx[rand_pairs[:,1]]]).T)
        pfn_out_pairs = np.stack([pfn_neg_outs[rand_pairs[:,0]],
            pfn_pos_outs[rand_pairs[:,1]]]).T

        # Load logged test info for above network
        hln_idx = []
        hln_outs = []
        hln_labels = []
        for i in range(num_gpus):
            hln_idx.append(np.load(f'{version_dir}/test_idx_cuda:{i}.npy'))
            hln_outs.append(np.load(f'{version_dir}/test_outs_cuda:{i}.npy'))
            hln_labels.append(np.load(f'{version_dir}/test_labels_cuda:{i}.npy'))
        hln_idx = np.concatenate(hln_idx)
        hln_idx, unique_filter = np.unique(hln_idx,return_index=True)
        hln_outs = np.concatenate(hln_outs)[unique_filter]
        hln_labels = np.concatenate(hln_labels)[unique_filter]

        # Make sure these are sorted same way as previously created pairs
        hln_out_pairs = np.empty(idx_pairs.shape)
        for i,pair_idx in enumerate(idx_pairs):
            hln_out_pairs[i][0] = hln_outs[hln_idx==pair_idx[0]]
            hln_out_pairs[i][1] = hln_outs[hln_idx==pair_idx[1]]

        # Calculate decision ordering comparing above network and PFN, grab differently orderd pairs
        print('')
        print('Calculating hln / pfn decision ordering')
        print('')
        hln_pfn_decision_ordering = calc_do(fx0=hln_out_pairs[:,0], fx1=hln_out_pairs[:,1],
            gx0=pfn_out_pairs[:,0], gx1=pfn_out_pairs[:,1])
        hln_pfn_inverse_decisions = np.argwhere(hln_pfn_decision_ordering==0).squeeze()
        
        neg_inverse_pfn_outs = pfn_out_pairs[:,0][hln_pfn_inverse_decisions]
        pos_inverse_pfn_outs = pfn_out_pairs[:,1][hln_pfn_inverse_decisions]
        inverse_rand_pairs = rand_pairs[hln_pfn_inverse_decisions]
        neg_inverse_efps = efps_neg[inverse_rand_pairs[:,0]]
        pos_inverse_efps = efps_pos[inverse_rand_pairs[:,1]]

        # Loop over EFPs
        for i,kb in enumerate(kappa_betas):
            for g in np.arange(num_graphs):
                efp_1d_index = i*num_graphs + g
                print(f'Checking EFP {efp_1d_index}/{len(kappa_betas)*num_graphs}\r',end='')

                # Check if EFP has already been chosen
                if len(selected_kappa_betas) > 0:
                    kb_idx = np.argwhere(
                        np.prod(np.array(selected_kappa_betas)==kb,axis=1).astype(bool))
                    if len(kb_idx) > 0:
                        if g in np.array(selected_graphs[kb_idx[0][0]]):
                            continue

                # Compare decision ordering of given EFP and PFN
                efp_pfn_decision_ordering = calc_do(
                    fx0=neg_inverse_efps[:,i,g],
                    fx1=pos_inverse_efps[:,i,g],
                    gx0=neg_inverse_pfn_outs,
                    gx1=pos_inverse_pfn_outs)
                
                # Evaluate ADO recording EFP this is highest ADO so far
                ado = np.mean(efp_pfn_decision_ordering)
                if ado < 0.5: ado = 1. - ado
                if ado > max_ado:
                    max_ado = ado
                    max_kappa, max_beta = kb
                    max_graph = g

        # Store the optimal EFP from this iteration
        print('')
        print(f'Optimal EFP - Kappa {max_kappa} Beta {max_beta} Graph {max_graph}')
        print(f'ADO: {max_ado}')
        print('')

        # Save optimal kappa / beta / graph
        kb_not_yet_stored = True
        if len(selected_kappa_betas) > 0:
            kb_idx = np.argwhere(np.prod(np.array(selected_kappa_betas)==[max_kappa,max_beta],
                axis=1).astype(bool))
            if len(kb_idx) > 0:
                selected_graphs[kb_idx[0][0]].append(max_graph)
                kb_not_yet_stored = False
        if kb_not_yet_stored:
            selected_kappa_betas.append([max_kappa, max_beta])
            selected_graphs.append([max_graph])

        with open(f'{version_dir}/kappa_betas.pickle','wb') as handle:
            pickle.dump(selected_kappa_betas, handle)
        with open(f'{version_dir}/graphs.pickle','wb') as handle:
            pickle.dump(selected_graphs, handle)

    print('')
    print(f'EFPs selected:')
    for kb, g in zip(selected_kappa_betas, selected_graphs):
        print(f'kappa, beta: {kb}')
        print(f'graphs: {g}')
    print('')