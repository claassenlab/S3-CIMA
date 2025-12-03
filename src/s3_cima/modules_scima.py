"""
last edit on The Nov  3 19:12:31 2022

@author: sepidehbabaei
"""
"""
change the directory to the scima folder
cd Desktop/CrC/scima

Call the function from the commandline like this:

python3 spatialinput.py --nn 200 --cnt 13
"""

import numpy as np
import os, sys, errno
from datetime import datetime
import pickle

from model import CellCnn
from utils import  mkdir_p


def spatial_input_per_sample(Anchor,image, pat, intensity, ct, x, y, cellid, K, label, OUTDIR):
    
    #pat0 = np.unique(pat)
    image0 = np.unique(image)
    np.random.seed(12345)

    Totintens = np.empty((0, intensity.shape[1]))
    ranTotintens = np.empty((0, intensity.shape[1]))
    ranTotobj , Totobj =[] , []
    pid = []
    nset = []

    for i in image0: #pat0:       
        #inx = np.where(pat == i)[0]
        inx = np.where(image == i)[0]
        ipat= np.unique(pat[inx])
        #pdata = subdata[inx,:]
        px = x[inx]  
        py = y[inx]  
        pcellid = cellid[inx]  
        pint = intensity[inx]
        pct = ct[inx]

        inx = np.where(pct == Anchor)[0]
        #rpdata = np.delete(pdata, inx, 0)
        icell = np.arange(len(pct))
        raninx = np.delete(icell, inx, 0)
        raninx = np.random.choice(raninx, len(inx), replace= False)
               
        nset.append(len(inx))
        #pid = []
        for ianchor in inx:
            Totintens, Totobj = make_KNN_anchor(ianchor,px, py, K, pint, pcellid, Totintens, Totobj)
            pid = pid + np.repeat(ipat,K).tolist()
    
        for ianchor in raninx:
            ranTotintens, ranTotobj = make_KNN_anchor(ianchor,px, py, K, pint, pcellid, ranTotintens, ranTotobj)


    d = Totintens
    s = [pid, Totobj]

    rd = ranTotintens
    rs = [pid, ranTotobj]

    mkdir_p(OUTDIR)

    lab = str(label)
    pickle.dump(d, open(OUTDIR+'/d'+ lab +'.p','wb'))
    pickle.dump(rd, open(OUTDIR+'/drand'+ lab +'.p','wb'))

    pickle.dump(s, open(OUTDIR+'/s'+ lab +'.p','wb'))
    pickle.dump(rs, open(OUTDIR+'/srand'+ lab +'.p','wb'))

    pickle.dump(nset, open(OUTDIR+'/nset'+ lab +'.p','wb'))
    pickle.dump(nset, open(OUTDIR+'/nsetrand'+ lab +'.p','wb'))

    return d,s,rd,rs, nset



def make_KNN_anchor(ianchor,px, py, K, pint, pcellid, Totintens, Totobj):
    dist= np.array(((px-px[ianchor])**2) + ((py -py[ianchor])**2),dtype=np.float32)
    dist = np.sqrt(dist)
    nnix=np.argsort(dist)[1:K+1]
    intens = pint[nnix,:]
    Totintens = np.concatenate((Totintens, intens), axis=0)
    obj = pcellid[nnix].tolist()
    Totobj = Totobj+obj
    
    return Totintens, Totobj



def BG_spatial_input_per_sample(N, image, pat, intensity, ct, x, y, cellid, K, label, OUTDIR):
    
    #pat = pat[inx]
    #intensity = Intensity[inx,:]
    #ct = ct[inx]
    #x =x[inx]
    #y= y[inx]
    #cellid = cellid[inx]
    #label = labels[i]
    #image = image[inx]
    
    #pat0 = np.unique(pat)
    image0 = np.unique(image)
    np.random.seed(12345)

    ranTotintens = np.empty((0, intensity.shape[1]))
    ranTotobj =[]
    pid = []
    nset = []

    for i in image0: #pat0:       
        #inx = np.where(pat == i)[0]
        inx = np.where(image == i)[0]
        ipat= np.unique(pat[inx])

        px = x[inx]  
        py = y[inx]  
        pcellid = cellid[inx]  
        pint = intensity[inx]
        pct = ct[inx]

        raninx = np.random.choice(len(pct), N, replace= False)
               
        nset.append(len(raninx))
        #pid = []
        for ianchor in raninx:
            ranTotintens, ranTotobj = make_KNN_anchor(ianchor,px, py, K, pint, pcellid, ranTotintens, ranTotobj)
            pid = pid + np.repeat(ipat,K).tolist()


    rd = ranTotintens
    rs = [pid, ranTotobj]

    mkdir_p(OUTDIR)

    lab = str(label)
    pickle.dump(rd, open(OUTDIR+'/d'+ lab +'.p','wb'))

    pickle.dump(rs, open(OUTDIR+'/s'+ lab +'.p','wb'))

    pickle.dump(nset, open(OUTDIR+'/nset'+ lab +'.p','wb'))
    

    return rd,rs, nset


def make_inputset(Sample_ID, Cell_ID, IDx, Dx, K, k ):
    
    #Cell_ID = cell_id[lab]
    #IDx = idx
    #Dx = d[lab]
    #Sample_ID = sample_id[lab]
    
    group_list = []
    cell =[]
    
    inx = np.where(Sample_ID == IDx)[0]
    x= Dx[inx,]
    xcell = Cell_ID[inx]
    nset = int(len(inx)/K)
    for cnt in range(1, nset+1):
        start= (cnt-1) * K
        end = start + k #(cnt*nn)
        xset =x[start:end,]
        xcell_set=xcell[start:end,]
        group_list.append(xset)
        cell.append(xcell_set)
    pat_id = ([IDx]*nset)  
    return group_list,cell, pat_id, nset


def run_scima(Anchor, ntrain_per_class, K, k, nset_thr, labels, classes, path, nrun, background):


    ncell= k
    ntime_points=1
    nsubsets=1
    per_sample = True
    #transform = False
    scale= True
    qt_norm= True
    
    
    now = datetime.now()
    result_file = 'results'+str(Anchor)
    
    current_time = now.strftime("%H_%M_%S")
    old_stdout = sys.stdout
    
    INDIR = path + '/Anchor' + str(Anchor)
    
    
    try:
        log_dir = path+'/'+result_file+'/run_log/'
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    log_file = open(log_dir+"/data_run_log_"+str(current_time)+'.txt',"w+")
    print('output will be rdirected to',log_dir+"/data_run_log_"+str(current_time)+'.txt')
    sys.stdout = log_file
    
    OUTDIR = os.path.join(path, result_file, 'out_'+str(current_time))
    mkdir_p(OUTDIR)
    print('output directory', OUTDIR)
    
         
          
    d = dict()
    s = dict()
    nset = dict()
    
    for i in range(0, len(labels)):
        lab = str(labels[i])
        d[lab] = pickle.load(open(INDIR+'/d'+ lab +'.p','rb'))
        s[lab] = pickle.load(open(INDIR+'/s'+ lab +'.p','rb'))
        nset[lab] = pickle.load(open(INDIR+'/nset'+ lab +'.p','rb'))
    
    sample_id = dict()
    cell_id = dict()
    group = dict()
    for i in range(0, len(labels)):
        lab = str(labels[i])
        sample_id[lab] = np.asarray(s[lab][0])
        cell_id[lab] = np.asarray(s[lab][1])
        group[lab] = np.unique(np.asarray(s[lab][0]))
    
    
    
    np.random.seed(12345)
    
    train_idx = dict()
    test_idx = dict()
    
    if background:
        for i in range(0, int(len(labels)/2)):
            lab = str(labels[i])
            train_idx[lab] = list(np.random.choice(group[lab], size=ntrain_per_class, replace=False))
            test_idx[lab] = [j for j in group[lab] if j not in train_idx[lab]]
            rlab = str(labels[i+ int(len(labels)/2)])
            train_idx[rlab] = train_idx[lab]
            test_idx[rlab] = test_idx[lab]
    else:
        for i in range(0, len(labels)):
            lab = str(labels[i])
            train_idx[lab] = list(np.random.choice(group[lab], size=ntrain_per_class, replace=False))
            test_idx[lab] = [j for j in group[lab] if j not in train_idx[lab]]

    
    
    train_group_list = dict()
    train_pat_id= dict()
    train_cell = dict()
    train_nset = dict()
        
    for i in range(0, len(labels)):
        lab = str(labels[i])
        group_list = []
        pat_id=[]
        cell =[]
        nset =[]
    
        for idx in train_idx[lab]:
            group_list0,cell0, pat_id0, nset0 = make_inputset(sample_id[lab], cell_id[lab], idx, d[lab], K, k)
            group_list += group_list0
            cell += cell0
            pat_id += pat_id0  
            nset.append(nset0)
            train_group_list[lab] = group_list
            train_pat_id[lab] = pat_id
            train_cell[lab] = cell
            train_nset[lab] = nset
    
    test_group_list = dict()
    test_pat_id= dict()
    test_cell = dict()
    test_nset = dict()
        
    for i in range(0, len(labels)):
        lab = str(labels[i])
        group_list = []
        pat_id=[]
        cell =[]
        nset =[]
    
        for idx in test_idx[lab]:
            group_list0,cell0, pat_id0, nset0 = make_inputset(sample_id[lab], cell_id[lab], idx, d[lab], K, k)
            group_list += group_list0
            cell += cell0
            pat_id += pat_id0  
            nset.append(nset0)
            test_group_list[lab] = group_list
            test_pat_id[lab] = pat_id
            test_cell[lab] = cell
            test_nset[lab] = nset
    
    
    # finally prepare training and vallidation data
    mkdir_p(OUTDIR+'/model/')
    
    
    traincell = []
    trainphenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        traincell = traincell + (train_cell[lab])
        trainphenotypes = trainphenotypes + [classes[i]] * k *len(train_cell[lab])
    
    testcell = []
    testphenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        testcell = testcell + (test_cell[lab])
        testphenotypes = testphenotypes + [classes[i]] * k *len(test_cell[lab])
    
    
    pickle.dump(traincell, open(OUTDIR+'/model/train_cell_all.p','wb'))
    pickle.dump(testcell, open(OUTDIR+'/model/test_cell_all.p','wb'))
    
    pickle.dump(trainphenotypes, open(OUTDIR+'/model/train_phen_all.p','wb'))
    pickle.dump(testphenotypes, open(OUTDIR+'/model/test_phen_all.p','wb'))
    
    
    for i in labels:
        inx =np.random.choice(a=len(train_pat_id[i]), size=len(train_pat_id[i]), replace=False)
        group_list = (train_group_list[i])
        train_group_list[i] = [group_list[j] for j in inx]
        train_pat_id[i] = np.array(train_pat_id[i])[inx]
    
    for i in labels:
        inx =np.random.choice(a=len(test_pat_id[i]), size=len(test_pat_id[i]), replace=False)
        group_list = (test_group_list[i])
        test_group_list[i] = [group_list[j] for j in inx]
        test_pat_id[i] = np.array(test_pat_id[i])[inx]
    
    
    trnset = []
    tsnset = []
    for i in labels:
        trnset += train_nset[i]
        tsnset += test_nset[i]
    
    nsetmed = int(np.quantile(trnset + tsnset , nset_thr))
    
    g = dict()
    for i in labels:
        g[i] = []
        for idx in train_idx[i]:
            inx = np.where(train_pat_id[i]==idx)[0]
            data = (train_group_list[i])
            data = [data[j] for j in inx]
            if len(inx) > nsetmed:
                data = data[:nsetmed]
            g[i]= g[i] + data 
        
    tg = dict()
    for i in labels:
        tg[i] = []
        for idx in test_idx[i]:
            inx = np.where(test_pat_id[i]==idx)[0]
            data = (test_group_list[i])
            data = [data[j] for j in inx]
            if len(inx) > nsetmed:
                data = data[:nsetmed]
            tg[i]= tg[i] + data 
         
        
    specify_valid = True
    
    print('saving samples...')
    
    traincell = []
    trainphenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        traincell = traincell + (train_cell[lab])
        trainphenotypes = trainphenotypes + [classes[i]] * k *len(train_cell[lab])
    
    
    train_samples = []
    train_phenotypes = []
    valid_samples = []
    valid_phenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        g0 = g[lab]
        cut = int(.8 * len(g0))
        
        train_samples += g0[:cut]
        train_phenotypes += [classes[i]] * len(g0[:cut])
        
        valid_samples += g0[cut:] 
        valid_phenotypes += [classes[i]] * len(g0[cut:])
    
    print('training and validation phenotypes', train_phenotypes, valid_phenotypes)
    pickle.dump(valid_samples, open(OUTDIR+'/model/valid_samples.p','wb'))
    pickle.dump(valid_phenotypes, open(OUTDIR+'/model/valid_phenotypes.p','wb')) 
    
    pickle.dump(train_samples, open(OUTDIR+'/model/train_samples.p','wb'))
    pickle.dump(train_phenotypes, open(OUTDIR+'/model/train_phenotypes.p','wb'))
     
    test_samples = []
    test_phenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        g0 = tg[lab]
        
        test_samples += g0
        test_phenotypes += [classes[i]] * len(g0)
    
    pickle.dump(test_samples, open(OUTDIR+'/model/test_samples.p','wb'))
    pickle.dump(test_phenotypes, open(OUTDIR+'/model/test_phenotypes.p','wb'))
    
    
    print('ntime_points', ntime_points)
    print('ncell', ncell)
    print('nsubsets', nsubsets)
    print('nrun', nrun)
    print('quant_normed',qt_norm)
    print('scale',scale)
    
    
    model = CellCnn(ntime_points=ntime_points, ncell=ncell, nsubset=nsubsets, 
                                 nrun=nrun, scale=scale, quant_normed=qt_norm, verbose=0, 
                                 per_sample = per_sample)
    
    train_sample = dict()
    valid_sample = dict()
    
    for nt in range(1, ntime_points+1):
        train_sample[str(nt)] = train_samples
        if specify_valid:
           valid_sample[str(nt)] = valid_samples
    
    
    if specify_valid:
       model.fit(ntime_points, train_samples=train_sample, train_phenotypes=train_phenotypes,
              valid_samples=valid_sample, valid_phenotypes=valid_phenotypes,outdir=OUTDIR)
    else:
       model.fit(ntime_points, train_samples=train_sample, train_phenotypes=train_phenotypes,outdir=OUTDIR)
    
    pickle.dump(model.results, open(OUTDIR+'/model/results.p','wb'))
    test_pred = model.predict(test_samples)
    
    pickle.dump(test_pred, open(OUTDIR+'/model/test_pred.p','wb'))
    
    
    from sklearn.metrics import accuracy_score
    test_phenotypespre=np.argmax(test_pred, axis=1).astype(float)
    accuracy_score = accuracy_score(test_phenotypes,test_phenotypespre)
    #accuracy_score = accuracy_score(test_phenotypes,np.around(test_pred[:,1]))
    
    print('Accuracy score: {0:0.2f}'.format(accuracy_score))
    print('Anchor:')
    print(Anchor)
    print('nset:')
    print(trnset )
    print(tsnset)
    print('nsetmed:')
    print(nsetmed)
    
    print(train_idx)
    print(test_idx)
    
    
    
    log_file.close()
    sys.stdout = old_stdout
    
    print('Accuracy score: {0:0.2f}'.format(accuracy_score))
    print('Anchor:')
    print(Anchor)







