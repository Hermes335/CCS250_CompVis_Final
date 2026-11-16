"""Versioned, reproducible training and evaluation helpers for iMAGE."""
from __future__ import annotations
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm
from .data import DenoiseDataset
from .models import ROOT, NAMES, create_model, load_checkpoint
from .online import get_active_checkpoint
from .inference import tiled_forward


def setup_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize()
    elif device.type == 'mps': torch.mps.synchronize()


def dataset_counts(manifest=ROOT/'data/splits.csv'):
    with Path(manifest).open(newline='',encoding='utf8') as f: rows=list(csv.DictReader(f))
    required={'split','noisy_path','clean_path'}
    source_key='dataset' if rows and 'dataset' in rows[0] else 'source'
    if not rows or not required.issubset(rows[0]) or source_key not in rows[0]: raise ValueError('Manifest needs dataset/source, split, noisy_path and clean_path columns')
    expected={('SIDD','train'):2800,('SIDD','val'):350,('SIDD','test'):350,('PolyU','train'):1200,('PolyU','val'):150,('PolyU','test'):150}
    actual={k:0 for k in expected}; seen=set()
    for row in rows:
        key=(row[source_key],row['split'])
        if key not in actual: raise ValueError(f'Unexpected dataset/split: {key}')
        actual[key]+=1
        pair=(row[source_key],row['noisy_path'],row['clean_path'])
        if pair in seen: raise ValueError(f'Duplicate pair: {pair}')
        seen.add(pair)
    if actual!=expected: raise ValueError(f'Expected SIDD/PolyU quotas {expected}, found {actual}')
    return {s:sum(actual[(d,s)] for d in ('SIDD','PolyU')) for s in ('train','val','test')}


def get_network(name,device=None,checkpoint=None):
    if name not in NAMES: raise ValueError(f'Unknown model: {name}')
    device=torch.device(device) if device else setup_device()
    net=create_model(name); source=Path(checkpoint) if checkpoint else get_active_checkpoint(name)
    load_checkpoint(net,name,source)
    return net.to(device),source


def validation_psnr(net,dataset,device,limit=500):
    net.eval();scores=[]
    with torch.inference_mode():
        for i,(noisy,clean) in enumerate(DataLoader(dataset,batch_size=1,shuffle=False)):
            if limit and i>=limit: break
            noisy,clean=noisy.to(device),clean.to(device);pred=net(noisy).clamp(0,1)
            mse=((pred-clean)**2).mean().clamp(min=1e-12);scores.append(float(-10*torch.log10(mse)))
    if not scores: raise ValueError('Validation split is empty')
    return float(np.mean(scores))


def fine_tune(name,*,output_dir,checkpoint=None,epochs=5,batch_size=1,patch=128,sigma=25,lr=1e-5,validation_limit=500,max_train_steps=0,seed=42,patience=3):
    """Fine-tune a compatible checkpoint and isolate outputs in output_dir."""
    if name not in NAMES: raise ValueError(f'Unknown model: {name}')
    if epochs<1 or batch_size<1 or patch<8: raise ValueError('Invalid training configuration')
    counts=dataset_counts(); device=setup_device();torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    output=Path(output_dir).expanduser().resolve(); ckpt_dir=output/'checkpoints';log_dir=output/'logs'
    ckpt_dir.mkdir(parents=True,exist_ok=True);log_dir.mkdir(parents=True,exist_ok=True)
    train=DenoiseDataset(ROOT/'data/splits.csv','train',patch=patch,sigma=sigma,seed=seed)
    val=DenoiseDataset(ROOT/'data/splits.csv','val',patch=patch,sigma=sigma,seed=seed)
    loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    net,source=get_network(name,device,checkpoint);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    print(f'{name}: {device}; initialized from {source}; {counts}')
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history=[];best=-float('inf');stale=0;started=time.perf_counter();destination=ckpt_dir/f'{name}.pth'
    for epoch in range(epochs):
        train.epoch=epoch;net.train();losses=[]
        for step,(noisy,clean) in enumerate(tqdm(loader,desc=f'{name} {epoch+1}/{epochs}')):
            noisy,clean=noisy.to(device),clean.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=device.type=='cuda'): pred=net(noisy);loss=nn.functional.l1_loss(pred,clean)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);scaler.step(optimizer);scaler.update();losses.append(float(loss.item()))
            if max_train_steps and step+1>=max_train_steps: break
        score=validation_psnr(net,val,device,validation_limit);elapsed=(time.perf_counter()-started)/60
        row={'epoch':epoch+1,'train_l1':float(np.mean(losses)),'validation_psnr_db':score,'elapsed_min':round(elapsed,3)};history.append(row);print(row)
        if score>best+1e-4:
            best=score;stale=0
            torch.save({'state_dict':net.state_dict(),'meta':{'name':name,'epoch':epoch+1,'validation_psnr':score,'seed':seed,'training_samples':counts['train'],'validation_samples':len(val),'source_checkpoint':str(source),'source_checkpoint_sha256':source_hash,'device':str(device),'patch':patch,'batch_size':batch_size,'learning_rate':lr,'epochs_requested':epochs}},destination)
        else:
            stale+=1
            if stale>=patience: print('Early stopping');break
    with (log_dir/f'{name}_training.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
    if device.type=='cuda':torch.cuda.empty_cache()
    print(f'Best validation PSNR: {best:.4f} dB; checkpoint: {destination}')
    return history


def evaluate_model(name,*,checkpoint,output_dir,patch=256,sigma=25,seed=42,limit=0,tile=256):
    if name not in NAMES: raise ValueError(f'Unknown model: {name}')
    dataset_counts();device=setup_device();ds=DenoiseDataset(ROOT/'data/splits.csv','test',patch=patch,sigma=sigma,seed=seed)
    net,source=get_network(name,device,checkpoint);net.eval();first=ds[0][0][None].to(device)
    with torch.inference_mode():
        for _ in range(2): tiled_forward(net,first,name,tile=tile)
    synchronize(device);rows=[]
    with torch.inference_mode():
        for i,(noisy,clean) in enumerate(tqdm(DataLoader(ds,batch_size=1,shuffle=False),desc=f'Evaluating {name}')):
            if limit and i>=limit:break
            noisy=noisy.to(device);synchronize(device);start=time.perf_counter();pred=tiled_forward(net,noisy,name,tile=tile);synchronize(device)
            gt=clean[0].permute(1,2,0).numpy();out=pred[0].permute(1,2,0).cpu().numpy()
            rows.append({'index':i,'psnr':float(peak_signal_noise_ratio(gt,out,data_range=1)),'ssim':float(structural_similarity(gt,out,data_range=1,channel_axis=2)),'runtime_ms':(time.perf_counter()-start)*1000})
    if not rows:raise ValueError('No test images evaluated')
    report=Path(output_dir).expanduser().resolve()/'evaluation';report.mkdir(parents=True,exist_ok=True)
    with (report/f'{name}_test.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'model':name,'checkpoint':str(source),'checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((ROOT/'data/splits.csv').read_bytes()).hexdigest(),'split':'test','number_of_test_images':len(rows),'limited_test':bool(limit),'patch':patch,'seed':seed,'tile':tile,'device':str(device),'psnr_db':round(float(np.mean([r['psnr'] for r in rows])),4),'ssim':round(float(np.mean([r['ssim'] for r in rows])),5),'mean_runtime_ms':round(float(np.mean([r['runtime_ms'] for r in rows])),2),'parameters_m':round(sum(p.numel() for p in net.parameters())/1e6,3)}
    (report/f'{name}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
    if device.type=='cuda':torch.cuda.empty_cache()
    return summary
