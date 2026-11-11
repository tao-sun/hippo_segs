"""Compare frozen logs. Plot arithmetic region means at each epoch, with no smoothing."""
from pathlib import Path
from datetime import datetime, timezone
import ast,csv,hashlib,json,re,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SNAP=OUT/'snapshots';SNAP.mkdir(exist_ok=True)
SCRATCH=Path('/gpfs/scratch1/shared/apiaghiardelli')
FILES={'smmlif':'logs24/SMMLIF_training.out','ds':'SMMLIF_training_DS_01_29077508.out','unet2d':'logs24/UNET2D_01_training.out','unet2d_results':'logs24/UNET2D_01_results_29118352.out'}
RUNS={'smmlif':'SNN_paper_unet_sel3_fold0','ds':'SNN_2D_Unet_SMMLif_DS_01','unet2d':'SNN_2D_Unet'}
LABEL={'smmlif':'SMMLIF (no DS)','ds':'SMMLIF + deep supervision','unet2d':'UNET2D (non-spiking, no DS)'}
COLOR={'smmlif':'#2864a5','ds':'#d96625','unet2d':'#23865c'}
REGIONS=['WT','TC','ET'];NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
provenance={}
def snapshot(source,name):
 target=SNAP/name
 if not target.exists():target.write_bytes(source.read_bytes())
 return target

def parse_epochs(text):
 rows=[];row=None;config=None
 for no,line in enumerate(text.splitlines(),1):
  if line.lstrip().startswith("{'dataset_name':"):config=ast.literal_eval(line.strip())['snn_config']
  m=re.search(r': Epoch (\d+)\s*$',line)
  if m:row={'epoch':int(m[1])+1,'epoch_index':int(m[1]),'start_line':no}
  if row is None:continue
  for label,key in [('train_loss','train_loss'),('val_loss','val_loss'),('Current learning rate','lr'),('Epoch time','seconds')]:
   m=re.search(re.escape(label)+r':?\s+('+NUM+')',line)
   if m:row[key]=float(m[1])
  if 'Pseudo dice [' in line:
   values=[float(v) for v in re.findall(r'np.float32\(([^)]+)\)',line)];assert len(values)==3
   row.update(dict(zip(REGIONS,values)));row['mean_dice']=st.mean(values);row['metric_line']=no
  if 'Epoch time:' in line:
   assert all(k in row for k in ['mean_dice','val_loss','train_loss','lr','seconds'])
   rows.append(row.copy())
 assert config and [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
 return rows,config

def parse_cases(text):
 cases={};case=None;total=None
 for line in text.splitlines():
  m=re.match(r'^\[(\d+)/(\d+)\]\s+(\S+)',line)
  if m:case=m[3];total=int(m[2])
  m=re.search(r'Dice (raw|postprocessed): WT=('+NUM+r') TC=('+NUM+r') ET=('+NUM+r') mean=('+NUM+r')',line)
  if m and case:
   cases.setdefault(case,{})[m[1]]=dict(zip([*REGIONS,'printed_mean'],map(float,m.groups()[1:])))
 complete={k:v for k,v in cases.items() if 'raw' in v and 'postprocessed' in v}
 return complete,total

def average_rows(rows):
 return {k:st.mean(r[k] for r in rows) for k in [*REGIONS,'mean_dice','val_loss','train_loss']}

def average_cases(cases,ids):
 return {stage:{**{reg:st.mean(cases[i][stage][reg] for i in ids) for reg in REGIONS},'mean':st.mean(st.mean(cases[i][stage][reg] for reg in REGIONS) for i in ids)} for stage in ['raw','postprocessed']}

texts={};data={};configs={};debug={}
for key,name in FILES.items():
 f=snapshot(ROOT/name,Path(name).name);texts[key]=f.read_text()
 provenance[key]={'source':str(ROOT/name),'snapshot':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 if key in RUNS:
  data[key],configs[key]=parse_epochs(texts[key]);provenance[key]['completed_epochs']=len(data[key])
for key,run in RUNS.items():
 base=SCRATCH/run
 saved=json.loads(snapshot(base/'run_config.json',key+'_run_config.json').read_text())
 snapshot(base/'experiment.yaml',key+'_experiment.yaml')
 assert saved['snn_config']==configs[key]
 folder=base/'Dataset001_BraTS24GLI/nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres/fold_0'
 d=json.loads(snapshot(folder/'debug.json',key+'_debug.json').read_text())
 debug[key]={k:d.get(k) for k in ['num_iterations_per_epoch','num_val_iterations_per_epoch','initial_lr','num_epochs']}
common=min(map(len,data.values()))
summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'provenance':provenance,'common_completed_epochs':common,'epoch_convention':'1-based completed epoch; original Epoch indices are 0-based','epoch_mean_definition':'(WT + TC + ET) / 3 within the same epoch; no moving average or EMA','configs':configs,'debug':debug,'at_common_epoch':{k:v[common-1] for k,v in data.items()},'last':{k:v[-1] for k,v in data.items()},'last10_common_arithmetic_summary':{k:average_rows(v[common-10:common]) for k,v in data.items()}}
full={}
for key,run in RUNS.items():
 folder=SCRATCH/run/'Dataset001_BraTS24GLI/nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres/fold_0'
 for name,relative in [('epoch50','validation_periodic/epoch_0050/summary.json'),('epoch100','validation_periodic/epoch_0100/summary.json'),('epoch150','validation_periodic/epoch_0150/summary.json'),('best','validation_checkpoint_best_81x128x128_postprocessed/summary.json')]:
  source=folder/relative
  if source.exists():
   val=json.loads(snapshot(source,key+'_validation_'+name+'.json').read_text())
   assert len(val['cases'])==270
   full.setdefault(key,{})[name]=val
summary['full_volume']={k:{name:{field:value for field,value in v.items() if field!='cases'} for name,v in vals.items()} for k,vals in full.items()}
ds_cases,ds_total=parse_cases(texts['ds']);unet_cases,unet_total=parse_cases(texts['unet2d_results'])
old_cases={case['case']:{stage:{r:case[stage][r]['dice'] for r in REGIONS} for stage in ['raw','postprocessed']} for case in full['smmlif']['best']['cases']}
matched=sorted(set(ds_cases)&set(unet_cases)&set(old_cases))
summary['partial_validation']={'ds':{'completed_cases':len(ds_cases),'total':ds_total,'checkpoint':'epoch50'},'unet2d_results':{'completed_cases':len(unet_cases),'total':unet_total,'checkpoint':'checkpoint_best.pth'},'matched_cases':matched,'matched_case_count':len(matched),'matched_means':{k:average_cases(cases,matched) for k,cases in [('smmlif',old_cases),('ds',ds_cases),('unet2d',unet_cases)]} if matched else {},'caution':'Provisional; only completed cases shared by all three runs. Different checkpoint budgets. Not a full-fold ranking.'}
summary['mean_thresholds']={k:{str(t):next((r['epoch'] for r in v if r['mean_dice']>t),None) for t in [.7,.8]} for k,v in data.items()}
paired50={}
if matched:
 for k in ['smmlif','unet2d']:
  cases={c['case']:{stage:{reg:c[stage][reg]['dice'] for reg in REGIONS} for stage in ['raw','postprocessed']} for c in full[k]['epoch50']['cases']}
  paired50[k]=average_cases(cases,matched)
 paired50['ds']=average_cases(ds_cases,matched)
summary['partial_validation']['epoch50_matched_means']=paired50

with (OUT/'epochs.csv').open('w',newline='') as f:
 fields=['run','epoch','epoch_index',*REGIONS,'mean_dice','train_loss','val_loss','lr','seconds','start_line','metric_line'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for k,rows in data.items():w.writerows(dict(run=k,**row) for row in rows)
with (OUT/'matched_validation_cases.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['run','case','stage',*REGIONS,'mean_dice'])
 for k,cases in [('smmlif',old_cases),('ds',ds_cases),('unet2d',unet_cases)]:
  for case in matched:
   for stage in ['raw','postprocessed']:
    v=[cases[case][stage][r] for r in REGIONS];w.writerow([k,case,stage,*v,st.mean(v)])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
plotted={}
def draw(ax,metric,limit=None):
 for k,rows in data.items():
  rr=rows[:limit] if limit else rows;x=[r['epoch'] for r in rr];y=[r[metric] for r in rr]
  ax.plot(x,y,color=COLOR[k],label=LABEL[k],lw=1.5,marker='.',markersize=3)
  plotted.setdefault(metric,{})[k+'_common' if limit else k+'_all']={'x':x,'y':y}
 ax.set_xlabel('Completed epoch (1-based)');ax.set_ylabel('Pseudo-Dice');ax.set_ylim(0,1);ax.grid(alpha=.22)
fig,axes=plt.subplots(1,2,figsize=(15,5.6))
draw(axes[0],'mean_dice',common);axes[0].set_title(f'Matched training budget: epochs 1–{common}');axes[0].legend(fontsize=9,loc='lower right')
draw(axes[1],'mean_dice');axes[1].set_title('All available completed epochs')
axes[1].axvline(common,color='#555555',ls=':',lw=1)
fig.suptitle('Mean validation pseudo-Dice at each epoch',fontsize=17)
fig.text(.5,.91,'Arithmetic mean of WT, TC and ET within each epoch · No moving average or EMA',ha='center',fontsize=11)
fig.text(.5,.015,'Different initial learning rates and scheduler budgets: SMMLIF 0.001 / 200 epochs; SMMLIF + DS and UNET2D 0.01 / 400 epochs.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.88));fig.savefig(OUT/'mean_dice_by_epoch.png',dpi=180);fig.savefig(OUT/'mean_dice_by_epoch.pdf');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(13,9))
for ax,metric,title in zip(axes.flat,[*REGIONS,'mean_dice'],['Whole tumor (WT)','Tumor core (TC)','Enhancing tumor (ET)','Arithmetic mean: (WT + TC + ET) / 3']):
 draw(ax,metric,common);ax.set_title(title)
axes[0,0].legend(fontsize=9,loc='lower right')
fig.suptitle(f'Validation pseudo-Dice by epoch · Common epochs 1–{common}',fontsize=16)
fig.text(.5,.012,'Each point is the value for that epoch. Lines connect observed points only; no smoothing or EMA.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(OUT/'regional_dice_by_epoch.png',dpi=180);fig.savefig(OUT/'regional_dice_by_epoch.pdf');plt.close(fig)
if matched:
 fig,axes=plt.subplots(1,2,figsize=(14,5.6));x=np.arange(4)
 for ax,stage in zip(axes,['raw','postprocessed']):
  for offset,(k,means) in zip([-.25,0,.25],summary['partial_validation']['matched_means'].items()):
   vals=[means[stage][reg] for reg in [*REGIONS,'mean']];bars=ax.bar(x+offset,vals,width=.24,color=COLOR[k],label=LABEL[k]);ax.bar_label(bars,fmt='%.3f',padding=2,fontsize=8)
  ax.set_xticks(x,[*REGIONS,'Mean']);ax.set_ylim(0,1.08);ax.set_ylabel('Mean full-volume Dice');ax.set_title('Raw predictions' if stage=='raw' else 'After postprocessing');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].legend(fontsize=8,loc='lower right')
 fig.suptitle(f'Provisional validation: {len(matched)} matched completed cases out of 270',fontsize=15)
 fig.text(.5,.015,'Same cases, different checkpoint budgets: SMMLIF best / SMMLIF + DS epoch 50 / UNET2D best. Incomplete cohort; not a final ranking.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.045,1,.94));fig.savefig(OUT/'partial_matched_validation.png',dpi=180);plt.close(fig)
(OUT/'plot_data.json').write_text(json.dumps(plotted,indent=2)+'\n')
print(json.dumps({'epochs':{k:len(v) for k,v in data.items()},'at_common_epoch':summary['at_common_epoch'],'last10_common':summary['last10_common_arithmetic_summary'],'partial_validation':summary['partial_validation']},indent=2))
