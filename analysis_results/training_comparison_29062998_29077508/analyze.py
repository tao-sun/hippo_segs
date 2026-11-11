"""Reproduce the comparison using frozen input snapshots; no model loading."""
from pathlib import Path
import ast, csv, hashlib, json, re, statistics as st
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN_ROOT=Path('/gpfs/scratch1/shared/apiaghiardelli')
SNAP=OUT/'snapshots'
SNAP.mkdir(exist_ok=True)
RUNS={'full':('slurm_snn_brats24_f0_train_val_29062998.out','SNN_2D_Unet_SMMLif_01'), 'native':('SMMLIF_training_DS_01_29077508.out','SNN_2D_Unet_SMMLif_DS_01')}
LABELS={'full':'DS full [1, 0.3, 0.1] · A100','native':'DS native [1, 0.5, 0.25] · H100'}
COLORS={'full':'#2764a5','native':'#dc682e'}
REGIONS=['WT','TC','ET']
def snapshot(source,name):
 p=SNAP/name
 if not p.exists(): p.write_bytes(source.read_bytes())
 return p

def parse(text):
 rows=[]; row=None; config=None
 for no,line in enumerate(text.splitlines(),1):
  if line.lstrip().startswith("{'dataset_name':"): config=ast.literal_eval(line.strip())['snn_config']
  m=re.search(r': Epoch (\d+)\s*$',line)
  if m: row={'epoch_index':int(m[1]),'epoch':int(m[1])+1,'start_line':no}
  if row is None: continue
  for label,key in [('train_loss','train_loss'),('val_loss','val_loss'),('Current learning rate','lr'),('Epoch time','seconds')]:
   m=re.search(re.escape(label)+r':?\s+(-?[\d.]+)',line)
   if m: row[key]=float(m[1])
  if 'Pseudo dice' in line:
   values=[float(x) for x in re.findall(r'np.float32\(([^)]+)\)',line)]
   assert len(values)==3
   row.update(dict(zip(REGIONS,values))); row['mean_dice']=st.mean(values);row['metric_line']=no
   row['timestamp']=line.split(': Pseudo dice')[0]
  if 'Recurrent neuron diagnostics:' in line: row['diagnostics']=ast.literal_eval(line.split('Recurrent neuron diagnostics:',1)[1].strip())
  if 'Epoch time:' in line:
   assert all(k in row for k in ['train_loss','val_loss','mean_dice','seconds'])
   rows.append(row.copy())
 assert config and rows and [r['epoch_index'] for r in rows]==list(range(len(rows)))
 return rows,config

def averages(rows):
 return {k:st.mean(r[k] for r in rows) for k in [*REGIONS,'mean_dice','train_loss','val_loss','seconds']}
def diff(a,b,p=''):
 if isinstance(a,dict) and isinstance(b,dict):
  return [d for k in sorted(a.keys()|b.keys()) for d in diff(a.get(k),b.get(k),p+'.'+k)]
 return [] if a==b else [{'path':p,'full':a,'native':b}]

data={};configs={};saved={};provenance={}
for key,(filename,run) in RUNS.items():
 p=snapshot(ROOT/filename,filename);data[key],configs[key]=parse(p.read_text())
 saved[key]=json.loads(snapshot(RUN_ROOT/run/'run_config.json',key+'_run_config.json').read_text())
 snapshot(RUN_ROOT/run/'experiment.yaml',key+'_experiment.yaml')
 assert configs[key]==saved[key]['snn_config']
 provenance[key]={'log':str(ROOT/filename),'snapshot_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'completed_epochs':len(data[key]),'last_metric_timestamp_log_timezone_unspecified':data[key][-1]['timestamp']}
common=min(map(len,data.values()));assert common>=25
summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'epoch_convention':'Report uses 1-based completed epochs; original Epoch index is 0-based.','provenance':provenance,'common_completed_epochs':common,'snn_config_differences':diff(configs['full'],configs['native']),'saved_config_differences':diff(saved['full'],saved['native']),'first25':{k:averages(v[:25]) for k,v in data.items()},'epochs21_25':{k:averages(v[20:25]) for k,v in data.items()},'common_last5':{k:averages(v[common-5:common]) for k,v in data.items()},'last':{k:v[-1] for k,v in data.items()},'best_pseudo':{k:max(v,key=lambda r:r['mean_dice']) for k,v in data.items()},'median_epoch_seconds':{k:st.median(r['seconds'] for r in v) for k,v in data.items()},'onset':{k:{reg:{str(th):next((r['epoch'] for r in v if r[reg]>th),None) for th in [0,0.01,0.05,0.1,0.5,0.7,0.8]} for reg in REGIONS} for k,v in data.items()}}
valpath=RUN_ROOT/RUNS['full'][1]/'Dataset001_BraTS24GLI/nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres/fold_0/validation_periodic/epoch_0050/summary.json'
validation=json.loads(snapshot(valpath,'full_validation_epoch0050.json').read_text());groups={}
for reg in REGIONS:
 present=[c for c in validation['cases'] if c['raw'][reg]['gt_voxels']>0];empty=[c for c in validation['cases'] if c['raw'][reg]['gt_voxels']==0]
 groups[reg]={'present_cases':len(present),'empty_cases':len(empty)}
 for stage in ['raw','postprocessed']:
  groups[reg][stage]={'present_mean_dice':st.mean(c[stage][reg]['dice'] for c in present),'empty_mean_dice':st.mean(c[stage][reg]['dice'] for c in empty) if empty else None,'empty_cases_with_false_positives':sum(c[stage][reg]['pred_voxels']>0 for c in empty)}
summary['full_validation_epoch50']={k:v for k,v in validation.items() if k!='cases'}
summary['full_validation_groups']=groups;summary['full_validation_case_count']=len(validation['cases'])
with (OUT/'epochs.csv').open('w',newline='') as f:
 fields=['run','epoch_index','epoch','train_loss','val_loss',*REGIONS,'mean_dice','lr','seconds','start_line','metric_line','timestamp']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
 for k,rows in data.items(): w.writerows(dict(run=k,**r) for r in rows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(16,9))
for index,(ax,metric) in enumerate(zip(axes.flat,[*REGIONS,'mean_dice','mean_dice','val_loss'])):
 restricted=index<4
 for k,rows in data.items():
  selected=rows[:common] if restricted else rows;x=np.array([r['epoch'] for r in selected]);y=np.array([r[metric] for r in selected])
  ax.plot(x,y,color=COLORS[k],alpha=.3,lw=1)
  rolling=np.array([y[max(0,i-4):i+1].mean() for i in range(len(y))])
  ax.plot(x,rolling,color=COLORS[k],lw=2,label=LABELS[k]);ax.scatter([x[-1]],[y[-1]],color=COLORS[k],s=16,zorder=3)
 ax.set_xlabel('Epoche completate (1-based)');ax.grid(alpha=.2)
 ax.set_title(metric+' · epoche comuni' if index<4 else 'Pseudo-Dice medio · intera storia' if metric=='mean_dice' else 'Loss di validazione · testa finale')
 if metric!='val_loss': ax.set_ylim(-.015,.92 if index in [0,4] else .62)
 if not restricted: ax.axvspan(common,max(map(len,data.values())),color='#888888',alpha=.07)
 if index==0: ax.legend(fontsize=8,loc='lower right')
fig.suptitle('SMMLIF: entrambe le run usano deep supervision\nLinea tenue: valore per epoca; linea marcata: media mobile causale di 5 epoche',fontsize=15)
fig.tight_layout(rect=(0,.035,1,.94));fig.text(.5,.012,'Pseudo-Dice su patch. Volumi completi: solo run full a epoca 50. GPU diverse: i tempi non isolano il costo DS.',ha='center',fontsize=10)
fig.savefig(OUT/'training_comparison.png',dpi=160);fig.savefig(OUT/'training_comparison.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.8));x=np.arange(3)
for off,stage,color in [(-.18,'raw','#2764a5'),(.18,'postprocessed','#dc682e')]:
 bars=axes[0].bar(x+off,[validation['mean'][stage][r] for r in REGIONS],.34,label=stage,color=color);axes[0].bar_label(bars,fmt='%.3f',padding=3)
axes[0].set_xticks(x,REGIONS);axes[0].set_ylim(0,1);axes[0].set_title('Dice medio sui 270 volumi');axes[0].legend();x=np.arange(2)
for off,stage,color in [(-.18,'raw','#2764a5'),(.18,'postprocessed','#dc682e')]:
 bars=axes[1].bar(x+off,[groups[r][stage]['empty_cases_with_false_positives'] for r in ['TC','ET']],.34,label=stage,color=color);axes[1].bar_label(bars,padding=3)
axes[1].set_xticks(x,['TC assente (77 casi)','ET assente (78 casi)']);axes[1].set_ylim(0,80);axes[1].set_title('Casi negativi con falsi positivi')
for ax in axes: ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Run 29062998 · validazione completa a epoca 50');fig.tight_layout();fig.savefig(OUT/'full_validation.png',dpi=160);plt.close(fig)
print(json.dumps({k:summary[k] for k in ['provenance','common_completed_epochs','epochs21_25','common_last5','median_epoch_seconds']},indent=2))
print('Output:',OUT)
