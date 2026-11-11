"""Read-only audit of archived nnUNet3D logs, current split and reference branch."""
from pathlib import Path
import argparse,ast,contextlib,hashlib,io,json,re,subprocess
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];SNAP=OUT/'snapshots';SNAP.mkdir(exist_ok=True)
SCRATCH=Path('/gpfs/scratch1/shared/apiaghiardelli');PRE=SCRATCH/'nnUNet_preprocessed/Dataset001_BraTS24GLI'
provenance={}
def freeze(source,name):
 target=SNAP/name
 if not target.exists():target.write_bytes(source.read_bytes())
 provenance[str(source)]={'snapshot':name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 return target.read_text()

sources=['logs24/nnUNET3D_training.out','logs24/nnUNET3D_results.out','snn_nnunet/custom_full_validation.py','snn_nnunet/run_model.py','snn_nnunet/dataset_conversion.py','snn_nnunet/empty_dice_evaluation.py','snn_nnunet/postprocess_saved_volumes.py','postprocessing.py','validate_unet3d_brats24_fold0_best.sbatch','postprocess_nnunet.sbatch']
texts={name:freeze(ROOT/name,name.replace('/','__')) for name in sources}
splits=json.loads(freeze(PRE/'splits_final.json','splits_final.json'));freeze(PRE/'dataset.json','dataset.json')
reference={}
for name in ['snn_nnunet/custom_full_validation.py','snn_nnunet/run_model.py','snn_nnunet/dataset_conversion.py','postprocessing.py']:
 p=SNAP/('reference__'+name.replace('/','__'))
 if not p.exists():p.write_text(subprocess.check_output(['git','show','nnUnet_alignment:'+name],text=True))
 reference[name]=p.read_text()
lines=texts['logs24/nnUNET3D_training.out'].splitlines()
configuration=next(ast.literal_eval(line.strip()) for line in lines if line.lstrip().startswith("{'data_identifier':"))
num=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?';rows=[];row=None
for line in lines:
 m=re.search(r': Epoch (\d+)\s*$',line)
 if m:row={'epoch_index':int(m[1]),'completed_epoch':int(m[1])+1}
 if row is None:continue
 for label,key in [('Current learning rate','lr'),('train_loss','train_loss'),('val_loss','val_loss'),('Epoch time','seconds')]:
  m=re.search(re.escape(label)+r':?\s+('+num+')',line)
  if m:row[key]=float(m[1])
 if 'Pseudo dice [' in line:
  values=[float(v) for v in re.findall(r'np.float32\(([^)]+)\)',line)];row['pseudo_dice']=values;row['pseudo_mean']=float(np.mean(values))
 if 'Epoch time:' in line:rows.append(row.copy())
result=texts['logs24/nnUNET3D_results.out'];m=re.search(r'Mean Dice \(raw\): ('+num+r'); postprocessed: ('+num+')',result);raw,pp=map(float,m.groups())
val_ids=[m[1] for line in result.splitlines() if (m:=re.match(r'^\[\d+/270\]\s+(\S+)',line))]
s=splits[0];group=lambda x:x.rsplit('-',1)[0];tr={group(x) for x in s['train']};va={group(x) for x in s['val']};shared=tr&va
all_ids=sorted(s['train']+s['val']);idx=np.arange(len(all_ids));np.random.RandomState(12345).shuffle(idx)
reconstructed={all_ids[i] for i in idx[:len(s['val'])]}
def functions(src):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
a=functions(reference['snn_nnunet/custom_full_validation.py']);b=functions(texts['snn_nnunet/custom_full_validation.py'])
identity={name:a[name]==b[name] for name in ['labelmap_to_regions','threshold_probabilities','dice_metrics','postprocess_regions','_canonicalize_regions']}
module=ast.parse(texts['snn_nnunet/custom_full_validation.py']);nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['dice_metrics','summarize_cases']]
ns={'np':np,'REGION_ORDER':('WT','TC','ET'),'DICE_CONVENTION':{'empty_reference_and_prediction':1.,'empty_reference_with_false_positives':0.}}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<metric audit>','exec'),ns)
empty=np.zeros((3,2,2,2),dtype=np.uint8);one=empty.copy();one[:,0,0,0]=1
metric_checks={'empty_empty_one':all(v['dice']==1 for v in ns['dice_metrics'](empty,empty).values()),'empty_with_fp_zero':all(v['dice']==0 for v in ns['dice_metrics'](empty,one).values()),'perfect_one':all(v['dice']==1 for v in ns['dice_metrics'](one,one).values())}
tree=ast.parse(texts['snn_nnunet/postprocess_saved_volumes.py']);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
def no_volume_io(*args,**kwargs):raise AssertionError('Unexpected volume IO')
ns2={'argparse':argparse,'Path':Path,'postprocess_saved_volumes':no_volume_io,'__doc__':'Isolated parser audit'}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<parser audit>','exec'),ns2);err=io.StringIO();parser_error=None
with contextlib.redirect_stderr(err):
 try:ns2['main'](['--input-dir','/tmp/pred','--reference-dir','/tmp/ref','--output-dir','/tmp/out','--plans-json','/tmp/plans.json','--dataset-json','/tmp/dataset.json','--min-component-sizes','20,','30,','50','--closing-radius','1'])
 except SystemExit as error:parser_error={'exit_code':error.code,'stderr':err.getvalue()}
model=SCRATCH/'nnUNet_results/unet3d_fold0_baseline/Dataset001_BraTS24GLI/nnUNetTrainer__nnUNetPlans__3d_fullres'
expected={'plans':model/'plans.json','dataset':model/'dataset.json','checkpoint_best':model/'fold_0/checkpoint_best.pth','custom_summary':model/'fold_0/validation_96patch_postprocessed/summary.json','native_summary':model/'fold_0/validation/summary.json'}
model_files={key:{'path':str(p),'exists':p.is_file()} for key,p in expected.items()}
audit={'date':'2026-11-11','scope':'Model and pipeline read-only; analysis artifacts only','reference_branch':'nnUnet_alignment','reference_commit':subprocess.check_output(['git','rev-parse','nnUnet_alignment'],text=True).strip(),'paper':{'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC12808724/','confirmed_by_user':True,'brats2024_mean':.8534,'protocol':'Patient-level five-fold CV; 128^3 training crops; reported experiments batch 4 and 300 epochs; inference 96^3, overlap 50%, Gaussian blending, CCA and closing.','unreported_details':['Exact fold identifiers','Baseline-specific saved plans','Empty-region averaging policy','Exact TC mask (prose ambiguous)','CCA thresholds and morphology kernel']},'historical_run':{'raw_mean':raw,'postprocessed_mean':pp,'gap_to_paper':.8534-pp,'postprocessing_gain':pp-raw,'fold':0,'n_cases':len(val_ids),'configuration':configuration,'archived_completed_epochs':len(rows),'last_epoch':rows[-1],'scheduler_1000epoch_poly09_max_lr_error':max(abs(r['lr']-.01*(1-r['epoch_index']/1000)**.9) for r in rows),'checkpoint':'checkpoint_best.pth, exact epoch unavailable','inference_patch':[96,96,96]},'split_audit':{'train':len(s['train']),'validation':len(s['val']),'matches_historical_validation_cases':set(val_ids)==set(s['val']),'reproduced_casewise_kfold_seed12345':reconstructed==set(s['val']),'shared_prefix_groups':len(shared),'validation_cases_with_train_prefix':sum(group(x) in tr for x in s['val']),'interpretation':'If BraTS-GLI-xxxxx is patient ID, this split is not patient-disjoint; expected bias is optimistic.'},'branch_comparison':{'dataset_conversion_identical':reference['snn_nnunet/dataset_conversion.py']==texts['snn_nnunet/dataset_conversion.py'],'morphology_identical':reference['postprocessing.py']==texts['postprocessing.py'],'function_identity':identity,'training':'Both call stock nnUNetTrainer / nnUNetPlans / 3d_fullres. New empty-region wrapper changes evaluation only.','reference_eval_loader':'Uses shared preprocessed plans; --results-root is checked but actual model folder depends on environment. The historical checkpoint path matches the requested run.','current_eval_loader':'Uses saved model plans/dataset and explicitly sets result root.'},'metric_checks':metric_checks,'parser_error':parser_error,'parser_error_causality':'Not a demonstrated cause of 0.815342: successful validation uses valid thresholds 161 166 50.','original_model_files':model_files,'provenance':provenance}
assert all(metric_checks.values()) and all(identity.values()) and reconstructed==set(s['val']) and set(val_ids)==set(s['val']) and parser_error['exit_code']==2
supplement=OUT/'supplemental_checks.json'
if supplement.exists():audit.update(json.loads(supplement.read_text()))
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'training_epochs.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'raw':raw,'postprocessed':pp,'gap':.8534-pp,'epochs_in_log':len(rows),'split':audit['split_audit'],'metrics':metric_checks,'model_files':model_files},indent=2))
