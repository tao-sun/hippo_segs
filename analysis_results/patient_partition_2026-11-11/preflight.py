"""CPU-only verification of real metadata and both trainer split implementations."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from snn_nnunet.partition_runtime import (
    load_partition, stage_run_partition, restore_run_partition, evaluation_case_ids,
)
from snn_nnunet.patient_partition import patient_id
from snn_nnunet.run_model import _config

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/gpfs/scratch1/shared/apiaghiardelli/nnUNet_preprocessed/Dataset001_BraTS24GLI')
PARTITION=ROOT/'data_partitions/brats24_patient_seed42'
REPORT=Path(__file__).resolve().parent
sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
original={name:sha(SOURCE/name) for name in ('splits_final.json','SNNPlans.json','dataset.json')}
manifest=load_partition(PARTITION)
configs=[_config(ROOT/f'configs/patient_wise/fold_{fold}.yaml') for fold in range(5)]
assert _config(ROOT/'unet_config.yaml')['data_partition']==str(PARTITION)
assert len(manifest['cases'])==1350
assert len({patient_id(x) for x in manifest['cases']})==613
assert len({config['run_dir'] for config in configs})==5
assert all(config['data_partition']==str(PARTITION) for config in configs)
checks=[]
with tempfile.TemporaryDirectory(prefix='patient-preflight-') as temporary:
    run=Path(temporary)/'run'
    staged=stage_run_partition(SOURCE,run,PARTITION,'nnUNetPlans_3d_fullres')
    restore_run_partition(run)
    os.environ['nnUNet_results']=str(run)
    os.environ['nnUNet_raw']='/gpfs/scratch1/shared/apiaghiardelli/nnUNet_raw'
    os.environ['nnUNet_compile']='false'
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from snn_nnunet.trainer import nnUNetTrainerSNNFPTT
    plans=json.loads((staged/SOURCE.name/'SNNPlans.json').read_text())
    dataset=json.loads((staged/SOURCE.name/'dataset.json').read_text())
    with (REPORT/'trainer_preflight.log').open('w') as log, redirect_stdout(log):
        for fold in range(5):
            for cls in (nnUNetTrainer,nnUNetTrainerSNNFPTT):
                trainer=cls(plans=dict(deepcopy(plans), continue_training=False),configuration='3d_fullres',fold=fold,
                            dataset_json=dataset,device=torch.device('cpu'))
                train,val=trainer.do_split()
                test=evaluation_case_ids(trainer,'test')
                expected=manifest['folds'][fold]
                assert list(train)==expected['train']
                assert list(val)==expected['val']
                assert list(test)==expected['test']
                assert not set(test)&set(train+val)
                checks.append({'fold':fold,'trainer':cls.__name__,
                               'train_cases':len(train),'val_cases':len(val),'test_cases':len(test)})
    # Prove run-local plans can be edited without modifying the shared file.
    (staged/SOURCE.name/'SNNPlans.json').write_text('{"private_preflight_marker":true}')
assert original=={name:sha(SOURCE/name) for name in original}
assert original['splits_final.json']==manifest['legacy_split_sha256']
report={'cases':1350,'base_patient_ids':613,'seed':manifest['seed'],
        'counts':manifest['counts'],'trainer_checks':checks,
        'shared_metadata_unchanged':True,'source_sha256':original,
        'gpu_training_executed':False}
(REPORT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
