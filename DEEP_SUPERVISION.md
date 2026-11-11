# Deep supervision per Paper U-Net

La deep supervision si configura direttamente in `unet_config.yaml`, nella
sezione `snn.paper_unet.deep_supervision`. Il config principale la abilita con
pesi `[1.0, 0.5, 0.25]` e `resolution: native`; impostare `enabled: false` per disabilitarla.
Aggiunge due teste durante il training. L'effetto sul Dice deve essere misurato
confrontando run nuovi a parità delle altre impostazioni.

## Configurazione

```yaml
snn:
  paper_unet:
    deep_supervision:
      enabled: true
      weights: [1.0, 0.5, 0.25]
      resolution: native
```

L'ordine è uscita finale, testa su `concat2_conv` a metà risoluzione, testa su
`concat1_conv` a un quarto della risoluzione. Le teste sono convoluzioni 2D 1x1
con tre logits continui WT/TC/ET e bias, senza neuroni o normalizzazione.

La loss è `(L_finale + 0.5 L_meta + 0.25 L_quarto) / 1.75`: pesi effettivi
0.5714, 0.2857, 0.1429. Tutte e tre le loss sono attive. Ogni termine usa la stessa
Dice+BCE nativa. Con `resolution: native`, i logits ausiliari restano sulle griglie
H/W a metà e a un quarto e i target sono ridotti con nearest-exact; l’asse della
sequenza non viene ridotto, anche se `temporal_axis` è 1 o 2. La ignore mask nativa
segue la stessa trasformazione.

Per dimensioni dispari, prima di ridurre i target viene riprodotto il padding
inferiore/destro della rete. I voxel aggiunti sono background, oppure ignorati
quando la loss usa un canale ignore. Questo mantiene le coordinate delle teste
native senza deformare il target per adattarlo al padding. Alle scale ridotte
lesioni molto piccole possono scomparire; la loss finale resta a piena risoluzione.

nnU-Net usa pesi geometrici normalizzati e normalmente esclude l’uscita più
grossolana. Qui quella testa resta attiva per scelta dell’esperimento.

`resolution: full` conserva il comportamento precedente: logits ausiliari
interpolati e ritagliati prima della loss sui target originali. È il valore
predefinito quando il campo manca, così i checkpoint/config precedenti conservano
il loro obiettivo. La modalità è salvata nella configurazione dei checkpoint;
per passare a `native` usare un nuovo run, evitando di mescolare le due loss
nella stessa storia di training.

`enable_deep_supervision` nativo del trainer resta falso: qui controlla le piramidi
di target di nnU-Net. Le uscite ausiliarie vengono richieste esplicitamente dal
nostro `train_step` con `return_auxiliary=True`. La loss di training è la media
pesata delle tre teste; la loss di validazione e il pseudo-Dice misurano solo la
testa finale. Le due loss non descrivono più esattamente lo stesso obiettivo.

## Avviare il nuovo esperimento

```bash
sbatch --export=ALL,CONFIG_PATH=/gpfs/home6/apiaghiardelli/hippo_segs/unet_config.yaml train_validate_snn_brats24_fold0.sbatch
```

La directory dei risultati è quella specificata in `run_dir` nel config.
Scegliere una directory nuova quando si cambia la deep supervision. Il limite
temporale dello script esistente resta un giorno: 400 epoche possono richiedere
riprese da checkpoint. Per una ripresa, specificare la directory del run avviato:

```bash
sbatch --export=ALL,RUN_DIR=/percorso/del/run,TOTAL_EPOCHS=400 resume_snn_brats24_fold0.sbatch
```

Per il primo confronto si parte da un run nuovo: i checkpoint senza teste non
possono riprendere direttamente un training con teste/optimizer differenti.
I checkpoint esistenti continuano a essere caricabili con deep supervision
disabilitata. I nuovi checkpoint conservano teste, optimizer e configurazione.

## Velocità e memoria

- Encoder e decoder vengono eseguiti una volta. Una sola backward combina tutte
  le loss; non si eseguono tre training completi.
- Parametri aggiunti con i canali attuali: `2 * (128 * 3 + 3) = 774`, circa 0.063%
  dei parametri della rete. I soli pesi FP32 occupano circa 3 KiB.
- Batch 2, patch 81x128x128, modalità native: i logits ausiliari sono
  [2,3,81,64,64] e [2,3,81,32,32], per 9,953,280 byte (circa 9.49 MiB) in FP32.
  La modalità full richiede 60.75 MiB per questi tensori. Gradienti, target, loss
  e workspace aumentano ulteriormente il picco di memoria, da misurare sulla GPU.
- Durante validazione e inferenza non vengono eseguite le teste ausiliarie:
  il normale forward restituisce un solo tensore, con le stesse operazioni
  principali. Non è richiesto un ensemble. Le teste restano nel checkpoint.
- Il training aggiunge convoluzioni 1x1, riduzione dei target e due Dice+BCE con
  le rispettive backward. Non è disponibile qui una misura del rallentamento A100;
  non si può dedurlo dal solo numero di parametri.
- Una convergenza migliore o più rapida in epoche non è ancora dimostrata.

## Misurare il costo sull'A100

```bash
sbatch profile_deep_supervision.sbatch
```

Il job misura in FP32 batch 2 e patch 81x128x128, senza leggere il dataset e senza
scrivere checkpoint. Salva il report in
`analysis_results/deep_supervision/a100_<job_id>.json`. Non avvia un training.

Il benchmark riporta le forme effettive delle teste e i loro byte nel campo
`additional_auxiliary_logits_bytes`, oltre a modalità e pesi della supervisione.
I report storici con `additional_full_resolution_logits_bytes` descrivono la
vecchia modalità full e non vanno letti come misure della nuova modalità native.

Il benchmark confronta gli stessi pesi e dati sintetici, con seed accoppiati,
warmup e ordine delle prove alternato. Include forward, Dice+BCE nativa,
backward e clipping; esclude aggiornamento SGD, caricamento, augmentation,
diagnostiche e validazioni complete. Riporta tempi e picchi CUDA. Usa
`batch_dice=False`, come il piano attuale, e forza `k` alla profondità della patch:
non rappresenta configurazioni con finestre più corte o FPTT.

Una prova funzionale ridotta su CPU è riproducibile con:

```bash
.venv/bin/python -m snn_nnunet.profile_deep_supervision --device cpu --patch-size 4 32 32 --batch-size 1 --warmup 3 --repeats 15 --output analysis_results/deep_supervision/cpu_smoke.json
```

I tempi CPU su patch ridotta non prevedono la percentuale di rallentamento A100.
Per stimare il tempo per epoca, misurare anche epoche reali dopo il warmup,
confrontando gli stessi 250 batch ed escludendo la validazione periodica.

## Verifiche

I test coprono gradienti e aggiornamento delle teste, loss normalizzata,
maschere ignorate, assi temporali, finestre, dimensioni dispari, fusioni
bidirezionali, equivalenza dei pesi comuni e output singolo in validazione.
La revisione indipendente ha anche provato save/load nativo con/senza teste e
FPTT e un ciclo DDP CPU a singolo processo. Non è una verifica CUDA multi-GPU.

Il confronto di accuratezza richiede lo stesso budget di aggiornamenti della
baseline e il Dice sui medesimi volumi completi, prima e dopo il postprocessing.
