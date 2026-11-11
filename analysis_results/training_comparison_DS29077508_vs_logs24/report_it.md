# SMMLIF: DS 29077508 vs baseline logs24

**Risultato:** la run DS fa emergere TC prima, ma dopo il salto di apprendimento delle regioni tumorali le due curve sono quasi sovrapposte. Al momento non è dimostrato un vantaggio stabile della DS. La learning rate iniziale cambia di un fattore 10, quindi questi due esperimenti non isolano il suo effetto.

## Snapshot e confronto corretto

- `logs24/SMMLIF_training.out`: 198 epoche completate nel log, run `SNN_paper_unet_sel3_fold0`, avviata il 2 novembre 2026.
- `SMMLIF_training_DS_01_29077508.out`: 39 epoche completate nello snapshot, run `SNN_2D_Unet_SMMLif_DS_01`, avviata l'11 novembre 2026.
- Le epoche qui sono numerate da 1: epoca 39 significa `Epoch 38` nel log. La DS è ancora in corso; sono state congelate copie dei file per rendere riproducibili i numeri.
- Il confronto diretto usa le prime 39 epoche: 9750 aggiornamenti per run. I risultati sulle epoche successive della baseline sono soltanto contesto.

## Configurazioni

| Impostazione | Baseline logs24 | Nuova DS |
| --- | --- | --- |
| Deep supervision | Assente nella configurazione storica | Attiva, native, pesi 1 / 0.5 / 0.25 |
| LR iniziale | 0.001 | 0.01 |
| Budget dello scheduler | 200 epoche | 400 epoche |
| GPU | A100-SXM4-40GB | H100 |
| Batch / patch | 2 / 81×128×128 | 2 / 81×128×128 |
| Batch per epoca train / validation | 250 / 50 | 250 / 50 |
| Seed / determinismo / precisione | 42 / richiesto / FP32 | 42 / richiesto / FP32 |
| Fold e casi | 0, 1080 train / 270 validation | Uguali |
| Asse temporale / k / FPTT | 0 / 81 / disattivato | Uguali |

Le configurazioni stampate nei log coincidono con i rispettivi `run_config.json`. I conteggi di batch, non espliciti nel vecchio `snn_config`, sono confermati dai `debug.json` salvati. La rete principale ha gli stessi canali encoder/decoder, neuroni selettivi a tre memorie nell'encoder, dropout, GroupNorm e testa continua. Il vecchio schema `neurons.selective` è diventato `neurons.multi_memory` più `neurons.selective_readout`: il cambio di posizione dei campi non va interpretato automaticamente come un'architettura diversa. Non è disponibile qui una fotografia versionata del codice eseguito nei due lanci; non si può garantire l'equivalenza di tutti i dettagli storici.

Il flag nativo `enable_deep_supervision=False` nei `debug.json` è presente anche nella nuova run: riguarda la gestione delle piramidi nnU-Net. La DS personalizzata è controllata da `paper_unet.deep_supervision` ed è attiva nel nuovo training. La baseline storica non contiene quella opzione e precede l'integrazione delle teste DS.

## Curve alle stesse epoche

Media del pseudo-Dice nelle ultime cinque epoche comuni, **35–39**:

| Metrica | Baseline senza DS | Nuova DS | Delta in punti percentuali |
| --- | ---: | ---: | ---: |
| WT | 0.8491 | 0.8402 | -0.89 |
| TC | 0.7829 | 0.7888 | +0.59 |
| ET | 0.8023 | 0.8161 | +1.38 |
| mean_dice | 0.8114 | 0.8150 | +0.36 |

Media sulle ultime dieci epoche comuni, **30–39**: **0.8014 → 0.8026**, circa +0.12 punti. È una differenza piccola rispetto alle oscillazioni visibili.

L'andamento ha tre fasi:

1. **Avvio:** WT compare prima nella nuova run; sulle epoche 1–10 il pseudo-Dice medio è 0.1253 contro 0.1880. La LR dieci volte più alta è un fattore confondente importante.
2. **Segnali precoci di TC:** TC supera 0.01 all'epoca 19 nella DS contro 26 nella baseline; supera 0.1 all'epoca 22 contro 28. ET supera 0.01 all'epoca 24 contro 25. Nelle epoche 21–25 la media è 0.2698 contro 0.3187 (+4.89 punti).
3. **Salto e convergenza iniziale:** la baseline recupera rapidamente intorno alle epoche 28–30. TC ed ET superano 0.7 all'epoca 28 senza DS e 29 con DS: anticipare valori piccoli non significa raggiungere prima una segmentazione buona. Il pseudo-Dice medio supera 0.8 all'epoca 28 nella baseline e 30 nella DS. Nelle epoche 29–38 le medie sono 0.80141 e 0.80115, sostanzialmente uguali.

Nelle epoche 35–39, WT è leggermente migliore senza DS; TC ed ET leggermente migliori nella DS. La validazione loss della testa finale media passa da -0.51486 a -0.51720: anche questo segnale è piccolo. La training loss non è direttamente confrontabile: nella baseline usa la sola testa finale, nella DS una combinazione normalizzata di tre loss.

La media mobile a 5 epoche nei grafici è causale. Non sono intervalli di confidenza e non è stata stimata significatività statistica: un seed e metriche correlate nel tempo non consentono un'affermazione di superiorità robusta.

## Quanto migliora ancora la baseline

La baseline prosegue dopo il tratto comune: pseudo-Dice medio 0.8499 all'epoca 50; media 0.8497 sulle epoche 151–198; massimo singolo 0.8853 all'epoca 195; ultimo valore 0.8624 all'epoca 198. La nuova DS ha massimo singolo 0.8280 all'epoca 37 e ultimo valore 0.8144 all'epoca 39. Questi massimi non sono un confronto alla stessa maturità del training.

Le curve disponibili non mostrano un peggioramento sostenuto della validazione sufficiente a diagnosticare overfitting. Il nuovo training è ancora vicino all'inizio dello scheduler da 400 epoche e non può essere giudicato come risultato finale.

## Volumi completi

I risultati full-volume disponibili per la baseline, sui 270 casi del fold, sono:

| Checkpoint | Dice medio raw | Dopo postprocessing |
| --- | ---: | ---: |
| epoca 50 | 0.6259 | 0.7112 |
| epoca 100 | 0.6466 | 0.7413 |
| epoca 150 | 0.6720 | 0.7518 |
| checkpoint_best | 0.6696 | 0.7581 |

Per il checkpoint best, i Dice postprocessing sono WT=0.8487, TC=0.7055, ET=0.7202. Il file `logs24/SMMLIF_results.out` riporta gli stessi valori aggregati del `summary.json` salvato. Il summary della validazione best non contiene l'epoca del checkpoint, quindi non viene assegnata arbitrariamente a 195 o 198.

La DS non ha ancora una validazione completa nello snapshot. Il primo confronto utile sarà quindi **epoca 50 contro epoca 50**, con riferimento baseline **0.6259 raw / 0.7112 postprocessing**. Il valore best storico 0.7581 è invece un riferimento successivo, ottenuto con una maturità diversa. Il pseudo-Dice online sulle patch non è intercambiabile con il Dice medio per volume.

## Interpretazione e prossima verifica

Non aumenterei ulteriormente il peso delle teste sulla base di questi dati. La configurazione DS è funzionante e arriva allo stesso livello iniziale della baseline, ma il guadagno stabile non è ancora emerso. Per capire se aiuta davvero:

- confrontare raw e postprocessing a epoche 50 e 100, distinguendo casi con TC/ET presenti o assenti e lesioni piccole;
- eseguire un confronto con DS attiva/disattiva a **stessa LR, stesso budget dello scheduler, stessa GPU e stessa versione del codice**;
- giudicare medie su più epoche e validazioni full-volume, evitando di scegliere un singolo picco;
- replicare con altri seed la variante più promettente prima di attribuire piccoli scarti alla DS.

Queste sono indicazioni per il confronto sperimentale; nessun training è stato avviato o modificato.

## Costo e artefatti

Mediana per epoca: 719.30 s sulla A100 contro 460.51 s sulla H100. Questi tempi non isolano l'overhead della DS. La durata delle validazioni periodiche complete non è rappresentata dal solo `Epoch time`.

Grafici: `training_comparison.png` / `.pdf`, `full_validation.png` (baseline a epoca 50). Dati: `epochs.csv`, `summary.json`, snapshot dei log e delle configurazioni. Riproduzione: `python analysis_results/training_comparison_DS29077508_vs_logs24/analyze.py`, che riusa gli snapshot esistenti. I timestamp dei log non dichiarano il fuso e sono conservati senza conversioni.
