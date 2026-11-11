# Confronto aggiornato SMMLIF / SMMLIF DS / UNET2D

**Conclusione:** a epoca 50 le tre run hanno pseudo-Dice medio simile. La DS non mostra ancora un vantaggio stabile sullo SMMLIF storico; UNET2D non spiking raggiunge valori alti molto prima. Le validazioni full-volume della DS e del file UNET2D richiesto sono ancora incomplete nello snapshot.

## Grafici richiesti

I grafici sono interamente in inglese. Ogni punto usa il valore di quella singola epoca, con media aritmetica `(WT + TC + ET) / 3`. Non sono presenti moving average, EMA, medie cumulative o altre forme di smoothing. Le linee collegano solo le osservazioni disponibili; non estrapolano oltre la fine di un training. Le epoche sono numerate da 1: epoca 50 corrisponde a `Epoch 49` nei log.

- `mean_dice_by_epoch.png` e `.pdf`: media per epoca, prime 50 epoche comuni e intera storia disponibile.
- `regional_dice_by_epoch.png` e `.pdf`: WT, TC, ET e media sulle prime 50 epoche comuni.
- `partial_matched_validation.png`: confronto provvisorio sui soli casi già valutati da tutti; non è una curva di training né un risultato finale del fold.

## File e stato

| Fonte | Contenuto usato | Stato nello snapshot |
| --- | --- | --- |
| `SMMLIF_training_DS_01_29077508.out` | Training DS e validazione epoca 50 in corso | 50 epoche; 20/270 casi validati |
| `logs24/SMMLIF_training.out` | Training storico senza DS | 198 epoche completate |
| `logs24/UNET2D_01_results_29118352.out` | Validazione di `checkpoint_best.pth` | 168/270 casi validati |
| `logs24/UNET2D_01_training.out` | Training associato allo stesso run UNET2D | 150 epoche completate |

Il file UNET2D richiesto non contiene metriche per epoca: per aggiungerne l'evoluzione al grafico è stato usato il training associato, verificando la stessa directory `SNN_2D_Unet`. Tutti gli input sono congelati in `snapshots/`; hash e dimensioni sono in `summary.json`. Le validazioni in corso possono essere avanzate dopo questo snapshot.

## Confrontabilità

| Impostazione | SMMLIF storico | SMMLIF DS | UNET2D |
| --- | --- | --- | --- |
| Spiking encoder / decoder | Attivo | Attivo | Disattivo, usa GELU |
| Deep supervision | Assente | Native, 1 / 0.5 / 0.25 | Disattiva |
| LR iniziale | 0.001 | 0.01 | 0.01 |
| Scheduler | 200 epoche | 400 epoche | 400 epoche |
| GPU | A100 | H100 | A100 |

Sono uguali fold 0, seed 42, FP32, patch 81×128×128, batch 2 e 250 batch train / 50 validation per epoca. Le prime 50 epoche corrispondono a 12500 aggiornamenti per run. Le configurazioni stampate nei log coincidono con i rispettivi `run_config.json` salvati. Non è disponibile una fotografia versionata del codice di ciascun lancio.

Il nome del trainer SNN e gli override neuronali presenti nel config UNET2D non implicano neuroni attivi: `encoder_spiking=false` e `decoder_spiking=false` selezionano GELU nei blocchi. La testa è continua anche nelle SNN. Il confronto con UNET2D cambia quindi la dinamica del modello, oltre alla DS. Non è un'ablation della sola deep supervision.

## Pseudo-Dice a epoca 50

| Modello | WT | TC | ET | Media della singola epoca |
| --- | ---: | ---: | ---: | ---: |
| SMMLIF senza DS | 0.8605 | 0.8328 | 0.8565 | 0.8499 |
| SMMLIF con DS | 0.8611 | 0.8378 | 0.8560 | 0.8516 |
| UNET2D non spiking | 0.8763 | 0.8318 | 0.8594 | 0.8558 |

A questa epoca la DS supera lo SMMLIF storico di **0.17 punti percentuali** nella media; UNET2D supera la DS di **0.42 punti**. Sono differenze piccole e non indicano una superiorità robusta.

La lettura delle curve aiuta a evitare di basarsi sul solo ultimo punto:

- UNET2D supera pseudo-Dice medio 0.8 all'epoca **8**, lo SMMLIF storico all'epoca **28**, la DS all'epoca **30**. È il vantaggio iniziale più evidente.
- Lo SMMLIF con DS fa emergere TC prima dello storico, ma il salto di apprendimento delle SNN porta poi le due curve vicino a 0.8 tra le epoche 28 e 30.
- Come riepilogo numerico, la media dei dieci valori delle epoche 41–50 è **0.8198 SMMLIF**, **0.8164 SMMLIF DS**, **0.8316 UNET2D**. Questa statistica non viene tracciata nei grafici: serve solo a mostrare che il vantaggio DS dell'ultimo punto non è persistente nel tratto recente.
- La validation loss a epoca 50 è -0.5353, -0.5356 e -0.5545 rispettivamente: SMMLIF e DS sono praticamente uguali, UNET2D è migliore in questa metrica. La training loss DS combina tre teste e non va confrontata direttamente con le loss delle run a testa singola.

L'anticipo iniziale della DS non è attribuibile alla sola DS perché rispetto allo storico cambia anche la LR di un fattore 10. La curva non spiking molto più rapida è coerente con una maggiore facilità di ottimizzazione in questo esperimento, ma non identifica da sola il meccanismo responsabile.

## Validazioni complete a epoca 50

Sui 270 volumi completi sono disponibili:

| Modello | Dice medio raw | Dice medio postprocessing |
| --- | ---: | ---: |
| SMMLIF senza DS | 0.6259 | 0.7112 |
| UNET2D non spiking | 0.6156 | 0.7123 |
| SMMLIF DS | In corso | In corso |

Il Dice medio full-volume è una media per caso; il pseudo-Dice online aggrega statistiche sulle patch. Sono misure diverse e non devono essere sovrapposte nello stesso grafico come se fossero equivalenti. A epoca 50 le due validazioni complete disponibili sono molto vicine dopo postprocessing, benché UNET2D abbia convergenza online iniziale molto più rapida.

## Risultati parziali del file UNET2D richiesto

Il file `UNET2D_01_results_29118352.out` contiene una validazione del checkpoint best ancora in corso. Non viene stimato un risultato finale usando soltanto i primi 168 casi, né confrontata quella media con 270 casi di un altro modello.

Il grafico provvisorio `partial_matched_validation.png` usa gli **stessi 20 casi completati** in tutte e tre le fonti: SMMLIF storico best, DS epoca 50, UNET2D best. Le diverse maturità dei checkpoint sono indicate nel grafico e impediscono un confronto controllato della DS.

Per distinguere anche il budget, sui medesimi 20 casi è stato calcolato il confronto a **epoca 50 per tutti**:

| Modello | Dice medio raw sui casi comuni | Postprocessing sui casi comuni |
| --- | ---: | ---: |
| SMMLIF senza DS | 0.6871 | 0.7201 |
| SMMLIF con DS | 0.5928 | 0.6621 |
| UNET2D non spiking | 0.6364 | 0.7713 |

In questo piccolo sottoinsieme la DS è inferiore agli altri due modelli. Sono i primi casi dell'ordine di valutazione, non un campione casuale; non costituiscono una stima affidabile della classifica sui 270 volumi. Aspettare la fine della validazione è necessario prima di concludere che la DS peggiori la generalizzazione.

## Valutazione pratica

Al momento la DS non ha mostrato un miglioramento stabile delle metriche online. Il risultato finale sui volumi è ancora aperto. Prima di cambiare pesi o architettura confronterei i summary completi a epoca 50, distinguendo regioni presenti/assenti e lesioni piccole. Per misurare l'effetto della sola DS serve una run SMMLIF senza DS con la stessa LR 0.01, scheduler da 400 epoche, GPU e versione del codice.

Non si può attribuire un vantaggio di velocità alla DS dai tempi osservati, perché la GPU cambia. Nessun training è stato modificato o avviato da questa analisi.

## Riproducibilità

`epochs.csv` contiene i valori originali per epoca e i numeri di riga delle fonti; `plot_data.json` contiene esattamente i punti tracciati. `matched_validation_cases.csv` contiene il sottoinsieme provvisorio. `summary.json` raccoglie configurazioni e statistiche. Eseguire `python analysis_results/training_comparison_3models_2026-11-11/analyze.py` per rigenerare gli artefatti dagli snapshot esistenti. Non si rileggono automaticamente nuove righe dei log già congelati.
