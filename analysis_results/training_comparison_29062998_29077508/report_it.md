# Confronto SMMLIF 29062998 / 29077508

**Conclusione:** la nuova configurazione di deep supervision mostra una convergenza iniziale leggermente più rapida, soprattutto su TC ed ET. Non è ancora dimostrato un miglioramento finale sui volumi completi. Entrambi i training hanno già la deep supervision attiva.

## Dati e convenzioni

Analisi dell'11 novembre 2026, su snapshot con **83 epoche completate** per 29062998 e **26** per 29077508. I job continuano a scrivere nei log; questi risultati descrivono le copie in `snapshots/`, non uno stato finale. Gli hash e l'ultimo timestamp delle metriche sono in `summary.json`. Il fuso dei timestamp dei log non è dichiarato, quindi non viene convertito.

Nel report le epoche sono numerate da 1: l'epoca 26 corrisponde a `Epoch 25` nei log originali. Il confronto diretto usa solo le prime 26 epoche comuni: 250 aggiornamenti per epoca, quindi 6500 aggiornamenti per run. La media mobile nei grafici è causale sulle ultime 5 epoche. I numeri derivano dai valori arrotondati stampati nei log.

## Che cosa cambia davvero

| Impostazione | 29062998, precedente | 29077508, nuovo |
| --- | --- | --- |
| Deep supervision | Attiva | Attiva |
| Pesi finale / metà / quarto | 1 / 0.3 / 0.1 | 1 / 0.5 / 0.25 |
| Pesi normalizzati | 0.7143 / 0.2143 / 0.0714 | 0.5714 / 0.2857 / 0.1429 |
| Risoluzione loss ausiliarie | Full: logits interpolati bilinearmente | Native: target ridotti nearest-exact |
| GPU | A100-SXM4-40GB | H100 |

Nel run precedente `resolution` è assente: il default compatibile con le configurazioni storiche è `full`, come confermano `paper_config.py` e `DEEP_SUPERVISION.md`. Il dizionario `snn_config` stampato nei due log coincide con quello salvato nei rispettivi `run_config.json`. Le sole differenze in questo dizionario sono pesi DS e `resolution`. Anche le configurazioni generali salvate differiscono solo per questi campi, directory e data di creazione.

Sono uguali: dataset BraTS24GLI, fold 0, 1080 casi train e 270 validation, seed 42, determinismo richiesto, 8 worker augmentation, batch 2, patch 81×128×128, 250 batch train/50 batch validation, FP32, rete paper_unet, neuroni selettivi multiscala nell'encoder, LR iniziale 0.01 e budget 400 epoche. L'asse temporale resta lungo 81 anche nelle teste native. `use_fptt=False`: il nome del trainer e i flag delle policy temporali non significano che sia attiva la regolarizzazione FPTT.

Il confronto resta limitato a un seed, GPU differenti e assenza di una fotografia versionata del codice eseguito al lancio. Non si può separare l'effetto dei pesi da quello della risoluzione usando solo queste due run.

## Andamento alle stesse epoche

Media del pseudo-Dice nelle ultime 5 epoche comuni, **22–26**:

| Metrica | DS full precedente | DS native nuova | Differenza in punti percentuali |
| --- | ---: | ---: | ---: |
| WT | 0.7750 | 0.7819 | +0.69 |
| TC | 0.1041 | 0.1768 | +7.27 |
| ET | 0.0000 | 0.0311 | +3.11 |
| mean_dice | 0.2930 | 0.3299 | +3.69 |

La media sulle epoche 21–25 conferma la stessa direzione: 0.2865 → 0.3187, cioè +3.22 punti percentuali. Su tutte le prime 25 epoche la differenza è più piccola: 0.2324 → 0.2417, perché gran parte della fase iniziale è quasi sovrapposta.

- **WT:** apprendimento rapido in entrambi, poi crescita graduale verso circa 0.78; vantaggio recente piccolo e rumoroso. All'ultima epoca comune il valore singolo della nuova run è inferiore, quindi non c'è una superiorità uniforme.
- **TC:** primo pseudo-Dice >0.01 all'epoca 19 nel nuovo run contro 21 nel precedente. TC >0.1 compare all'epoca 22 contro 24. È il segnale iniziale più consistente.
- **ET:** primo pseudo-Dice >0.01 all'epoca 24 contro 28. Nell'epoca 26 il nuovo run arriva a 0.0778, mentre il precedente è ancora a 0. Non è però ancora una buona segmentazione ET.
- **Loss di validazione della testa finale:** nelle epoche 22–26, -0.38496 → -0.39594; un piccolo miglioramento coerente con l'anticipo nelle metriche. Il singolo valore finale oscilla e non basta per un verdetto.

Le loss negative sono compatibili con la Dice+BCE usata. La loss di training combina tre teste; quella di validazione riguarda solo la testa finale. Pesi e target delle loss ausiliarie cambiano tra i run: la training loss più bassa del nuovo non prova, da sola, un miglioramento. Non si può usare il divario train/validation come una misura diretta di overfitting in questa configurazione.

## Storia più lunga del run precedente

Il run 29062998 passa da una fase dominata da WT a un forte miglioramento TC/ET intorno alle epoche 28–35. All'epoca 35 TC ed ET superano entrambi 0.8. All'epoca 50 il pseudo-Dice medio è 0.8538. Tra le epoche 36–50 la media è 0.8121; tra 71–83 è 0.8213: dopo il salto iniziale la crescita è molto più lenta e oscillante. Il massimo nello snapshot è 0.8542 all'epoca 83.

Quindi il valore attuale più basso della nuova run non è un segnale di fallimento: è ancora prima della transizione osservata nella precedente. Non si può però prevedere che la nuova supererà il plateau finale. La LR è ancora 0.00813 all'epoca 83: il programma da 400 epoche è lontano dal termine. Le metriche disponibili non mostrano un deterioramento sostenuto sufficiente a diagnosticare overfitting.

I due file `.err` erano vuoti all'ispezione. Il messaggio iniziale di ricerca del trainer è seguito dal caricamento riuscito del trainer esterno; l'assenza di `hiddenlayer` impedisce solo il disegno dell'architettura. Le diagnostiche disponibili mostrano firing rate non nulli nei tre blocchi e gap tra lambda circa 0.14 nelle epoche comuni; non emerge un collasso evidente. Non sono registrate loss per singola testa o norme dei gradienti per testa: il meccanismo del vantaggio non è identificabile da questi log.

## Validazione sui volumi completi

È disponibile solo per 29062998, all'epoca 50, sui **270 volumi**:

| Regione | Dice raw | Dice dopo postprocessing |
| --- | ---: | ---: |
| WT | 0.7856 | 0.7853 |
| TC | 0.5033 | 0.6393 |
| ET | 0.5208 | 0.6527 |
| mean | 0.6032 | 0.6924 |

Il pseudo-Dice medio a epoca 50 è 0.8538, mentre il Dice medio sui volumi è 0.6032 raw / 0.6924 postprocessing. Le metriche non sono intercambiabili: online si aggregano TP/FP/FN sulle patch campionate; la validazione completa usa sliding window, inferenza su volumi, media per caso e una convenzione esplicita per regioni vuote. La differenza non misura da sola un errore del modello o della pipeline.

Una parte sostanziale del limite pratico riguarda i **falsi positivi quando TC o ET sono assenti**:

| Gruppo | Casi | Casi con falsi positivi raw | Dopo postprocessing |
| --- | ---: | ---: | ---: |
| TC assente | 77 | 64 | 27 |
| ET assente | 78 | 64 | 27 |

Tra i casi positivi, il Dice TC è 0.6368 raw e 0.6353 postprocessing; ET è 0.6595 e 0.6522. Il guadagno globale del postprocessing deriva quindi principalmente dai casi negativi, non da una migliore sovrapposizione sulle lesioni presenti. Rimangono falsi positivi in circa il 35% dei volumi negativi. La convenzione è Dice=1 se GT e predizione sono vuoti, Dice=0 se la GT è vuota ma esistono falsi positivi.

## Miglioramenti da valutare, in ordine di priorità

1. **Portare la run native almeno alla validazione completa di epoca 50**, e confrontare i due `summary.json` a parità di epoca: Dice raw e postprocessing, regioni presenti/assenti, lesioni piccole. Se l'anticipo iniziale si conserva, proseguire il confronto a epoca 100. Il checkpoint va giudicato rispetto alla metrica di impiego: il migliore pseudo-Dice può non essere il migliore Dice medio sui volumi.
2. **Isolare i fattori prima di attribuire il guadagno alla DS.** Per misurare il solo effetto della risoluzione, confrontare full e native con gli stessi pesi, ad esempio aggiungendo native `[1,0.3,0.1]`. Per distinguere anche l'effetto dei pesi, il quarto punto del confronto è full `[1,0.5,0.25]`. Per misurare il vantaggio della presenza della DS serve inoltre una run `enabled: false`. Usare stessi budget, GPU e seed; replicare con altri seed la variante migliore prima di considerare robusto il risultato. Questi sono esperimenti proposti, non avviati.
3. **Valutare una supervisione più debole a 1/4 solo se emergono limiti su ET/lesioni piccole.** Una prova compatibile con il parser attuale è native `[1,0.5,0.1]`: aumenta il peso relativo della testa finale e riduce quello grossolano. Non è dimostrato che migliori questa run. Il nearest-exact può perdere piccoli oggetti alla riduzione: misurare prima quante componenti TC/ET sopravvivono nei target a 1/2 e 1/4. Evitare di sostituirlo automaticamente con max pooling, che cambierebbe la geometria delle lesioni. Il parser corrente richiede tre pesi strettamente positivi, quindi `[1,0.5,0]` richiederebbe un adeguamento del codice.
4. **Aggiungere diagnostica per distinguere apprendimento reale e bilanciamento della loss:** loss finale/metà/quarto, gradienti prodotti da ogni testa su uno stesso blocco condiviso e frequenza del clipping. Il nuovo schema assegna alle teste ausiliarie il 42.9% del peso totale contro il 28.6% precedente; il coefficiente diretto della testa finale scende da 0.714 a 0.571. Da questo dato non segue che il gradiente totale sia minore: le teste condividono parte della rete e i contributi possono allinearsi o interferire.
5. **Misurare separatamente i falsi positivi nei volumi negativi.** Prima di cambiare BCE/Dice, campionamento o soglie, verificare se la nuova DS riduce questo problema senza perdere sensibilità sulle lesioni piccole. Un miglioramento del solo pseudo-Dice non risponde a questa domanda.

Come riferimento, nnU-Net usa pesi geometrici normalizzati e annulla (o rende trascurabile in DDP) il peso dell'uscita più grossolana: è una motivazione per studiare il peso a 1/4, non una prova che questa architettura debba copiarlo. [Implementazione ufficiale nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py).

La modalità `nearest-exact` è documentata da PyTorch come interpolazione nearest coerente con PIL/scikit-image; il suo nome non garantisce la conservazione delle lesioni ridotte. [Documentazione ufficiale PyTorch](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.interpolate.html).

## Tempi

Mediana per epoca: 744.20 s (12.40 min) sulla A100 e 460.76 s (7.68 min) sulla H100. La seconda configurazione è più rapida nel tempo osservato, ma cambiano GPU e obiettivo ausiliario. Non si può attribuire la differenza alla deep supervision native né misurare il suo overhead rispetto a DS disattivata da questi log. I tempi per epoca non rappresentano l'intero costo delle validazioni complete periodiche.

## Artefatti e riproduzione

- `training_comparison.png` / `.pdf`: curve sulle epoche comuni e storia completa.
- `full_validation.png`: risultati full-volume e falsi positivi nei casi negativi.
- `epochs.csv`: metriche per epoca, con indici e righe di origine.
- `summary.json`: statistiche, differenze di configurazione, hash e sottogruppi.
- `snapshots/`: copie dei due log, configurazioni salvate e summary full-volume.
- `analyze.py`: riproduzione con `python analysis_results/training_comparison_29062998_29077508/analyze.py`; riusa gli snapshot esistenti.

L'analisi non modifica il codice del modello o i training in corso.
