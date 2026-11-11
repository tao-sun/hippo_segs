# Audit nnUNet3D: risultato 0.815342 e allineamento al paper

**Esito:** ci sono differenze di protocollo verificabili, ma non è stato identificato un errore di training che dimostri di causare il divario di 3.81 punti. Nel branch `nnUnet_alignment` il training volumetrico usa il trainer nnU-Net standard. La valutazione personalizzata e lo split richiedono invece attenzione. Checkpoint e predizioni originali non sono più disponibili nei percorsi registrati, quindi non è possibile attribuire quantitativamente il divario a una singola scelta.

## Risultato ricostruito

Il log `logs24/nnUNET3D_results.out` contiene 270 casi, fold 0, checkpoint `checkpoint_best.pth`, patch di inferenza 96³, overlap 50%, Gaussian blending, CCA e closing. Riporta **Dice medio raw 0.784477** e **postprocessing 0.815342**. Il guadagno del postprocessing è **0.030865**, cioè 3.09 punti percentuali. Non è il pseudo-Dice online e non è una media di più fold.

Il riferimento confermato dall'utente è [SwinCLNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC12808724/), tabella 5: nnU-Net su BraTS2024 **0.8534 ± 0.028**, su cinque fold per paziente. Il testo dichiara training 128³, batch 4, 300 epoche; inferenza 96³, overlap 50%, Gaussian blending e morfologia. Non pubblica qui split esatti, piani specifici della baseline, politica delle regioni vuote o parametri completi della morfologia. La descrizione di TC è ambigua rispetto alla definizione standard. Queste informazioni non bastano a una replica esatta.

## Differenze confermate

| Punto | Pipeline osservata | Conseguenza |
| --- | --- | --- |
| Aggregazione del risultato | Solo fold 0 | Non è la media dei cinque fold del riferimento; manca una stima della variabilità tra fold |
| Split | KFold casuale sugli identificativi delle scansioni, seed 12345 | Non raggruppa esplicitamente sessioni dello stesso paziente |
| Training patch | 128×160×112 | Configurazione auto-pianificata effettiva; non un crop cubico 128³ |
| Batch size | 2 | Differisce dall'impostazione dichiarata nel riferimento |
| Scheduler | LR del log compatibile con PolyLR su 1000 epoche | Il launcher nativo non usa `snn.epochs` o `snn.lr` del YAML |
| Stato del training | 643 epoche completate visibili nell'archivio | Non dimostra che il job sia terminato lì; non documenta il completamento delle 1000 epoche |
| Checkpoint | `checkpoint_best.pth` | Selezionato dal pseudo-Dice EMA online, non dal Dice medio sui 270 volumi; epoca esatta non ricostruibile senza checkpoint |
| Valutazione | Sliding window 96³ personalizzata | Riprende il formato descritto nel riferimento, ma differisce dall'inferenza nativa con la patch dei piani |
| Dice vuoto-vuoto | 1 nel codice personalizzato | Differisce dall'esclusione tramite NaN della valutazione nnU-Net nativa |

Le differenze di batch, geometria e scheduler sono **disallineamenti**, non prove che una variante debba necessariamente perdere 3.81 punti. Più epoche o una patch diversa non hanno un effetto negativo deterministico.

## Split: verifica diretta, non una supposizione

Ho ricostruito la selezione casuale del fold con `RandomState(12345)` sugli identificativi ordinati: coincide esattamente con `splits_final.json`. Anche i 270 identificativi stampati nel log della validazione storica coincidono esattamente con il fold 0 di quel file. Il log del training registra la creazione automatica del nuovo split.

Raggruppando gli identificativi prima dell'ultimo suffisso di sessione:

- train: 1080 scansioni; validation: 270;
- 188 prefissi compaiono in entrambi;
- 209/270 scansioni di validation hanno un prefisso presente nel training;
- esempio: train `BraTS-GLI-00020-101`, validation `BraTS-GLI-00020-100`.

Se `BraTS-GLI-xxxxx` identifica il paziente, come suggerisce la nomenclatura, questo split contiene sovrapposizione di pazienti. L'identità va confermata con i metadati originali. Il codice standard `crossval_split.py` usa `KFold`, non un raggruppamento per paziente.

**Questo può rendere la validazione ottimistica: non è una spiegazione del risultato inferiore.** È comunque un problema da risolvere prima di dichiarare una replica del protocollo di riferimento. Un nuovo split deve appartenere a un nuovo esperimento: non va sostituito quello usato dai training già avviati.

## Cosa NON risulta disallineato nel training

Ho confrontato il branch `nnUnet_alignment` (commit `e58768c64c51e73924dc1083b4ab5407ca0e1769`) con la pipeline attuale.

Per `family: unet3d`, entrambi usano `run_training` con `trainer_class_name="nnUNetTrainer"`, `plans_identifier="nnUNetPlans"` e configurazione `3d_fullres`. Non passano attraverso il trainer SNN. Il ramo nativo non riceve le opzioni della sezione `snn`.

Nel log è presente una PlainConvUNet 3D a sei stadi, InstanceNorm3D e LeakyReLU; batch Dice disattivo. Non emerge una sostituzione accidentale con la paper_unet 2D o una rimozione della deep supervision nativa. Il wrapper recente per le regioni vuote modifica la valutazione, non gli aggiornamenti dell'ottimizzatore.

La conversione del dataset e il modulo di morfologia sono identici al branch di riferimento. Le funzioni `labelmap_to_regions`, `threshold_probabilities`, `dice_metrics`, `postprocess_regions` e `_canonicalize_regions` sono identiche a livello AST. Il branch di riferimento contiene modifiche successive alla data dei log: è un riferimento di codice, non una fotografia certificata dell'eseguibile di ottobre.

## Valutazione: controlli eseguiti e incertezze residue

### Formula e regioni

Il codice usa WT={1,2,3}, TC={1,3}, ET={3}. Legge tre logits regionali e applica sigmoid, coerente con le tre regioni sovrapposte. La media è per caso e poi per regione; non emerge un errore aritmetico nel Dice. Le prove isolate confermano Dice=1 per predizione perfetta e vuoto-vuoto, Dice=0 per GT vuota con falsi positivi. Non è giustificato cambiare la definizione standard di TC per inseguire il numero del riferimento.

### Regioni assenti

Ho letto i `class_locations` dei 270 campioni preprocessed con l'ambiente Python del progetto: TC è assente in **77** casi, ET in **78**; non mancano metadati. Questo rende importante specificare la convenzione delle regioni vuote.

Il codice personalizzato assegna 1 al vuoto-vuoto; nnU-Net nativo assegna NaN e lo esclude dalla media regionale. A parità di predizioni, aggiungere quei punteggi 1 può solo aumentare o lasciare invariata la media rispetto all'esclusione. Quindi questa specifica differenza non spiega una riduzione rispetto alla metrica nativa. La convenzione del riferimento resta sconosciuta.

Non sono disponibili i conteggi dei falsi positivi della run nnUNet3D originale: i dati di altre reti non possono sostituirli.

### Geometria di inferenza

La patch 96³ è voluta nello script e compare nel log. Cambiarla con quella dei piani serve a misurare la differenza dalla pipeline nativa, non è una correzione già dimostrata. Il codice personalizzato valuta le maschere sulla griglia preprocessed prima dell'esportazione; un confronto con la valutazione delle NIfTI esportate deve usare gli stessi casi, spazi e convenzioni metriche.

### Caricamento dei piani nel branch di riferimento

Il vecchio `main` chiama `get_trainer_from_args`, che legge `nnUNetPlans.json` e `dataset.json` dal preprocessing condiviso. Il parametro `--results-root` viene solo controllato; la directory effettiva del trainer dipende dall'ambiente. Questa è una fragilità se cambiano i metadati condivisi o le variabili di ambiente.

Nel log storico il percorso del checkpoint è quello atteso: non ho prova che abbia caricato un'altra run. Il codice attuale legge i metadati dalla directory del modello e imposta esplicitamente la root dei risultati, risolvendo questa fragilità. Non è dimostrato che fosse la causa del valore 0.815342.

## Bug separato nello script di postprocessing attuale

`postprocess_nnunet.sbatch:63` passa `--min-component-sizes 20, 30, 50`, mentre il parser richiede interi senza virgole. La riproduzione isolata restituisce `SystemExit(2)` prima di leggere volumi. Inoltre il messaggio stampato indica 161/166/50, valori diversi da quelli passati.

**Questo bug non è una spiegazione dimostrata del risultato storico:** lo script di validazione che ha prodotto 0.815342 passa correttamente `161 166 50`. Non ho modificato nessuno dei due script durante l'audit.

## Perché non posso chiudere la diagnosi con un'unica causa

Nei percorsi registrati non sono presenti `checkpoint_best.pth`, `plans.json`, `dataset.json` del modello, summary di valutazione o maschere raw/postprocessed. Le directory di output esistono ma sono vuote. Anche l'utente non conosce un percorso di archivio alternativo. Senza queste predizioni non è possibile ricalcolare il Dice per caso, provare diverse convenzioni o separare l'effetto di CCA e closing su quella stessa run.

Non sarebbe corretto affermare che basta cambiare batch size, soglie o patch per passare sicuramente da 0.8153 a 0.8534.

## Verifica decisiva per una nuova replica

1. Conservare split e metadati del dataset e stabilire un raggruppamento per paziente verificato. Eseguire tutti i cinque fold per confrontare una media di cross-validation.
2. Salvare piani, versione di nnU-Net, commit del codice, configurazione e checkpoint in una directory dedicata. Rendere espliciti batch, LR, numero di aggiornamenti e orizzonte dello scheduler nel trainer nativo; non affidarsi ai campi `snn`.
3. Sullo stesso checkpoint e sugli stessi casi, valutare: pipeline nativa con patch dei piani; custom 96³ raw; custom 96³ con sola CCA; custom 96³ con CCA e closing. Conservare predizioni e probabilità.
4. Calcolare per ogni variante Dice per regione e per caso, separando GT presente/assente; indicare la convenzione vuoto-vuoto. Confrontare anche griglia preprocessed e NIfTI finale.
5. Riportare media e variabilità dei fold. Trattare una divergenza residua come differenza sperimentale finché la causa non viene misurata.

Il controllo iniziale più utile è quindi la pipeline di **valutazione dello stesso modello**, non l'aggiunta di nuove loss o moduli.

## Artefatti

`audit.json`: evidenze e limiti; `training_epochs.json`: metriche del log; `snapshots/`: log, split, metadati disponibili e codice di riferimento; `analyze.py`: verifica riproducibile del protocollo, delle formule e del parser. Le verifiche sulle presenze TC/ET provengono da una lettura separata dei 270 pickle con `.venv/bin/python`, necessaria per la compatibilità con NumPy 2.x.
