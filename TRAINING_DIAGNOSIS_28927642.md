# Diagnosi del job 28927642

Il modello è collassato verso lo sfondo. La saturazione del ramo di linear
attention nel primo encoder è la causa più plausibile identificata; non è
ancora dimostrato quale modifica consenta a un nuovo training di convergere.
L'architettura e i pesi salvati non sono stati modificati.

## Evidenze

- Il log contiene 32 epoche completate, tutte con Dice online [0, 0, 0].
  La loss scende da 0.2149 a circa 0.05; ogni epoca costa circa 20 minuti.
- Il rapporto RMS proiezione/input del primo encoder passa da 17.57 nella
  prima epoca a 53.98 nella seconda e circa 60 nelle successive. Il firing
  rate arriva circa a 0.5, mentre i parametri temporali del primo blocco
  diventano quasi immobili. Un firing rate medio di 0.5 da solo non prova
  un funzionamento sano: può nascondere canali sempre accesi o spenti.
- Il solo checkpoint disponibile nel fold è `checkpoint_best.pth`, salvato
  dopo la prima epoca. I gradienti diagnostici di quella epoca sono presenti
  e finiti: non emerge un'interruzione iniziale del grafo autograd.
- Prova CPU/FP32 sul checkpoint, patch [1, 4, 81, 128, 128] del caso reale
  di validazione `BraTS-GLI-00020-100`, centrata su un voxel tumorale:
  frazioni target WT/TC/ET = 9.178%, 1.761%, 1.652%; frazioni predette a
  soglia sigmoid > 0.5 = 0%, 0%, 0%. Logit in [-11.3784, 0].
  Loss totale = 0.18857, BCE = 0.21989, termine Dice = -0.03132.
  La loss nativa è BCE meno soft Dice; non è `1 - hard Dice`.
- Sulla stessa patch, all'ultima slice, 28 dei 32 canali di spike del primo
  encoder sono costanti nello spazio. Attenuando solo la sua proiezione
  ricorrente, senza riaddestrare, i canali costanti diventano 22 con fattore
  0.1 e 8 con fattore 0. La deviazione standard spaziale media degli spike
  passa rispettivamente da 0.03527 a 0.10045 e 0.19956.
  Questa ablation collega il ramo di attention alla perdita di informazione
  spaziale; non dimostra il recupero della segmentazione. Azzerare il ramo
  nel checkpoint produce ancora zero voxel positivi in uscita.

## Interpretazione e prossimo esperimento

L'attention usa statistiche globali delle memorie, poi due proiezioni lineari
senza un limite sull'ampiezza del residuo prima della soma. Le memorie possono
essere stabili mentre questa proiezione domina il segnale locale. La rete
perde contrasto spaziale già nel primo encoder e impara a mantenere negative
le uscite. La prevalenza dello sfondo permette alla BCE di scendere anche
senza predizioni tumorali. Non è una dimostrazione che la loss o il learning
rate siano, da soli, la causa primaria.

Per verificare una correzione, confrontare training nuovi con stesso seed,
stessi batch e stessa architettura: baseline PLIF/SiLIF e attention con ramo
residuo inizialmente attenuato o controllato. Verificare prima l'overfitting di
una patch fissa con tumore, misurando BCE, soft Dice, frazione positiva,
varianza spaziale degli spike e gradienti per blocco. Non basta una loss in
calo. La prova completa di convergenza richiede GPU, qui non disponibile.

## Prove rapide

Per una prova rapida, impostare in `unet_config.yaml` 3 epoche, 5 batch di
training e 2 batch di validazione online per epoca, mantenendo architettura e
patch desiderate e disabilitando le validazioni sui volumi completi. I default
del config principale restano 250/50. I budget contano minibatch per processo/GPU,
indipendentemente da `k`; il learning rate continua ad avanzare per epoca.
Usare un nuovo `run_dir` per ogni esperimento.
