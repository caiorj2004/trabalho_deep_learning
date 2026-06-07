# Features & Baseline — Leonardo Araujo

Parte de **extração de features** e **baseline clássico** do trabalho de classificação
de gêneros musicais no GTZAN.

## Estratificação (importante)

Toda a divisão treino/teste segue **exclusivamente** os folds estratificados de
Juliano Foleiss — [`gtzan_sturm_filter_3folds_stratified`](https://github.com/julianofoleiss/gtzan_sturm_filter_3folds_stratified),
que aplicam as recomendações de **Sturm (2013)**: remoção de duplicatas/faixas
distorcidas e **filtro por artista** (nenhum artista aparece em treino e teste ao mesmo
tempo). Restam **948** das 1000 faixas. Use sempre a coluna `fold` do `manifest.csv`
para manter baseline, CNN e LSTM diretamente comparáveis, sem *data leakage*.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `GTZAN_Leonardo_Araujo_Features_Baseline.ipynb` | Notebook completo: splits, baseline (SVM/RF + tuning) e extração de features |
| `manifest.csv` | clipe → gênero → **fold** → caminho do `.npy` |
| `feature_names.json` | nomes das 34 features por frame + parâmetros (sr, hop, n_fft, segmentos) |
| `baseline_resultados.csv` | resultados do baseline clássico (3-fold, por clipe) |
| `svm_grid_search.csv` | busca em grade do SVM (C × gamma) |
| `comparacao_literatura.csv` | comparação com resultados publicados no GTZAN |
| `mfcc_seq.zip.001` / `mfcc_seq.zip.002` | features `.npy` por clipe (divididas por causa do limite de 100 MB do GitHub) |

## Como remontar e extrair as features `.npy`

O zip foi dividido em 2 partes. Junte-as e descompacte:

**Windows (PowerShell ou CMD):**
```cmd
copy /b mfcc_seq.zip.001 + mfcc_seq.zip.002 mfcc_seq.zip
```

**Linux / macOS:**
```bash
cat mfcc_seq.zip.001 mfcc_seq.zip.002 > mfcc_seq.zip
```

Depois descompacte `mfcc_seq.zip` → gera a pasta `mfcc_seq/` com um `.npy` por clipe.

Cada `.npy` tem forma **`(10, ~130, 34)`** = 10 segmentos de 3 s × frames × 34 features
(MFCC 20 + Chroma 12 + ZCR 1 + Spectral Centroid 1).

## Como usar (colegas)

```python
import numpy as np, pandas as pd
mani = pd.read_csv("manifest.csv")               # clip, genre, fold, path
seq  = np.load("mfcc_seq/blues.00000.npy")       # (10, 130, 34)

# Validação cruzada 3-fold (mesma estratificação):
for k in (1, 2, 3):
    treino = mani[mani.fold != k]
    teste  = mani[mani.fold == k]
    # ... treinar no 'treino', avaliar no 'teste', agregar segmentos por clipe (majority voting)
```

- **Caio (LSTM):** usar as sequências `(frames, 34)` de cada segmento como entrada temporal.
- **Borges (CNN):** alinhar o split dos mel-spectrogramas pela coluna `fold` do `manifest.csv`.

## Baseline (3-fold artist-filtered, métrica por clipe)

| Modelo | Acurácia | F1-macro |
|---|---|---|
| SVM (RBF, C=10, γ=scale) | 0.517 | 0.497 |
| SVM tuned (C=5, γ=0.01) | 0.520 | 0.504 |
| Random Forest | 0.458 | 0.442 |

> Os ~0.52 são esperados sob o protocolo *artist-filtered* (Sturm/Foleiss). Resultados
> de ~0.85–0.92 vistos em notebooks de Kaggle vêm de split aleatório **com vazamento de
> artista** e não são comparáveis. Detalhes e referências no notebook (seção 2.2).
