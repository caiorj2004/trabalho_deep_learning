# Classificação Automática de Gênero Musical — GTZAN

Trabalho de Deep Learning desenvolvido em grupo para classificação automática dos 10 gêneros musicais do **GTZAN Dataset**, comparando três famílias de modelos: métodos clássicos com features handcrafted (baseline), rede recorrente Bi-LSTM e rede convolucional 2D (CNN sobre mel-espectrograma).

---

## Visão Geral do Projeto

O objetivo principal foi avaliar e comparar diferentes paradigmas de aprendizado de máquina sobre um problema real de Recuperação de Informação Musical (MIR), adotando um protocolo de avaliação rigoroso que reflete as dificuldades reais do dataset — e não resultados inflados por vazamento de dados.

O projeto está organizado em três módulos independentes, mas que compartilham os mesmos dados, os mesmos folds de validação e o mesmo protocolo de avaliação, tornando todos os resultados diretamente comparáveis.

---

## Protocolo de Avaliação: Validação Cruzada Estratificada com Filtro de Artista

O GTZAN possui problemas documentados que invalidam comparações diretas com a literatura: duplicatas exatas, faixas distorcidas e, o mais crítico, dezenas de músicas do mesmo artista distribuídas por vários gêneros — o que gera *data leakage* severo em splits aleatórios.

Para endereçar isso, adotamos um **split de 3-fold estratificado com filtro de artista**, construído com base nas recomendações de Sturm (2013). O procedimento envolveu:

- Remoção de todas as duplicatas exatas do dataset;
- Remoção de faixas com distorção irreconhecível;
- **Filtro de artista:** para qualquer fold dado, nenhum artista aparece simultaneamente nos conjuntos de treino e de teste;
- Estratificação por gênero para manter a distribuição de classes equilibrada em cada fold.

Após a filtragem, restam **948 das 1.000 faixas** originais. O resultado é um `manifest.csv` com colunas `clip`, `genre`, `fold` e `path`, que indexa todos os arquivos `.npy` de features. A coluna `fold` é a única fonte de verdade para a divisão treino/teste em todos os módulos — isso garante que baseline, LSTM e CNN sejam comparados nas mesmas condições.

> **Por que ~52% e não ~85%?** Resultados de 85–92% reportados em tutoriais e notebooks do Kaggle provêm de splits aleatórios que ignoram a identidade do artista. Com o filtro de artista aplicado corretamente, o teto de acurácia documentado na literatura cai para a faixa de 50–65%, e nossos resultados estão coerentes com isso. Comparações sem filtro de artista não são reproduzíveis nem generalizáveis.

---

## Estrutura do Repositório

```
.
├── README.md                          ← este arquivo
│
├── features_baseline/
│   ├── GTZAN_Leonardo_Araujo_Features_Baseline.ipynb
│   ├── manifest.csv                   ← fonte de verdade dos folds (use sempre este)
│   ├── feature_names.json             ← 34 features por frame + parâmetros de extração
│   ├── baseline_resultados.csv
│   ├── svm_grid_search.csv
│   ├── comparacao_literatura.csv
│   ├── mfcc_seq.zip.001               ← features .npy em 2 partes (> 100 MB total)
│   └── mfcc_seq.zip.002
│
├── lstm/
│   ├── lstm_caio.ipynb
│   ├── lstm_resultados.csv
│   ├── confusion_lstm_fold{1,2,3}.png
│   └── curves_lstm_fold{1,2,3}.png
│
└── cnn/
    ├── gtzan_cnn_sturm.ipynb
    ├── gtzan_cnn_best.pth             ← pesos do melhor modelo (Fold 2)
    ├── training_curves.png
    └── confusion_matrix.png
```

---

## Módulo 1 — Features & Baseline (Leonardo Araújo)

**Notebook:** `features_baseline/GTZAN_Leonardo_Araujo_Features_Baseline.ipynb`

Este módulo realiza duas funções: (1) extrair e persistir todas as features acústicas usadas pelos modelos subsequentes, e (2) estabelecer um baseline clássico com SVM e Random Forest.

### Extração de Features

Cada faixa `.wav` de 30s é segmentada em **10 janelas de 3s** (sem sobreposição). Para cada janela, são calculados **34 coeficientes por frame**:

| Feature | Dimensão |
|---------|----------|
| MFCC | 20 coeficientes |
| Chroma STFT | 12 bandas |
| Zero Crossing Rate (ZCR) | 1 |
| Spectral Centroid | 1 |

Parâmetros: `sr=22050 Hz`, `n_fft=2048`, `hop_length=512`. O resultado de cada clipe é salvo como um array `.npy` de forma `(10, ~130, 34)` — 10 segmentos × ~130 frames × 34 features.

Para o baseline clássico, cada segmento é reduzido a um vetor de médias temporais `(34,)`, e a predição por clipe é feita por **majority voting** sobre os 10 segmentos.

### Resultados do Baseline

| Modelo | Acurácia (clipe) | F1-Macro (clipe) |
|--------|-----------------|-----------------|
| SVM RBF (C=10, γ=scale) | 51.7% ± 4.1% | 49.7% ± 4.7% |
| SVM Tuned (C=5, γ=0.01) | **52.0% ± 5.8%** | **50.4% ± 6.1%** |
| Random Forest | 45.8% ± 5.9% | 44.2% ± 6.2% |

### Como remontar as features `.npy`

```bash
# Linux / macOS
cat mfcc_seq.zip.001 mfcc_seq.zip.002 > mfcc_seq.zip
unzip mfcc_seq.zip   # gera mfcc_seq/ com um .npy por clipe
```

```cmd
# Windows (PowerShell)
copy /b mfcc_seq.zip.001 + mfcc_seq.zip.002 mfcc_seq.zip
```

---

## Módulo 2 — Modelagem Temporal com Bi-LSTM (Caio Rosendo)

**Notebook:** `lstm/lstm_caio.ipynb`

Este módulo avalia se a estrutura temporal das features acústicas carrega informação relevante para discriminar gêneros — isto é, se a *ordem* dos frames ao longo do tempo importa além de sua distribuição estatística.

### Arquitetura

O modelo `MusicLSTM` usa uma **Bi-LSTM** bidirecional com `hidden_size=64`. Em vez de usar apenas o estado oculto do último frame, aplica **Mean Pooling Temporal** sobre todos os 130 estados ocultos — o que produz uma representação mais robusta do trecho completo e menos sensível à posição de eventos musicais.

Para a inferência em nível de clipe, os 10 segmentos de 3s são processados individualmente e suas probabilidades são somadas (**Soft Majority Voting**).

Estratégias de regularização para conter o overfitting característico do GTZAN:
- Injeção de ruído Gaussiano leve nas features durante o treino (`std=0.1`)
- Early stopping com `patience=15` épocas

### Resultados

| Fold | Acurácia (clipe) | F1-Macro |
|------|-----------------|----------|
| Fold 1 | 51.7% | 49.8% |
| Fold 2 | 51.7% | 51.6% |
| Fold 3 | 53.3% | 53.7% |
| **Média** | **52.3%** | **51.7%** |

A Bi-LSTM alcança desempenho muito próximo ao SVM Tuned, mas com característica distinta: apresenta bom balanceamento entre classes (F1-macro competitivo), porém é sensível ao ruído frame a frame — gêneros espectralmente próximos como Rock e Blues permanecem os maiores desafios.

---

## Módulo 3 — CNN sobre Mel-Espectrograma (Leonardo Borges)

**Notebook:** `cnn/gtzan_cnn_sturm.ipynb`

Este módulo trata o problema de classificação de gênero como **classificação visual de texturas espectrais**: cada faixa é convertida em um mel-espectrograma 2D e fornecida a uma CNN.

### Feature: Mel-Espectrograma

| Parâmetro | Valor |
|-----------|-------|
| Sample rate | 22.050 Hz |
| N° bandas mel | 128 |
| FFT size | 2048 |
| Hop length | 512 |
| Frames (fixos) | 1.292 |
| Forma do tensor | `(1, 128, 1292)` |

Os espectrogramas são pré-extraídos e cacheados como `.npy` antes do treino.

### Arquitetura

```
Input: (1, 128, 1292)

Bloco 1: Conv2d(1→32,  3×3) → BN → ReLU → MaxPool(2×4) → Dropout2d(0.30)
Bloco 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×4) → Dropout2d(0.30)
Bloco 3: Conv2d(64→128,3×3) → BN → ReLU → MaxPool(2×4) → Dropout2d(0.30)
Bloco 4: Conv2d(128→256,3×3)→ BN → ReLU → AdaptiveAvgPool(4×4) → Dropout2d(0.30)

Classifier:
  Linear(4096 → 512) → ReLU → Dropout(0.55)
  Linear(512  → 128) → ReLU → Dropout(0.35)
  Linear(128  → 10)
```

**Parâmetros totais:** ~2.4M

**Data augmentation:** SpecAugment com `FrequencyMasking(20)` e `TimeMasking(30)`.

**Treinamento:** AdamW + OneCycleLR (max_lr=1e-3, warm-up nos primeiros 30%), `label_smoothing=0.1`, early stopping com `patience=15`.

### Resultados

| Fold | Acurácia | F1-Macro |
|------|----------|----------|
| Fold 1 | ~52% | ~51% |
| Fold 2 | ~57% | ~56% |
| Fold 3 | ~53% | ~52% |
| **Média** | **~54% ± 2%** | **~53% ± 2%** |

Acurácia por classe no melhor fold (Fold 2):

| Gênero | Acurácia |
|--------|----------|
| pop | 90% |
| classical | 88% |
| metal | 83% |
| hiphop | 67% |
| jazz | 62% |
| reggae | 59% |
| country | 47% |
| blues | 35% |
| disco | 34% |
| rock | 18% |

A variação alta entre rock/blues/disco e pop/classical/metal reflete sobreposição espectral real entre gêneros historicamente relacionados — comportamento esperado e documentado na literatura.

---

## Tabela Comparativa Final

| Modelo | Paradigma | Acurácia Média (clipe) | F1-Macro Médio | Responsável |
|--------|-----------|----------------------|----------------|-------------|
| Random Forest | Clássico + features tabulares | 45.8% | 44.2% | Leonardo Araújo |
| SVM Tuned | Clássico + features tabulares | 52.0% | 50.4% | Leonardo Araújo |
| **Bi-LSTM** | Sequencial (temporal) | **52.3%** | **51.7%** | Caio Rosendo |
| **CNN 2D** | Visual (espectrograma) | **~54%** | **~53%** | Leonardo Borges |

**CNN foi o modelo mais forte do grupo.** Representar cada faixa como textura visual (espectrograma) mostrou-se mais estável do que modelar a variação temporal bruta frame a frame — a CNN captura estruturas de frequência e tempo de forma hierárquica e é menos sensível ao ruído de curto prazo que afeta a LSTM.

A LSTM, por sua vez, supera o Random Forest e empata com o SVM Tuned, sendo um resultado razoável: a ordem temporal dos frames adiciona algum poder discriminativo, mas o ruído inerente ao GTZAN limita o ganho.

---

## Como Reproduzir (ordem recomendada)

```
1. Execute features_baseline/ → gera manifest.csv e mfcc_seq/
2. Execute cnn/               → usa manifest.csv para os folds; extrai seus próprios spectrogramas
3. Execute lstm/              → usa manifest.csv + mfcc_seq/
```

Todos os notebooks são projetados para rodar no **Google Colab** com GPU T4. Siga as instruções no início de cada notebook para montar o Google Drive e ajustar os caminhos.

---

## Dependências

```bash
pip install torch torchaudio librosa scikit-learn numpy pandas \
            matplotlib seaborn plotly tqdm
apt-get install -y ffmpeg   # necessário para alguns .wav com formato não-padrão
```

---

## Referências

- Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals*. IEEE Transactions on Speech and Audio Processing, 10(5), 293–302.
- Sturm, B. L. (2013). *The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use*. arXiv:1306.1461.
- Foleiss, J. H., & Tavares, T. F. (2020). *Texture selection for automatic music genre classification*. Applied Soft Computing, 89, 106127.
- Park, D. S., et al. (2019). *SpecAugment: A simple data augmentation method for automatic speech recognition*. Interspeech 2019.
- Kereliuk, C., Sturm, B. L., & Larsen, J. (2015). *Deep learning and music adversaries*. IEEE Transactions on Multimedia, 17(11), 2059–2071.
