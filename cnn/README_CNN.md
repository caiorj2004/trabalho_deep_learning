# Classificação de Gênero Musical com CNN — GTZAN Dataset

Implementação de uma **Convolutional Neural Network (CNN)** para classificação automática de 10 gêneros musicais usando o GTZAN Dataset, com validação cruzada estratificada com filtro de artista (Sturm, 2013).

---

## Resultados

| Fold | Acurácia | F1-Macro |
|------|----------|----------|
| Fold 1 | ~52% | ~51% |
| Fold 2 | ~57% | ~56% |
| Fold 3 | ~53% | ~52% |
| **Média** | **~54% ± 2%** | **~53% ± 2%** |

### Acurácia por classe (Fold 2 — melhor fold)

| Classe | Acurácia |
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

> Classes espectralmente próximas (blues/rock/country e disco/pop/hiphop) são as mais difíceis de separar — comportamento esperado para CNN simples no GTZAN.

---

## Dataset

**GTZAN** — 1.000 arquivos `.wav` de 30 segundos, 10 gêneros × 100 músicas cada.

```
genres_original/
  blues/        blues.00000.wav ... blues.00099.wav
  classical/
  country/
  disco/
  hiphop/
  jazz/
  metal/
  pop/
  reggae/
  rock/
```

**Folds de validação:** [gtzan_sturm_filter_3folds_stratified](https://github.com/julianofoleiss/gtzan_sturm_filter_3folds_stratified) — garante que músicas do mesmo artista não aparecem simultaneamente em treino e teste (filtro de Sturm).

---

## Arquitetura

```
Input: (1, 128, 1292)  ← mel-espectrograma mono-canal

Bloco 1: Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2×4) → Dropout2d(0.3)
Bloco 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×4) → Dropout2d(0.3)
Bloco 3: Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2×4) → Dropout2d(0.3)
Bloco 4: Conv2d(128→256, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4) → Dropout2d(0.3)

Classifier:
  Linear(4096 → 512) → ReLU → Dropout(0.55)
  Linear(512 → 128)  → ReLU → Dropout(0.35)
  Linear(128 → 10)
```

**Parâmetros totais:** ~2.4M

---

## Pipeline de Features

Cada arquivo `.wav` é convertido para um **Mel-Espectrograma** em dB:

| Parâmetro | Valor |
|-----------|-------|
| Sample rate | 22.050 Hz |
| Duração | 30s |
| N° bandas mel | 128 |
| FFT size | 2048 |
| Hop length | 512 |
| Frames fixos | 1292 |

Os espectrogramas são pré-extraídos e salvos como `.npy` antes do treino (~5× mais rápido por época).

**SpecAugment** aplicado no treino:
- `FrequencyMasking(freq_mask_param=20)`
- `TimeMasking(time_mask_param=30)`

---

## Treinamento

| Hiperparâmetro | Valor |
|----------------|-------|
| Épocas | 100 (com early stopping) |
| Batch size | 64 |
| Otimizador | AdamW |
| Learning rate | OneCycleLR (max=1e-3) |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Early stopping | patience=15 épocas |

**Scheduler:** `OneCycleLR` com warm-up nos primeiros 30% das épocas e cosine decay — permite convergência mais rápida com poucas épocas.

---

## Como Usar (Google Colab)

### 1. Pré-requisitos

Suba o GTZAN no Google Drive na estrutura:
```
MyDrive/
  gtzan/
    genres_original/    ← pasta com os 10 subdiretórios
    outputs/            ← criada automaticamente
```

### 2. Configurar o caminho

Na célula de configuração do notebook, ajuste:
```python
GTZAN_ROOT = Path('/content/drive/MyDrive/gtzan/genres_original')
OUTPUT_DIR = Path('/content/drive/MyDrive/gtzan/outputs')
```

### 3. Executar

Rode as células em ordem. O notebook irá:
1. Montar o Google Drive
2. Baixar os arquivos de folds automaticamente
3. Copiar o dataset para o disco local (mais rápido que Drive)
4. Pré-extrair todos os mel-spectrogramas para cache `.npy`
5. Treinar os 3 folds com checkpoints automáticos no Drive
6. Gerar curvas de treino e matrizes de confusão
7. Salvar o melhor modelo em `outputs/gtzan_cnn_best.pth`

> **GPU recomendada:** T4 ou superior. No Colab: *Runtime → Change runtime type → T4 GPU*

### 4. Inferência

```python
predicted_genre, probabilities = predict_genre('musica.wav', final_model)
print(f'Gênero: {predicted_genre}')
# {'blues': 0.03, 'classical': 0.01, 'metal': 0.89, ...}
```

---

## Dependências

```
torch >= 2.0
torchaudio >= 2.0
librosa >= 0.10
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

Instalação:
```bash
pip install librosa torch torchvision torchaudio scikit-learn matplotlib seaborn tqdm
apt-get install -y ffmpeg  # necessário para arquivos .wav com formato não-padrão
```

---

## Estrutura do Repositório

```
gtzan-cnn/
  gtzan_cnn_v4.ipynb      ← notebook principal
  README.md
  outputs/                ← gerado ao treinar
    checkpoint_fold1.pth
    checkpoint_fold2.pth
    checkpoint_fold3.pth
    gtzan_cnn_best.pth
    training_curves.png
    confusion_matrix.png
```

---

## Referências

- **Dataset GTZAN:** Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals*. IEEE Transactions on Speech and Audio Processing, 10(5), 293–302.
- **Filtro de Sturm:** Sturm, B. L. (2013). *The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use*. arXiv:1306.1461.
- **Folds estratificados:** Foleiss, J. H., & Tavares, T. F. (2020). *Texture selection for automatic music genre classification*. Applied Soft Computing, 89, 106127.
- **SpecAugment:** Park, D. S., et al. (2019). *SpecAugment: A simple data augmentation method for automatic speech recognition*. Interspeech 2019.
