# Modelagem Temporal com LSTM — Classificação de Gêneros Musicais (GTZAN)

Este diretório contém a implementação da arquitetura **Bi-LSTM** para a classificação de gêneros musicais, focando na modelagem temporal das sequências de áudio. Esta etapa é de responsabilidade de **Caio Rosendo** e consome as features pré-processadas na etapa de *Baseline* (Araújo).

---

## 1. Visão Geral

O objetivo deste pipeline é avaliar a capacidade de uma Rede Neural Recorrente em extrair padrões sequenciais, frame a frame, a partir de coeficientes acústicos (como os MFCCs). Para garantir o rigor científico e evitar *data leakage* (vazamento de dados), o treinamento utiliza estritamente o `manifest.csv` gerado na etapa anterior, que aplica o filtro de artistas (método Foleiss/Sturm) na divisão dos 3 *folds* de validação cruzada.

---

## 2. Estrutura de Arquivos Necessária

Para executar o notebook corretamente, certifique-se de que a etapa `features_baseline` foi concluída e descompactada. O notebook rastreará automaticamente os seguintes arquivos no ambiente:

* `manifest.csv`: Contém os metadados e as marcações de qual *fold* (1, 2 ou 3) cada segmento de 3s pertence.
* Arquivos `.npy` (ex: `blues.00000.npy`): Matrizes contendo os 130 frames temporais e 34 features acústicas extraídas de cada segmento de áudio.

---

## 3. Arquitetura do Modelo e Mitigação de Overfitting

O modelo principal (`MusicLSTM`) foi projetado com técnicas avançadas para mitigar o forte *overfitting* intrínseco ao dataset GTZAN:

* **Bi-LSTM:** Rede bidirecional (processa o áudio em ambas as direções temporais) com `hidden_size=64`, permitindo que o contexto passado e futuro influenciem a classificação.
* **Mean Pooling Temporal:** Em vez de utilizar apenas o estado oculto do último milissegundo do áudio, o modelo calcula a média global de todos os 130 estados ocultos, criando uma "assinatura" robusta do trecho inteiro.
* **Data Augmentation 1D:** Injeção de ruído Gaussiano leve (`std=0.1`) nas features durante o treinamento para forçar a rede a aprender a "vibe" do gênero e não decorar o timbre exato.
* **Soft Majority Voting:** Na inferência de uma música completa (30s), os 10 segmentos de 3s do áudio são processados e suas probabilidades (*logits*) são somadas para decidir o gênero final por consenso.

---

## 4. Como Executar

1. Suba o notebook `.ipynb` no **Google Colab** (recomendado devido à integração com o Drive e disponibilidade de GPU).
2. Certifique-se de que os `.npy` descompactados e o `manifest.csv` estão disponíveis na raiz do Colab ou no seu Google Drive montado.
3. Altere o tipo de ambiente para **GPU** (`Ambiente > Alterar tipo de ambiente de execução > T4 GPU`).
4. Execute as células sequencialmente. O notebook treinará os 3 folds, aplicará o *Early Stopping*, gerará gráficos interativos via Plotly (matrizes de confusão e curvas de loss) e exportará os relatórios finais e os pesos do modelo (`.pth`).

---

## 5. Resultados e Comparativo Final

Os testes com validação cruzada apresentaram os seguintes resultados médios, posicionando a modelagem temporal dentro do limite ideal esperado para dados submetidos ao filtro de artistas:

| Modelo / Abordagem | Acurácia Média (Clipe) | F1-Score Macro | Observação Técnica |
| --- | --- | --- | --- |
| **Random Forest** (Araújo) | 45.8% | 44.2% | Subajustado no baseline. |
| **Bi-LSTM Temporal** (Caio) | **51.0%** | **50.5%** | **Muito próximo do SVM.** Bom balanceamento de classes, porém extremamente sensível ao ruído caótico do frame a frame (confunde Rock/Blues). |
| **SVM Tuned** (Araújo) | 52.0% | 50.4% | Melhor baseline clássico. Beneficiou-se da redução da dimensão do tempo por meio de médias tabulares. |
| **CNN 2D Espacial** (Borges) | ~54.0% | ~53.0% | **Vencedor do Grupo.** A representação visual (espectrograma como textura) provou ser mais estável que a variação temporal bruta para este dataset específico. |

---

**Dependências Principais:** `torch`, `numpy`, `pandas`, `scikit-learn`, `plotly`
