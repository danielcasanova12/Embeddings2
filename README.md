# 🎙️ Projeto MOS Predictor: Multi-Embedding Fusion

Este repositório fornece uma estrutura completa para predição de MOS (Mean Opinion Score) usando fusão de múltiplos embeddings de áudio (Semântico, Acústico, Timbre e Pitch). Siga este guia passo a passo para configurar e rodar seus experimentos.

---

## 🚀 Guia Passo a Passo

### 1. Preparação do Dataset (Extração)
O primeiro passo é extrair os 4 tipos de embeddings dos seus áudios `.wav` e gerar um CSV unificado.

Rode o script `extract_all.py`:
```bash
python extract_all.py -c "seu_dataset.csv" -b "F:/Git/Embeddings2" -i "pasta_dos_audios" -col "coluna_com_nome_do_arquivo" -o "embeddings/seu_dataset"
```
**Resultado:** 
- Uma pasta `embeddings/seu_dataset/` contendo arquivos `.pt` (whisper, contentvec, speaker, f0).
- Um novo arquivo `seu_dataset_with_embs.csv` com as colunas `whisper_path`, `contentvec_path`, `speaker_path` e `f0_path` apontando para os arquivos gerados.

---

### 2. Configuração do Dataset
Agora, avise o sistema onde está o CSV que você acabou de gerar. Edite o arquivo em `configs/datasets/seu_dataset.yaml`:

```yaml
datasets:
  name: "nome_do_dataset"
  train:
    metadata_path: "caminho/para/seu_dataset_train_with_embs.csv"
    target_column: "mos"
  val:
    metadata_path: "caminho/para/seu_dataset_val_with_embs.csv"
    target_column: "mos"
  test:
    metadata_path: "caminho/para/seu_dataset_test_with_embs.csv"
    target_column: "mos"
```

---

### 3. Rodando os Experimentos
Você pode rodar um experimento isolado ou todos de uma vez.

**Para rodar TODOS os experimentos de todas as abordagens:**
```bash
python run_all.py -d configs/datasets/seu_dataset.yaml
```

**Para rodar apenas uma abordagem específica (ex: ab3):**
```bash
python run_all.py -d configs/datasets/seu_dataset.yaml -p "configs/experiments/ab3/*.yaml"
```

---

## 📊 Acompanhamento de Resultados

O `run_all.py` gerencia automaticamente dois arquivos CSV na raiz do projeto para você não se perder:

1.  **`train_results.csv`**: Um "mapa" dos seus treinos.
    - `experiment_name`: Nome do teste.
    - `config_path`: Qual arquivo `.yaml` foi usado.
    - `checkpoint_path`: **ONDE O MODELO FOI SALVO** (caminho do `.ckpt`).
    - `training_time_sec`: Quanto tempo demorou o treino.
    - `status`: Se terminou com sucesso ou erro.

2.  **`test_results.csv`**: O placar de performance (Evaluation).
    - `test_mse`: Erro médio quadrático no teste.
    - `test_pearson` / `test_spearman`: Métricas de correlação (quanto maior, melhor).
    - `val_spearman`: Melhor métrica atingida durante a validação.

---

## 📂 Organização dos Experimentos

As configurações estão separadas em 5 abordagens fundamentais:

- **AB1 (Single):** Testa cada embedding sozinho (Whisper, F0, etc).
- **AB2 (Cross):** Testa a interação entre **pares** de embeddings (ex: Whisper + F0).
- **AB3 (Ablation Concat):** Estudo de importância via concatenação. Rodamos o "All" e depois "All-minus-one" para medir a perda de cada componente.
- **AB4 (Reliability):** Fusão dinâmica onde o modelo aprende a dar mais peso aos embeddings mais confiáveis para aquele áudio.
- **AB5 (Transformer):** Fusão profunda usando um Transformer Encoder, também seguindo a lógica de ablação.

---

## 🛠️ Requisitos Técnicos
- **Extração:** `transformers`, `speechbrain`, `torchcrepe`, `librosa`.
- **Treino:** `pytorch-lightning`, `omegaconf`, `pandas`.

---

## 📝 Dicas de Senior
- **Logs:** Se ativado no `model.yaml`, você pode acompanhar as curvas de perda em tempo real via **WandB**.
- **Checkpoints:** Os modelos são salvos na pasta `checkpoints/`. Use o caminho guardado no `train_results.csv` para carregar o modelo em produção ou inferência.
- **Adicionando Novos Embeddings:** Basta criar o script de extração, adicionar a coluna no CSV via `extract_all.py` e referenciar o nome da nova coluna no seu arquivo de experimento `.yaml`.
