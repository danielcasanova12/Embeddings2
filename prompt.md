F:\Git\Embeddings2\extract_embs\extract_contentvec.py
F:\Git\Embeddings2\extract_embs\extract_f0.py
F:\Git\Embeddings2\extract_embs\extract_speaker_embeddings.py
F:\Git\Embeddings2\extract_embs\extract_whisper_embeddings.py

me ajude a criar esta arquitetura corretamente neste projeto
1. Single Embedding MOS Predictor
    
    Audio
    ↓
    Embedding model
    ↓
    MLP
    ↓
    MOS
    
2. Cross-Embedding Interaction
    
    Embeddings
    ↓
    Adapters
    ↓
    emb_i ⊗ emb_j
    ↓
    MLP
    ↓
    MOS
    
3. Multi-Embedding Fusion
    
    Embeddings
    ↓
    Adapters
    ↓
    Concatenation
    ↓
    MLP
    ↓
    MOS
    
    | Model | Embeddings | SRCC |
    | --- | --- | --- |
    | FULL | content + acoustic + texture + timbre + semantics | 0.82 |
    | - semantic | content + acoustic + texture + timbre | 0.71 |
    | - acoustic | content + texture + timbre + semantics | 0.78 |
    | - content | acoustic + texture + timbre + semantics | 0.80 |
    | - texture | content + acoustic + timbre + semantics | 0.79 |
    | - timbre | content + acoustic + texture + semantics | 0.77 |
    
    importance_i = SRCC_full - SRCC_without_i
    
    | embedding | importance |
    | --- | --- |
    | semantic | 0.11 |
    | acoustic | 0.04 |
    | timbre | 0.05 |
    | texture | 0.03 |
    | content | 0.02 |
4. Multi-embedding fusion + reliability weighting + perceptual factor modeling
    
    1- Embeddings
    │
    2- Adapters
    │
    3- Reliability estimation
    │
    4- Factor extraction
    │
    5- MOS predictor
    
5. Transformer Fusion Layer
    
    [whisper]
    Qf0 - 

    Acoustic structure - Content vec

    specker embedding
    ↓
    Transformer encoder
    ↓
    CLS token
    ↓
    MOS


e me ajude a editar os configs que fique mais simples para mim não me perder nos configs quanto eu estiver usando uns 3 datasets e estes 4 tipos de embeddins que eu te passei 
deixe eles de uma forma mais simples de usar por favor, se puder separar exemplo ab1_ de abordagem 1 ex1_ de experimento 1 exemplo aboradagem 1 é single embeding e no ex 1 é usar apenas o wisper 

algo que eu consiga enteder assim me ajude com isso