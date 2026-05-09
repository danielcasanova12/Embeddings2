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
    ↓
    Transformer encoder
    ↓
    CLS token
    ↓
    MOS


Isso já esta tudo implementado?
os sctipts de extração de embeddins estão implementados e corretamente? 
 Eu estava pensando aqui como esta agora como esta pensado o input exemplo entra assim