# Graph-RAG Document Processing Workflow

## Current Architecture Diagram

```mermaid
graph TD
    %% Input Layer
    A[Input Directory] --> B[DocumentProcessor.process_all_documents]
    
    %% File Discovery
    B --> C[TextPreprocessor.discover_input_files]
    C --> D{File Types}
    D --> |PDF, DOCX, HTML, MD, TXT| E[For Each File]
    
    %% Single Document Processing Flow
    E --> F[DocumentProcessor.process_document]
    
    %% Step 1: Preprocessing
    F --> G[TextPreprocessor.preprocess_file]
    G --> |Clean Content| H[Create DocumentMetadata]
    
    %% Step 2: Structure Analysis  
    H --> I[TableOfContentsDetector.detect_toc]
    I --> |TOC Entries| J[Create DocumentStructure]
    
    %% Step 3: Intelligent Chunking
    J --> K[TextChunker.process]
    K --> L{Dual Format Support}
    L --> |MD Primary| M[Markdown Content Analysis]
    L --> |JSON Enhanced| N[JSON Bounding Boxes/Tables/Figures]
    M --> O[DocumentElement Classification]
    N --> O
    
    %% TextChunker Internal Process
    O --> P[Structure-Aware Chunking]
    P --> |600-800 words| Q[Word-Based Splitting]
    Q --> |10-20 word overlap| R[Chunk Creation]
    R --> |Preserve boundaries| S[Element Integration]
    
    %% Output Processing
    S --> T[Convert to Chunk Objects]
    T --> U[Add TextChunker Metadata]
    U --> V[Return List of Chunks]
    
    %% Final Output
    V --> W[Save Processing Summary]
    W --> X[Output Directory]
    
    %% Component Classes
    subgraph "Core Components"
        Y[DocumentProcessor]
        Z[TextPreprocessor] 
        AA[TableOfContentsDetector]
        BB[TextChunker]
    end
    
    subgraph "Data Models"
        CC[DocumentMetadata]
        DD[DocumentStructure] 
        EE[Chunk]
        FF[ChunkMetadata]
        GG[DocumentElement]
        HH[BoundingBox]
    end
    
    subgraph "Processors Available"
        II[BaseProcessor - Abstract]
        JJ[BaseFileProcessor - Abstract]  
        KK[TextPreprocessor - File Discovery/Cleaning]
        LL[TableOfContentsDetector - TOC Analysis]
        MM[TextChunker - Intelligent Chunking]
        NN[LLMUtils - Available but unused]
    end
    
    %% Style
    classDef used fill:#90EE90
    classDef unused fill:#FFB6C1
    classDef core fill:#87CEEB
    
    class Y,Z,AA,BB,CC,DD,EE,FF,GG,HH,II,JJ,KK,LL,MM used
    class NN unused
    class A,X core
```

## Architecture Analysis

### ✅ **Currently Used Components:**
1. **DocumentProcessor** - Main orchestrator
2. **TextPreprocessor** - File discovery and content cleaning  
3. **TableOfContentsDetector** - Document structure analysis
4. **TextChunker** - Intelligent chunking with dual format support
5. **Data Models** - Complete model hierarchy (Chunk, DocumentMetadata, etc.)
6. **DocumentElement & BoundingBox** - Structure definitions

### ❓ **Potentially Redundant/Unused:**
1. **LLMUtils** - Created but not integrated into main workflow
2. **BaseProcessor/BaseFileProcessor** - Abstract classes, good for extensibility but could be simplified

### 🔧 **Missing Core Functions:**
1. **JSON/MD File Pair Processing** - TextChunker supports it, but main workflow only processes single files
2. **Embeddings Generation** - Chunk model has embedding field but no processor
3. **Entity Extraction** - ChunkMetadata has fields (regions, metrics, dates) but no extraction logic
4. **Reference Resolution** - Models support inbound/outbound refs but no processor
5. **Table-Specific Processing** - ChunkMetadata has table fields but limited table processing

### 🎯 **Workflow Efficiency:**
- **Good**: Linear, clear flow from input to chunked output
- **Good**: Proper separation of concerns between components
- **Good**: TOC detection integrated with chunking
- **Missing**: Parallel processing capabilities for multiple files
- **Missing**: Validation/quality checks on chunks

### 🚨 **Key Findings:**
1. **Main workflow is complete and functional** ✅
2. **TextChunker dual format support is ready but not fully utilized** ⚠️
3. **LLMUtils created but not integrated** ⚠️
4. **Advanced chunk metadata fields defined but not populated** ⚠️