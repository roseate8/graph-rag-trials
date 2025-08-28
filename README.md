# Chunking & Embedding

# Context

This document can act as a tiny guide to how we could/should do chunking and embedding for the RAG we are building. This builds strongly on the Layout-Aware document processing.

Our RAG infrastructure depends on high-quality chunks that preserve structure, semantics, and context across diverse document types. Naive chunking (fixed-size, sentence-based) underperforms on enterprise-scale documents (e.g., annual reports, technical manuals, compliance docs) because it severs context, loses structure, and misinterprets tables:

- Subpar results during retrieval owing to the variety of chunk types
- Loss of context (e.g., headers, footers, references stripped away)
- Poor retrieval relevance (tables, ToCs, and multi-section documents misinterpreted)

To address this, we experimented with layout-aware chunking that uses structural signals (hierarchies, headings, tables, references, table of content) to generate semantically coherent chunks enriched with metadata.

# Problem statement

## What

Build a generalizable, performant chunking service that ingests heterogeneous documents, preserves structure, and emits semantically coherent, metadata-rich chunks for retrieval, reranking, and agent reasoning.

**In scope for the chunking methods discussed below:**

- PDF/DOCX/HTML/MD as inputs; JSON is an internal layout carrier for MD.
- ToC detection and section mapping; table structure preservation & dedupe.
- Context propagation (heading path, references), quality scoring, observability.
- Creating, merging, pruning chunks over the workflow.
- Graph retrieval/agents (to later; this feature provides the substrate).

**Not in scope:**

- Full OCR for scanned images.
- Human-in-the-loop labeling; redaction/PII workflows.

## Metrics/Eval

North star: Answer Quality

- p@k (retrieval)
    - Fraction of the top-k retrieved chunks that are relevant to the query.
    - Offline eval harness; judged by humans or calibrated LLM-judge with spot human audits.
- nDCG@k (normalized Discounted Cumulative Gain)
    - Graded-relevance ranking quality (0 = not, 1 = partially, 2 = fully).
- Recall@k (guardrail)
- Of all relevant chunks, what percent appear in top-k?
- Hallucination / citation-mismatch rate
- LLM token spending
- Latency (P50/P90), Ingestion Throughput
- Ingestion/Embedding failure rate (<1%)

## Common problems I observed

While chunking failures can happen in various forms, the few common ones today surface across seven key dimensions:

1. **Table of Contents (ToC) Detection**
    - Highly variable formats (explicit labels vs. implicit lists).
    - False positives where lists or summary tables mimic ToCs.
    - Multiple candidates in the same document (main, appendices, figures).
    - → Leads to incorrect hierarchy reconstruction and misaligned chunks.
2. **Section Mapping**
    - Section boundaries are often implicit or unclear.
    - Nested hierarchies complicate mapping; page numbers don’t align.
    - Content can span multiple sections or fall outside defined ones.
    - → Results in chunks being mapped to wrong sections, lowering retrieval precision.
3. **Table Processing**
    - Structural complexity (headers, spanning cells, footnotes).
    - Cells lack meaning without surrounding context.
    - Duplication across versions (slight table variations).
    - → Causes partial or duplicate retrieval results that aren’t interpretable.
4. **Context Preservation**
    - Cross-references (“see section 3.2”) break without propagation.
    - Heading chains truncated or missing.
    - Implicit relationships severed at chunk boundaries.
    - → Retrieval outputs lose interpretability; agents hallucinate connections.
5. **Document Variability**
    - Wide spread of formats (PDF, DOCX, HTML, Markdown, scanned OCR).
    - Domain-specific conventions (legal vs. finance vs. technical).
    - Quality issues (low fidelity scans, inconsistent formatting).
    - → A single static chunking method underperforms across enterprise use cases.
6. **Performance and Scale**
    - Layout analysis (tables, ToCs) is computationally expensive.
    - Long reports (100s of pages) strain memory and time budgets.
    - Enterprise ingestion involves thousands of docs.
    - → Latency and infra costs balloon, blocking scalability.
7. **Quality Assurance**
    - Chunks may cut mid-thought, reducing coherence.
    - Metadata often missing or inconsistent.
    - Broken/orphaned chunks flow downstream without validation.
    - → Retrieval quality degrades and trust in the system erodes.

# Chunking Solution

## Preprocessing

Clean content and filter meaningless bits artefacts by removing GLYPH artefacts from PDF extraction, excessive whitespace, navigation patterns (tags like, skip to homepage, toggle navigation, etc.) and noise patterns including only bullets/dashes

## Structural Analysis

There are 4 things we do here specifically:

1. Document structure: Parse document into hierarchical representation. Largely, a part of the Parsing stage. Markdown for textual content with JSON for structural metadata.
2. Identifies Table of Contents for section mapping (explained further below)
3. Extracts heading structure and builds document outline (good to have)
4. Detects tables, figures, lists, key-value pairs and other structured elements

We use both JSON and MD to create chunks. We parse the JSON structure to extract document metadata, tables, and hierarchical elements. For headings, however, JSON can mislead on the same document sometimes. Therefore, use markdown for textual content, headings, lists, and basic formatting.

### Table of contents detection approach

ToC acts like a reference to the document hierarchy that we use to enrich the metadata for every chunk. We go through a multi-faceted analysis to be sure we catch the right table here and do a simple confidence scoring that combine these four signals to determine ToC likelihood:

- Title match confidence. Identify (regex) headings or titles containing "Table of Contents", "Contents", etc.
- Content-based detection: Analyse text patterns indicative of ToC
    - Section/item references with page numbers
    - Sequential or Hierarchical page number listings/patterns
- Structural coherence: Column headers containing "section", "page", etc. Consistent formatting with progressive indentation.
- Position in document (typically near beginning)

### Metadata from doc structure

General information: Document title, Date, doc-id, version, source type (parsed/direct MD input), Author, Page number.

**Folder hierarchy**: Preserve order for hierarchical context for filtering (by folder).

**Document structure**:

- Table of Contents (ToC)
- Titles: H1-level headings and document titles.
- Sections: H2-level headings or major topical areas within the document.
- Subsections: H3+ level headings
- Content: Paragraphs, Figures, Tables, Lists
- Bounding box coordinates

## Chunking Implementation

Broadly, there are only three types of entities to chunk within a document which comes in different forms:

1. Text
2. Tables
3. Figures (TBD)

### Text

Split based on word limits while preserving sentence boundaries and maintains topical coherence. Generally aim for 400-800 word chunks. Avoid splitting across logical boundaries.

Append heading and subheading in each chunk.

Minimum length for a chunk: 200 words, this is handled in [[post-processing]] by merging chunks. Overlap of 10-20 words.

### Table chunking

Identify tables and respective table boundaries from the JSON index. Distinguish between column headers, row headers, and data cells.

Chunking: Treat entire tables as single chunks which will be created in Markdown format here. So preserves table structure in plain text while slicing the document. For LLMs, MD tables > CSV table (community says).

Multi‑page wraps: keep one table_id with continuation metadata. Ideally, we don’t want to break tables across chunks.

- Divide very large tables (>200 rows) into coherent chunks while preserving header context.
- Always repeat headers in every shard; don’t split header & data into different chunks.
- Unit drift across a report is common. Store units per column; never assume global.

### Table-Specific Metadata:

- Table ID
- Column headers (store as array)
- Row headers (good to have)
- Table title: First, search for the title within the JSON structure. If we don’t find it, generate a descriptive title using LLM. (expected: 100-150 tokens per call)
- Table caption and classification: Generate a short summary and classification label using an LLM call (150-200 tokens input, 50 max length output); can use gpt-4.1-nano for this.

# Enhancements

## Deduplication

### Metadata enrichment

### Entity Recognition (TBD)

*For the prototype’s sake, I just created a controlled vocabulary for this domain. But this was a bit hardcoded to find things like EMEA, APAC, ROI, ROE, EBITDA, CAGR, using basic regex functions. Need to discuss if there is a better approach to this extraction of specific entities.*

Identifying key-value pairs within the chunks to store in meta-data. These could be reference specific entities, geographic regions, time periods, product categories, definitions, metrics, properties, dates, etc. which are treated as discrete facts. Ideal for exact-match retrieval.

- regions[] (denormalized convenience): canonical regions in chunk, e.g., ["EMEA","APAC"]
- metrics[]: canonical metric identifiers, e.g., ["roi","roe","ebitda","cagr"]
- time_periods[]: normalized fiscal periods, e.g., ["FY2024","Q4-2024"]
- dates[]: ISO dates present, e.g., ["2024-04-30"]
- products[]: product/category canonical names (if applicable)

### Document Reference Extraction

Add the reference tables, figures, appendices, and other sections in the metadata. Documents often cross-reference like [this](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.wrc4muldhbyd), we need to capture this. Extracting these references helps with routing users and our search to find specific information.

- references.outbound[]: references mentioned in this chunk
    - ref_type: "table" | "figure" | "appendix" | "section"
    - label: e.g., "Table 3", "Figure 7", "Appendix B", "Section 2.1"
    - target_anchor: normalized anchor/id if resolvable (e.g., "table_3")
    - page: int
- references.inbound[]: IDs of objects that point to this chunk (filled post-pass)
    - source_anchor: id/label of the referring object
    - page: int
- section_path: array of headings e.g., ["Financials","Consolidated Results","Revenue"]
- page: int; coords?: {x1,y1,x2,y2} when available

## Post-Processing

### Merge tiny chunks

Check hierarchy compatibility (soft check). if the chunk length is less than 50 tokens. Typically, we’d prefer merging the adjacent chunks. If the adjacent chunks can’t be merged [because max-length can exceed], then we use one of these two methods:

1. Check h1, h2, h3 matches. If we find a strong match, merge.
2. Use Jaccard similarity to calculate the similarity score. Merge if it is above a threshold (>0.6?)

Don’t merge if the small chunk is a table.

### Deduplicate table chunks

Edge case. TBD if arises. For tables, good to have: Identify and filter duplicate or near-duplicate tables using content signatures. Tables may appear multiple times in different formats or with slight variations while parsing through Docling. Rare scenarios with large tables, keep logging to monitor and develop if required. Content Hashing: Table signatures enable efficient deduplication.

# Embedding Generation

I relied on BAAI/bge-small-en-v1.5 for embedding generation after testing a few options. This model works well for business documents and is rather fast.

- Dimension: 384 (compact yet effective) - Smaller embeddings mean less storage space needed
- Performance: Does well on MTEB leaderboard for retrieval tasks; Small size, CPU-friendly runs without needing expensive GPU infra; Works well with English business documents

# Appendix

More reading material here:

1. https://arxiv.org/abs/2411.10541 paper says MD is better for the newer models' prompts.
2. One similar method that can possibly act as a reference source is this public [github repo](https://github.com/aws-samples/layout-aware-document-processing-and-retrieval-augmented-generation?tab=readme-ov-file). But the problem with this approach is that it ends up creating chunks of widely dissimilar sizes, failing at scale. During retrieval, we retrieve single word chunks.