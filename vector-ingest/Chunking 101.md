# Untitled

# Chunking & Embedding

This can be treated as a child doc of [[WIP] PRD: Vector store for storage & retrieval of text data](https://docs.google.com/document/d/1HlcEyWZNF9GXeeg6CCvb5TGj_5JGpXIEPEjucZkKacU/edit?usp=sharing).

[Context](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.d0rev0ehwwgi)

[Problem statement](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.yuae4qdzsyk4)

[What](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.ufgwmir45ijr)

[Metrics/Eval](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.l6gcg3svwe38)

[Common problems I observed](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.v373u46srum0)

[Chunking Solution](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.cht1yhycbn59)

[Preprocessing](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.m7f4lct2qw4l)

[Structural Analysis](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.j44sex61fvt)

[Table of contents detection approach](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.tafhk6t38w28)

[Metadata from doc structure](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.1ufr9dydi6u4)

[Chunking Implementation](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.3wjbql6hahmz)

[Text](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.rlh1addyndrj)

[Table chunking](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.55rs86j8dicm)

[Table-Specific Metadata:](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.tt11qatckyt2)

[Enhancements](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.7eh3kv7zc4od)

[Deduplication](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.cc1gtxa4serr)

[Metadata enrichment](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.2cq5nk48uizu)

[Entity Recognition (TBD)](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.vpu3m3d6ir7w)

[Document Reference Extraction](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.wrc4muldhbyd)

[Post-Processing](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.5vbn2514jm1i)

[Merge tiny chunks](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.co9y6wv6pzkv)

[Deduplicate table chunks](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.24di297w2bk9)

[Embedding Generation](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.q8ybvvq55yhh)

[Appendix](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.p2q2am1eng5k)

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

**Not in scope:**

- Full OCR for scanned images.
- Human-in-the-loop labeling; redaction/PII workflows.
- Graph retrieval/agents (to support later; this feature provides the substrate).

## Metrics/Eval

North star: Answer Quality

- Latency (P50/P90)
- Ingestion Throughput
- p@k (retrieval)
    - Fraction of the top-k retrieved chunks that are relevant to the query.
    - Offline eval harness; judged by humans or calibrated LLM-judge with spot human audits.
- nDCG@k (normalized Discounted Cumulative Gain)
    - Graded-relevance ranking quality (0 = not, 1 = partially, 2 = fully).
- Recall@k (guardrail)
- Of all relevant chunks, what percent appear in top-k?
- LLM token spending
- Ingestion/Embedding failure rate (<1%)

## Common problems I observed

Cursor summary.

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

Clean content and filter meaningless bits of artefacts by removing GLYPH artefacts from PDF extraction, excessive whitespace, navigation patterns (tags like, skip to homepage, toggle navigation, etc.) and noise patterns including only bullets/dashes.

## Structural Analysis

There are 4 things we do here specifically:

1. Document structure: Parse document into hierarchical representation. Largely, a part of the Parsing stage. Markdown for textual content with JSON for structural metadata.
2. Identifies Table of Contents for section mapping (explained further below)
3. Extracts heading structure and builds document outline (good to have)
4. Detects tables, figures, lists, key-value pairs and other structured elements

We use both JSON and MD to create chunks. We parse the JSON structure to extract document metadata, tables, and hierarchical elements. For headings, however, JSON can mislead on the same document sometimes. Therefore, use markdown for textual content, headings, lists, and basic formatting.

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
- Footnotes

## Chunking Implementation

Broadly, there are only three types of entities to chunk within a document which comes in different forms:

1. Text
2. Tables
3. Figures (TBD)

### Text

- Sentence Boundary Preservation: Rather than splitting text at arbitrary points, we should identify sentence boundaries using regex patterns r'(?<=[.!?])\s+(?=[A-Z])', ensuring chunks maintain semantic coherence. Avoid splitting across logical boundaries.
- Append heading and subheading at the start of every chunk.
    - This heading and subheadings should also be a part of metadata, separately.
    - PS. When we add it inside the chunk, it increases the similarity score on the Vector DB search. The metadata fields enable filtering and enhance search for BM25 searches.
- Split based on word limits while preserving sentence boundaries and maintains topical coherence. Generally aim for 600-800 word chunks. Overlap of 10-20 words.
- Minimum length for a chunk: 200 words. If we end up creating small chunks in this stage (bound to happen), it will be handled in [post-processing](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.5vbn2514jm1i) by merging chunks.

### Table chunking

Identify tables and respective table boundaries from the JSON index. Distinguish between column headers, row headers, and data cells. We have the MD files as well to use directly for the chunk.

Chunking: Treat entire tables as single chunks which will be created in Markdown format here. So preserves table structure in plain text while slicing the document. For LLMs, MD tables > CSV table (community says).

Multi‑page wraps: keep one table_id with continuation metadata. Ideally, we don’t want to break tables across chunks.

- Divide very large tables (>200 rows) into coherent chunks while preserving header context.
- Always repeat headers in every shard; don’t split header & data into different chunks.
- Unit drift across a report is common. Store units per column; never assume global.
- We also need to append the Table title, summary to every table chunk we create
    - First, search for the title within the JSON structure. If we don’t find it, generate a descriptive title using LLM. (expected: 100-150 tokens per call)
    - Generate a short summary and the classification label using an LLM call (150-200 tokens input, 50 max length output); can use gpt-4.1-nano for this.
    - Add the classification label below in the metadata field.

### Table-Specific Metadata:

- Table ID
- Column headers (store as array)
- Row headers (good to have)
- Table shape {"rows": 27, "cols": 3}
- Table classification label
    - financial_statement, summary_kpi, fact_table, dimension_table, timeseries_table, pivot_matrix, list_table, form_like
    - Do not store LLM free-text tags as filterable fields without type + normalization.

# Enhancements

## Post-Processing

### Chunk cleanup

In my trials, I had to do the [pre-processing](https://docs.google.com/document/d/1ucZ8Ix2p9hfNQK5s5im3bQpQBvALqOLcsALupr2Am80/edit?tab=t.76na47tpy7gr#heading=h.m7f4lct2qw4l) clean-up again after creating the chunks.

### Metadata enrichment

Expected output: All embeddings should contain these metadata fields:

Operational: ingestion_method, extraction_model, schema_version, pii_flags, acl_tags.

Structural (layout): doc_id, version, section_path, page, bbox, table_shape (rows, cols), row_ids, col_headers.

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

### Footnote references

Missing footnote references: Tables and texts in general often have important footnotes that provide additional context but aren't being processed. This content needs to be appended in the same chunk that references it. [need to figure out implementation]

Semantic (facets): entities.{ORG, PERSON, DATE, GEO}, normalized numerics measures as (measure_id, value_numeric, unit), time_granularity.

### Named Entity Recognition (NER)

More sophisticated metadata implementation can be mostly based on NER here. I used a spaCy model.

Identifying key-value pairs within the chunks to store in meta-data. These could be reference specific entities, geographic regions, time periods, product categories, definitions, metrics, properties, dates, etc. which are treated as discrete facts. Ideal for exact-match retrieval.

- regions[] (denormalized convenience): canonical regions in chunk, e.g., ["EMEA","APAC"]
- "time_context":{"start":"2023-02-01","end":"2024-01-31","granularity":"fiscal_year"},
    - time_periods[]: normalized fiscal periods, e.g., ["FY2024","Q4-2024"]
    - dates[]: ISO dates present, e.g., ["2024-04-30"]
- "orgs": ["Elastic N.V."],
- "product_version":"v1"

### Merge tiny chunks

Small chunks create retrieval noise - too many irrelevant chunks might be returned during vector search. More importantly, Tiny chunks lack the surrounding information needed for proper understanding, leading to incorrect or incomplete answers. On the other hand, the semantic meaning gets diluted in large chunks, making vector embeddings less focused. The token limit can also exceed.

Therefore, ideally, we will aim for a size between 600-800 tokens for the chunks.

We want to merge the chunks if the length is less than 50 tokens. Typically, we’d prefer merging the adjacent chunks. Chunk Merging Policy:

- If a chunk is <50 tokens, attempt to merge it with an adjacent chunk.
- Preferred merge: with the immediately preceding or following chunk, provided the combined length does not exceed the max size of 800 tokens.
- If direct merge with adjacent chunk is not possible:
    1. Hierarchy check: merge with an adjacent chunk that shares the same heading path (H1/H2/H3).
    2. Semantic check: compute Jaccard similarity with the other chunks; merge if score >0.6.
- Exception: never merge table chunks.

### Deduplicate table chunks

Edge case. TBD if the need arises. For tables, good to have: Identify and filter duplicate or near-duplicate tables using content signatures. Tables may appear multiple times in different formats or with slight variations while parsing through Docling. Rare scenarios with large tables, keep logging to monitor and develop if required. Content Hashing: Table signatures enable efficient deduplication.

# Embedding Generation

I relied on BAAI/bge-small-en-v1.5 for embedding generation after testing a few options. This model works well for business documents and is rather fast.

- Dimension: 384 (compact yet effective) - Smaller embeddings mean less storage space needed
- Performance: Does well on MTEB leaderboard for retrieval tasks; Small size, CPU-friendly runs without needing expensive GPU infra; Works well with English business documents

# References

More reading material here:

1. [https://arxiv.org/abs/2411.10541](https://arxiv.org/abs/2411.10541) paper says MD is better for the newer models' prompts.
2. One similar method that can possibly act as a reference source is this public [github repo](https://github.com/aws-samples/layout-aware-document-processing-and-retrieval-augmented-generation?tab=readme-ov-file). But the problem with this approach is that it ends up creating chunks of widely dissimilar sizes, failing at scale. During retrieval, we retrieve single word chunks.
3. Reason to aim for NER: Example: [https://www.nature.com/articles/s41746-024-01377-1](https://www.nature.com/articles/s41746-024-01377-1)
4. 

More notes for future reference:

How Graph RAG plugs in later? You don’t need a graph to start; you need clean anchors to add a graph.

- Entity/edge projection: run an extractor over your existing children and produce:
- No re-embedding required to build the first KG; you’re reading from stored text using the stable spans.
- Retrieval routing in Agentic RAG can then choose:
    - Text RAG: standard dense/sparse over child chunks.
    - Graph hop: query KG, then bring back the evidence chunks via the span anchors for grounding.
- Provenance stays intact because edges always cite chunk_id + span.