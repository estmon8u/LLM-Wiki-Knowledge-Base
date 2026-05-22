# Per-question side-by-side review (WGR vs GraphRAG)

This tracked review artifact is rendered by `scripts/render_eval_reviews.py` so
the final verification report can be audited question by question.

Run label: `real_pdf_answers_v2` — see `eval/results/backend_summary_real_pdf_answers_v2.md` for the headline numbers.
Each section shows the question text, both backends' answers, citation counts, and what the retrieval surfaced.

## atlas_few_shot

**Question:** How does Atlas combine retrieval with few-shot learning?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 1.0 | 1.0 |
| grounded_entity_rate | 1.0 | 1.0 |
| insufficient_evidence | False | False |
| citation_count | 3 | 1 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 4.098503811001137 | 5.63750614500168 |

### WGR answer

```text
Atlas combines retrieval with few-shot learning by using a retrieval-augmented (semi-parametric) architecture: it first retrieves the top-k most relevant documents for the current input using a dense retriever, and then feeds the retrieved documents together with the query into a seq2seq language model to generate the output. Atlas is explicitly designed to learn knowledge-intensive tasks from very few training examples, showing strong few-shot performance while using retrieval during both pre-training and fine-tuning.
```

### GraphRAG answer

```text
## Overview: Retrieval-augmented few-shot learning in ATLAS
ATLAS is a **retrieval-augmented language model** that combines **retrieval** with **few-shot learning** by using two components—a **retriever** and a **sequence-to-sequence language model**. For each task query, ATLAS first retrieves relevant documents from a large text corpus, then feeds those retrieved documents (together with the query) into the language model to generate the output in a few-shot manner. [Data: Sources (90, 91)].
## Architecture: two sub-models + “retrieve then generate”
- **Retriever:** ATLAS uses a dense retriever based on **Contriever**, which embeds the **query** and **documents** using transformer encoders and ranks documents by similarity (dot product of embeddings). [Data: Sources (90, 91)].
- **Language model (reader):** ATLAS uses a **T5** model (Fusion-in-Decoder style), where each retrieved document is processed, and the decoder cross-attends over the combined document representations together with the query to produce the final text output. [Data: Sources (90, 91)].
## How few-shot learning comes into play
In downstream tasks, ATLAS is evaluated in **few-shot settings** (e.g., KILT and MMLU use “5-shot” by default), where the model is adapted via **fine-tuning** on a small number of examples per task. During inference/training, the retrieved documents serve as extra context, enabling the language model to leverage external knowledge and thereby achieve few-shot performance more effect
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Atlas: Few-shot Learning with Retrieval Augmented Language Models [TextUnit 0]` ← `raw/normalized/atlas-few-shot-learning-with-retrieval-augmented-language-models.`
- `Atlas: Few-shot Learning with Retrieval Augmented Language Models [TextUnit 1]` ← `raw/normalized/atlas-few-shot-learning-with-retrieval-augmented-language-models.`
- `Atlas: Few-shot Learning with Retrieval Augmented Language Models [TextUnit 25]` ← `raw/normalized/atlas-few-shot-learning-with-retrieval-augmented-language-models.`
- `Atlas: Few-shot Learning with Retrieval Augmented Language Models [TextUnit 12]` ← `raw/normalized/atlas-few-shot-learning-with-retrieval-augmented-language-models.`
- `Atlas: Few-shot Learning with Retrieval Augmented Language Models [TextUnit 11]` ← `raw/normalized/atlas-few-shot-learning-with-retrieval-augmented-language-models.`
- `Atlas: Few-shot Learning with Retrieval Augmented Language Models [TextUnit 13]` ← `raw/normalized/atlas-few-shot-learning-with-retrieval-augmented-language-models.`
- `Atlas: Few-shot Learning with Retrieval Augmented Language Models [TextUnit 10]` ← `raw/normalized/atlas-few-shot-learning-with-retrieval-augmented-language-models.`

**GraphRAG:**
- `text_unit 84` ← `graph://text_units/84`
- `Atlas Few-shot Learning with Retrieval-Augmented Language Models (few-shot/RAG r` ← `graph://community_reports/543`
- `text_unit 94` ← `graph://text_units/94`
- `ATLAS Few-Shot Learning with Retrieval-Augmented Language Models (ATLAS)` ← `graph://community_reports/207`
- `text_unit 93` ← `graph://text_units/93`
- `Atlas few-shot learning with retrieval-augmented language models (documented tra` ← `graph://community_reports/149`
- `Atlas: Few-shot Learning with Retrieval-Augmented Language Models (RAG) Communit` ← `graph://community_reports/437`
- `text_unit 90` ← `graph://text_units/90`

## corpus_retrieval_patterns

**Question:** What are the main retrieval patterns across the corpus?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.975 | 1.0 |
| grounded_entity_rate | 0.75 | 1.0 |
| insufficient_evidence | False | False |
| citation_count | 6 | 28 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 10.61924340800033 | 83.67621590199997 |

### WGR answer

```text
Across the retrieved works, the main “retrieval patterns” (how external context is selected and used across the corpus) are:

1) **Corpus-wide, graph-structured retrieval/summarization (GraphRAG)**: Build an **entity knowledge graph** from the source documents, detect **communities**, generate **hierarchical community summaries**, then answer queries using a **map-reduce style** aggregation over those community summaries.
- Evidence: GraphRAG is proposed for “global” corpus questions and uses community summaries with a map-reduce style process at query time. (citation_ref: wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summarization.md#chunk-1, citation_ref: wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summarization.md#chunk-0)

2) **Top‑k passage retrieval from a dense index (DPR / Dense bi-encoder retrieval)**: Use a **dense bi-encoder** trained on question–passage pairs to **rank passages**, then retrieve **top‑k** passages with FAISS (maximum inner product) from a large corpus.
- Evidence: DPR learns a question/passage dual-BERT bi-encoder, retrieves **top-k** passages using FAISS for efficient maximum inner-product retrieval over millions of passages, and translates retrieval precision into improved QA. (citation_ref: wiki/sources/dense-passage-retrieval-for-open-domain-question-answering.md#chunk-0)

3) **Top‑k retrieved documents prepended to the LM input (REPLUG)**: Retrieve a set of **top‑k documents**, reformulate in
```

### GraphRAG answer

```text
## Executive summary: dominant retrieval motif
Across the corpus, the main retrieval patterns shall center on **retrieval-augmented generation / retrieval-augmented prediction**, where a **retriever** first selects evidence from an external corpus/index and then a **generator/reader** conditions on that evidence to produce answers. This “**retrieve-then-generate / retrieve-then-predict**” workflow is repeatedly emphasized as a unifying paradigm for knowledge-intensive NLP and open-domain QA [Data: Reports (14, 154, 5, 30, 9, +more)].
---
## 1) Two-stage open-domain QA / RAG: retrieve → generate
The most common structure shall be:
1. **Retrieve support passages/documents** (candidate evidence) from an external corpus/index.
2. **Feed retrieved text to a generative/reading model** that produces the final answer or output.
This pattern is explicitly described for multiple families and pipelines, including an open-domain QA retrieval-then-generation flow and Fusion-in-Decoder style evidence fusion [Data: Reports (9, 44, 121, +more)].
**Key implication:** performance and factuality will depend materially on retrieval quality and how retrieval outputs are consumed by the downstream generator/decoder [Data: Reports (64, 2, 77, +more)].
---
## 2) In-context retrieval / prompt-time evidence injection (frozen LM + retrieved context)
A closely related pattern shall treat retrieval as an **inference-time prompt augmentation** mechanism:
- retrieved documents will be **prepended/inserted 
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Community 1: BERT, BM25, Dense Passage Retrieval for Open-Domain Question Answer` ← ``
- `Summary` ← `wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summariza`
- `Key Excerpt` ← `wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summariza`
- `Community 4: Codex, FAISS, GPT-3` ← ``
- `Summary` ← `wiki/sources/replug-retrieval-augmented-black-box-language-models.md`
- `Key Excerpt` ← `wiki/sources/replug-retrieval-augmented-black-box-language-models.md`
- `Community 3: Existing RALM, In-Context RALM, In-Context Retrieval-Augmented Lang` ← ``
- `Summary` ← `wiki/sources/atlas-few-shot-learning-with-retrieval-augmented-language-models.md`
- `Key Excerpt` ← `wiki/sources/atlas-few-shot-learning-with-retrieval-augmented-language-models.md`
- `Community 7: ATLAS, Atlas: Few-shot Learning with Retrieval Augmented Language M` ← ``
- `Community 5: Self-Rag, Self-rag: Learning to Retrieve, Generate, and Critique th` ← ``
- `Summary` ← `wiki/sources/dense-passage-retrieval-for-open-domain-question-answering.md`

**GraphRAG:**
- `text_unit 88` ← `graph://text_units/88`
- `text_unit 82` ← `graph://text_units/82`
- `text_unit 49` ← `graph://text_units/49`
- `text_unit 86` ← `graph://text_units/86`
- `text_unit 50` ← `graph://text_units/50`
- `text_unit 137` ← `graph://text_units/137`
- `text_unit 108` ← `graph://text_units/108`
- `text_unit 85` ← `graph://text_units/85`

## dense_passage_retrieval

**Question:** What role does dense passage retrieval play in retrieval-augmented generation?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 1.0 | 1.0 |
| grounded_entity_rate | 1.0 | 1.0 |
| insufficient_evidence | False | False |
| citation_count | 3 | 5 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 3.639171686001646 | 4.748977504998038 |

### WGR answer

```text
Dense passage retrieval provides the **non-parametric memory** component for retrieval-augmented generation: it uses a **dense vector index of Wikipedia** and a **pre-trained neural retriever** (specifically DPR) to retrieve top passages that are then used by the seq2seq generator to produce the final output. In this setup, the model can condition generation on retrieved passages, and the retriever can be trained (or jointly fine-tuned end-to-end) so that retrieval improves the downstream generation quality.
```

### GraphRAG answer

```text
Dense passage retrieval (DPR-style) acts as the **retriever** component inside retrieval-augmented generation (RAG) pipelines: it selects a small set of relevant passages (contexts) for a generative model to condition on. In the provided data, “passage retrieval” is explicitly linked with “dense retrieval” systems (including dense retrievers and dense retriever variants), indicating that dense methods are used to fetch the passages that the generator will then read/generate from [Data: Sources (29); Relationships (66391, 47732, 47696, 49612, 47740, +more)].
More specifically, the tables show dense retrieval is connected to **passage retrieval**, which is then combined with various retriever components (including traditional sparse components like BM25) in open-domain question answering setups. That same architecture pattern (retrieve passages → use them in downstream generative QA/LM) is the typical RAG role dense passage retrieval plays [Data: Sources (37, 29); Relationships (66391, 47732, 47696, 49612, 22241, +more)].
## Dense vs. sparse retrieval inside RAG
The source material contrasts dense retrieval with sparse/lexical retrieval (e.g., BM25), describing dense retrieval as using learned embeddings so it can retrieve semantically related passages even when exact token overlap is low—precisely the kind of retrieval signal that RAG systems rely on to provide useful context to the generator [Data: Sources (29); Relationships (22241, 20194, 20195, 83290, +more)].
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 0]` ← `raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.`
- `In-Context Retrieval-Augmented Language Models [TextUnit 10]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `Dense Passage Retrieval for Open-Domain Question Answering [TextUnit 8]` ← `raw/normalized/dense-passage-retrieval-for-open-domain-question-answering.md`
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 0]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `Summary` ← `wiki/sources/dense-passage-retrieval-for-open-domain-question-answering.md`
- `Dense Passage Retrieval for Open-Domain Question Answering [TextUnit 4]` ← `raw/normalized/dense-passage-retrieval-for-open-domain-question-answering.md`

**GraphRAG:**
- `Research community on retrieval-augmented and in-context retrieval language mode` ← `graph://community_reports/499`
- `Retrieval-Augmented Language Models (RAG): Dense Retrieval, Retrieval-Augmented ` ← `graph://community_reports/323`
- `Retrieval-Augmented Language Modeling Community (Dense Passage Retrieval & In-Co` ← `graph://community_reports/317`
- `Retrieval-Augmented Language Models (RAG) Community (Atlas, Dense Passage Retrie` ← `graph://community_reports/493`
- `Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks: Dense Re` ← `graph://community_reports/490`
- `Retrieval-Augmented Language Models (RAG) Research Community: Dense Passage Retr` ← `graph://community_reports/119`
- `Dense Passage Retrieval (DPR) and Retrieval-Augmented Generation (RAG) Community` ← `graph://community_reports/59`
- `Retrieval-Augmented Language Models (RALM): In-Context Retrieval, ATLAS, and Pas` ← `graph://community_reports/383`

## fid_generation

**Question:** How does FiD combine retrieval with generation?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.6 | 1.0 |
| grounded_entity_rate | 0.0 | 1.0 |
| insufficient_evidence | True | False |
| citation_count | 12 | 1 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 1.1991064230023767 | 3.2810033470013877 |

### WGR answer

```text
_Provider-free WikiGraphRAG synthesis._

WikiGraphRAG retrieved 12 context(s) using the `local` method from entities Existing RALM.

### Evidence summary

1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 0]** (raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.md#text-unit-0): # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Patrick Lewis^{†}^{‡}, Ethan Perez^{∗},
Aleksandra Piktus^{†}, Fabio Petroni^{†}, Vladimir Karpukhin^{†}, Naman Goyal^{†}, Heinrich Küttler^{†},
Mike Lewis^{†}, Wen-tau Yih^{†}, Tim Rocktäschel^{†}^{‡}, Sebastian Riedel^{†}^{‡}, Douwe Kiela^{†}
^{†}Facebook AI Research; ^{‡}University College London; ^{∗}New York University;
plewis@fb.com

###### Abstract

Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — mo
```

### GraphRAG answer

```text
## Fusion-in-Decoder (FiD): how retrieval is combined with generation
From the provided material on “Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering,” FiD works in two conceptual steps:
1. **Retrieve support passages first**
   Given a question, the system retrieves relevant text passages from an external knowledge source such as **Wikipedia**. (The figure describes retrieving “support text passages” and then conditioning generation on them.) [Data: Sources (41)].
2. **Generate with a seq2seq model conditioned on the retrieved passages**
   After retrieval, a **sequence-to-sequence (encoder-decoder) generative model** produces the answer **conditioned on both the question and the retrieved passages**. The method is described as: retrieve passages, then generate the answer taking the retrieved passages as input/context in addition to the question. [Data: Sources (41)].
### Evidence for “fusion” across multiple passages
FiD is specifically motivated by the idea that **generative models can combine evidence from multiple passages**. The paper states that performance **significantly improves when increasing the number of retrieved passages**, which they interpret as evidence that sequence-to-sequence models provide a flexible framework to aggregate and combine evidence. [Data: Sources (41)].
### Architecture detail mentioned
The input also references “Figure 2: Architecture of the Fusion-in-Decoder method,” indicating that the fusion happens
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 0]` ← `raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.`
- `Key Excerpt` ← `wiki/sources/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.md`
- `Key Excerpt` ← `wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summariza`
- `Summary` ← `wiki/sources/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.md`
- `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 8]` ← `raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.`
- `In-Context Retrieval-Augmented Language Models [TextUnit 0]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `Key Excerpt` ← `wiki/sources/leveraging-passage-retrieval-with-generative-models-for-open-domain`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 2]` ← `raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.`
- `Summary` ← `wiki/sources/in-context-retrieval-augmented-language-models.md`
- `Key Excerpt` ← `wiki/sources/in-context-retrieval-augmented-language-models.md`

**GraphRAG:**
- `Retrieval-Augmented Language Models (RAG): Dense Retrieval, Retrieval-Augmented ` ← `graph://community_reports/323`
- `Retrieval-Augmented Generation (RAG) Community: RePlug & Knowledge-Intensive NLP` ← `graph://community_reports/301`
- `Retrieval-Augmented Generation (RAG) and Dense/Latent Retrieval for Open-Domain ` ← `graph://community_reports/124`
- `Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks: Dense Re` ← `graph://community_reports/490`
- `text_unit 60` ← `graph://text_units/60`
- `text_unit 17` ← `graph://text_units/17`
- `Research community on retrieval-augmented and in-context retrieval language mode` ← `graph://community_reports/499`
- `text_unit 18` ← `graph://text_units/18`

## graph_query_modes

**Question:** Compare local, global, DRIFT, and basic GraphRAG query modes.

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.6 | 0.6 |
| grounded_entity_rate | 0.0 | 1.0 |
| insufficient_evidence | True | False |
| citation_count | 8 | 0 |
| citation_ref_valid_strict | 1.0 | 0.0 |
| latency_seconds | 1.060159337001096 | 185.73469000500336 |

### WGR answer

```text
_Provider-free WikiGraphRAG synthesis._

WikiGraphRAG retrieved 8 context(s) using the `drift-lite` method from entities Existing RALM, GraphRAG, RAG.

### Evidence summary

1. **Summary** (wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summarization.md#chunk-0): {"summary":"The document argues that conventional vector-based RAG fails for “global” sensemaking queries (query-focused summarization over an entire corpus) because such questions require dataset-wide synthesis rather than retrieval of a small relevant subset. It proposes GraphRAG, which builds an LLM-extracted entity knowledge graph from the source documents, detects communities in that graph, and generates hierarchical community summaries; at query time, it uses a map-reduce pattern over these community summaries to produce a comprehensive final answer. For evaluation, it introduces an adaptive benchmarking method that uses an LLM-judge workflow with automatically generated global sensemaking questions and compares GraphRAG to a conventional vector RAG baseline on multi-million-token-class settings. Results reported indicate GraphRAG substantially improves both comprehensiveness and diversity over vector RAG when using a strong LLM for generation/judging, but the paper leaves open questions about how performance varies with graph quality, community-detection choices, and more diverse question types without ground-truth answers.","key_points":["Thesis: global sensemaking requires query-focus
```

### GraphRAG answer

```text
## What I can and can’t answer from the provided data
The provided materials **do not define** “local, global, DRIFT, and basic GraphRAG query modes” in a single, formal way. In particular, the GraphRAG excerpt you provided describes **GraphRAG’s pipeline** and **community-hierarchy levels (C0–C3)**, but it does **not** introduce “DRIFT” or “basic GraphRAG” as named query modes with explicit algorithms. Because of that, I can only give a **conceptual mapping** (not a mode-by-mode specification) based on how GraphRAG’s *scope* is controlled in the text.
## GraphRAG scope in the provided sources (what “local vs global” corresponds to)
In the GraphRAG paper excerpt, “scope” is controlled by **which community levels you summarize** in a hierarchical community structure:
- **C0 (root-level)**: uses **root-level community summaries** (fewest summaries / most coarse).
- **C1, C2**: use progressively deeper community summaries (more granular).
- **C3 (low-level)**: uses **low-level community summaries** (most summaries / most fine-grained).
The excerpt also states GraphRAG **recursively creates increasingly global summaries** using the community hierarchy, and then produces a **global answer** via query-focused summarization over community summaries. [Data: from-local-to-global-a-graphrag-approach-to-query-focused-summarization (138, 4, 5, 6, 135)]
### Conceptual mapping to your “local vs global” labels
- **Local mode (conceptual)**: corresponds to using **deeper / lower-level commun
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Summary` ← `wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summariza`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `In-Context Retrieval-Augmented Language Models [TextUnit 0]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `In-Context Retrieval-Augmented Language Models [TextUnit 8]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `Key Excerpt` ← `wiki/sources/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 2]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 1]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`

**GraphRAG:**
- `text_unit 139` ← `graph://text_units/139`
- `text_unit 140` ← `graph://text_units/140`
- `text_unit 134` ← `graph://text_units/134`
- `text_unit 133` ← `graph://text_units/133`
- `text_unit 137` ← `graph://text_units/137`
- `text_unit 135` ← `graph://text_units/135`
- `text_unit 138` ← `graph://text_units/138`
- `Microsoft GraphRAG (GraphRAG) for Global Query-Focused Summarization (News & Pod` ← `graph://community_reports/365`

## graphrag_community_summaries

**Question:** How does Microsoft GraphRAG use community summaries for global queries?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 1.0 | 1.0 |
| grounded_entity_rate | 1.0 | 1.0 |
| insufficient_evidence | False | False |
| citation_count | 3 | 2 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 6.269999292002467 | 3.547054955000931 |

### WGR answer

```text
Microsoft GraphRAG uses community summaries as a hierarchical, global “index” for answering global/sensemaking queries:

- **Build a knowledge graph and partition it into communities.** GraphRAG uses an LLM to construct an entity knowledge graph from the corpus, then partitions the graph into a **hierarchy of communities** of closely related entities/nodes. (The hierarchy is created recursively with community detection until leaf communities can no longer be partitioned.)
- **Generate community summaries bottom-up through the hierarchy.** It **pre-generates summaries for communities at every level**. Community summaries at lower levels are used to produce summaries for higher-level communities, so higher-level summaries **recursively incorporate** lower-level summaries.
- **Answer a query via map-reduce over community summaries.** For a user question, GraphRAG performs query answering by:
  - **Map step:** using the relevant **community summaries to generate partial answers independently and in parallel**.
  - **Reduce step:** combining those partial answers and synthesizing them into a **final global answer**.

This design lets GraphRAG provide **global sensemaking** over the whole corpus (rather than just retrieving a small set of locally relevant chunks).
```

### GraphRAG answer

```text
## How GraphRAG uses *community summaries* for global queries
Microsoft GraphRAG first builds a **graph index** from the corpus (entities/relationships/claims), then **partitions that graph into a hierarchical set of communities** of closely related entities. After that, it generates **community summaries** at multiple levels of the hierarchy, using a bottom-up rollup where higher-level summaries incorporate lower-level ones. [Data: Sources (133, 136, +more)]
For a **user’s global query**, GraphRAG answers it via a **map-reduce** process over the relevant community summaries:
1. **Prepare community summaries (context set)**
   - The community summaries at the selected hierarchy level are **shuffled** and **divided into chunks** of a pre-specified token size, so the query processing is distributed rather than relying on a single context window. [Data: Sources (137, +more)]
2. **Map step: generate partial answers + helpfulness scores**
   - Each chunk is processed independently (in parallel). For each chunk, the LLM generates an **intermediate (partial) answer** to the query.
   - The LLM also produces a **helpfulness score from 0–100** indicating how useful that partial answer is for the query; summaries with score **0 are filtered out**. [Data: Sources (137, +more)]
3. **Reduce step: merge partial answers into the final global answer**
   - The intermediate answers are sorted by **descending helpfulness score**.
   - The system then iteratively adds these partial answers into
```

### Retrieval — top 8 (title | path)

**WGR:**
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `Summary` ← `wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summariza`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`

**GraphRAG:**
- `text_unit 137` ← `graph://text_units/137`
- `text_unit 140` ← `graph://text_units/140`
- `text_unit 138` ← `graph://text_units/138`
- `text_unit 133` ← `graph://text_units/133`
- `text_unit 139` ← `graph://text_units/139`
- `text_unit 136` ← `graph://text_units/136`
- `GraphRAG (Microsoft Research) Community of Entities for Global Sensemaking via Q` ← `graph://community_reports/369`
- `Microsoft GraphRAG (Microsoft/graphrag) Community for Global Sensemaking with Kn` ← `graph://community_reports/6`

## hallucination_mitigation

**Question:** What patterns does the corpus show for reducing hallucinations in retrieval-augmented systems?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.6 | 0.95 |
| grounded_entity_rate | 0.0 | 0.5 |
| insufficient_evidence | True | False |
| citation_count | 12 | 21 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 0.9765463849980733 | 127.0990462080008 |

### WGR answer

```text
_Provider-free WikiGraphRAG synthesis._

WikiGraphRAG retrieved 12 context(s) using the `global` method from communities community-4, community-3, community-7.

### Evidence summary

1. **Community 4: Codex, FAISS, GPT-3** (community-4): Entities: Codex, FAISS, GPT-3, MMLU, REPLUG, RePlug LSR, REPLUG: Retrieval-Augmented Black-Box Language Models, Retrieval-Augmented Language Modeling | Pages: REPLUG: Retrieval-Augmented Black-Box Language Models | Sources: 40d94ca0-3862-467a-a36c-050d5bd863f3, 45392d42-2e20-4b00-ae4e-fd10fc0c93aa, 0bb23c1e-ad74-4e9f-8a42-2c3a86f15dd1, f85697b7-cace-438e-add3-2efd3cdbc965
2. **Summary** (wiki/sources/replug-retrieval-augmented-black-box-language-models.md#chunk-0): {"summary":"The document proposes REPLUG, a retrieval-augmented language modeling approach that keeps the language model as a black box and improves predictions by retrieving relevant documents from an external corpus and prepending them to the input. Instead of retraining the language model with special cross-attention, REPLUG passes multiple “(retrieved document + input)” reformulations through the frozen LM and ensembles the resulting probabilities to overcome context-length limits. It also introduces LM-Supervised Retrieval (RePlug LSR), which treats the LM as a fixed scoring function and trains the retriever to prefer documents that reduce LM perplexity (and thereby improve downstream performance). Experiments reported in the excerpt indicate sizable gains for large models (e.
```

### GraphRAG answer

```text
## Core pattern: ground generation in retrieved evidence (not parametric memory)
The corpus repeatedly frames hallucination reduction as a **grounding effect**: systems should retrieve relevant external passages/documents first, and then generate answers **conditioned on that retrieved context** rather than relying primarily on the model’s internal (parametric) knowledge [Data: Reports (5, 30)]. This is explicitly described as improving “knowledge reliability,” including the claim that retrieval augmentation **reduces hallucination in conversation** [Data: Reports (28, 19, 81, 110)].
## Tighten the coupling between retrieval and generation (retrieval-then-generation)
A consistent mechanism is to **tighten how retrieved evidence constrains generation**, so the model is less able (and less likely) to produce statements that are not supported by the retrieved passages [Data: Reports (25)]. The corpus highlights this as a shift from “retrieve-and-generate” toward an **evidence-faithfulness pipeline** where evidence usage is checked during/around generation [Data: Reports (96, 89, 95, 95)].
## Self-RAG reflection tokens: verification gates for relevance and support
The most specific hallucination-reduction mechanism in the corpus is **Self-RAG–style reflection/critique**. It uses structured “reflection tokens” to decide (a) **whether to retrieve**, (b) whether retrieved evidence is **relevant**, and (c) whether generated statements are **supported**—including handling **contradict
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Community 4: Codex, FAISS, GPT-3` ← ``
- `Summary` ← `wiki/sources/replug-retrieval-augmented-black-box-language-models.md`
- `Key Excerpt` ← `wiki/sources/replug-retrieval-augmented-black-box-language-models.md`
- `Community 3: Existing RALM, In-Context RALM, In-Context Retrieval-Augmented Lang` ← ``
- `Summary` ← `wiki/sources/atlas-few-shot-learning-with-retrieval-augmented-language-models.md`
- `Key Excerpt` ← `wiki/sources/atlas-few-shot-learning-with-retrieval-augmented-language-models.md`
- `Community 7: ATLAS, Atlas: Few-shot Learning with Retrieval Augmented Language M` ← ``
- `Community 1: BERT, BM25, Dense Passage Retrieval for Open-Domain Question Answer` ← ``
- `Summary` ← `wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summariza`
- `Key Excerpt` ← `wiki/sources/from-local-to-global-a-graphrag-approach-to-query-focused-summariza`
- `Community 5: Self-Rag, Self-rag: Learning to Retrieve, Generate, and Critique th` ← ``
- `Summary` ← `wiki/sources/dense-passage-retrieval-for-open-domain-question-answering.md`

**GraphRAG:**
- `text_unit 88` ← `graph://text_units/88`
- `text_unit 111` ← `graph://text_units/111`
- `text_unit 85` ← `graph://text_units/85`
- `text_unit 137` ← `graph://text_units/137`
- `text_unit 115` ← `graph://text_units/115`
- `text_unit 96` ← `graph://text_units/96`
- `text_unit 49` ← `graph://text_units/49`
- `text_unit 82` ← `graph://text_units/82`

## in_context_ralm

**Question:** How does In-Context RALM use retrieved documents at inference time?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 1.0 | 1.0 |
| grounded_entity_rate | 1.0 | 1.0 |
| insufficient_evidence | False | False |
| citation_count | 5 | 1 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 6.748685702994408 | 4.25592231599876 |

### WGR answer

```text
At inference time, **In-Context RALM leaves the language model (LM) architecture unchanged** and **uses a zero-effort reading mechanism**: it **prepends/concatenates the selected retrieved grounding documents to the LM input text** before generation, so the LM conditions on those documents through the standard Transformer input.

Concretely, the method constructs the next-token predictions conditioned on **(a) the usual prefix tokens** and **(b) the retrieved documents inserted before that prefix**. When the concatenated input would exceed the LM’s maximum context length, it **truncates tokens from the beginning of the prefix** so the total input fits the model limit.

Additionally, In-Context RALM can **perform retrieval at a stride**: instead of retrieving at every generation step, it retrieves once every **s** tokens, and uses the **last ℓ tokens** of the current prefix as the **retrieval query** for BM25 (or analogous retriever) at each stride.

Finally, in the reranking variant, the inference-time documents are chosen by **reranking the top-k documents returned by BM25** (either with a smaller LM for zero-shot reranking, or via a trained predictive reranker), and then the chosen top document(s) are the ones prepended to the LM input for generation.
```

### GraphRAG answer

```text
## Inference-time use of retrieved documents in In-Context RALM
In **In-Context RALM**, the language model architecture is **not modified**. At inference time, the system **retrieves relevant documents** from a grounding corpus and then **prepends (inserts) the selected retrieved documents to the LM input text** before generation—i.e., the LM simply “reads” the retrieved passages as part of its prompt [Data: Sources (113; 111)].
This means the retrieved content is incorporated via the **LM’s normal context mechanism** (conditioning on the prefix text), rather than via specialized “retrieval attention” layers or other architecture changes [Data: Sources (113; 111)].
## Practical retrieval flow (as described)
- **Retrieve**: Use an off-the-shelf retriever (sparse BM25 or dense neural retrievers) to select top-scoring passages relevant to the current prefix/query during generation. [Data: Sources (113; 111)]
- **Form the prompt**: Take the selected passage(s) and **prepend** them to the LM’s input so that next-token prediction is conditioned on both the original prefix and the retrieved document text [Data: Sources (113; 111)].
```

### Retrieval — top 8 (title | path)

**WGR:**
- `In-Context Retrieval-Augmented Language Models [TextUnit 1]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 0]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 6]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 8]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 2]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 3]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 5]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`

**GraphRAG:**
- `text_unit 111` ← `graph://text_units/111`
- `text_unit 110` ← `graph://text_units/110`
- `text_unit 118` ← `graph://text_units/118`
- `text_unit 115` ← `graph://text_units/115`
- `In-Context Retrieval-Augmented Language Models (In-Context RALM) community: BM25` ← `graph://community_reports/201`
- `In-Context Retrieval-Augmented Language Models (In-Context RALM) for LM Gains an` ← `graph://community_reports/425`
- `In-Context Retrieval-Augmented Language Modeling (In-Context RALM)` ← `graph://community_reports/199`
- `text_unit 88` ← `graph://text_units/88`

## orqa_inverse_cloze

**Question:** How does ORQA pretrain its retriever?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.9667 | 0.9167 |
| grounded_entity_rate | 1.0 | 0.5 |
| insufficient_evidence | False | False |
| citation_count | 3 | 4 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 4.696657275999314 | 5.613077751004312 |

### WGR answer

```text
ORQA pre-trains its retriever using an unsupervised **Inverse Cloze Task (ICT)**. In ICT, a **sentence is treated as a pseudo-question** and the **sentence’s context is treated as pseudo-evidence**; the task is to select the corresponding pseudo-evidence out of candidates in a batch. This pre-training provides a strong initialization so the retriever and reader can then be fine-tuned end-to-end using only question-answer string pairs, with evidence retrieval treated as a latent variable over Wikipedia.
```

### GraphRAG answer

```text
## ORQA retriever pretraining (what the provided data shows)
In the tables you supplied, **ORQA** is represented as a retriever method/entity, and there are **relationships linking ORQA to retriever components and training/evaluation resources**, but the data does **not** include any description of *the specific pretraining procedure or objective* used for ORQA’s retriever.
What we *can* say from the relationships table is limited to that ORQA is treated as part of an overall pipeline and is associated with common systems/components, including:
- **ORQA ↔ Strong retriever** [Data: Relationships (83848; 75378)].
- **ORQA ↔ Wikipedia** (i.e., Wikipedia is among the corpora/indexes linked to ORQA in this dataset of relationships) [Data: Relationships (5256; 47742; 36929; 101752; 351)].
- **ORQA ↔ QA** [Data: Relationships (76532; 83890)].
- **ORQA ↔ TF-IDF** (suggesting ORQA may be connected to a TF-IDF baseline/variant in the relationship graph, but not explaining pretraining) [Data: Relationships (5254)].
## What’s missing
Your input tables **do not contain**:
- the ORQA retriever **loss function/objective**,
- whether it uses **contrastive learning**, **reader supervision**, **margins**, **weak supervision**, etc.,
- the **training data construction** (e.g., how positives/negatives are formed),
- or the **exact pretraining steps**.
## Conclusion
With only the provided tables, I **can’t determine how ORQA pretrains its retriever**—the tables only show high-level associations (
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Key Excerpt` ← `wiki/sources/realm-retrieval-augmented-language-model-pre-training.md`
- `Latent Retrieval for Weakly Supervised Open Domain Question Answering [TextUnit ` ← `raw/normalized/latent-retrieval-for-weakly-supervised-open-domain-question-answe`
- `Latent Retrieval for Weakly Supervised Open Domain Question Answering [TextUnit ` ← `raw/normalized/latent-retrieval-for-weakly-supervised-open-domain-question-answe`
- `Latent Retrieval for Weakly Supervised Open Domain Question Answering [TextUnit ` ← `raw/normalized/latent-retrieval-for-weakly-supervised-open-domain-question-answe`
- `Dense Passage Retrieval for Open-Domain Question Answering [TextUnit 7]` ← `raw/normalized/dense-passage-retrieval-for-open-domain-question-answering.md`
- `REALM: Retrieval-Augmented Language Model Pre-Training [TextUnit 7]` ← `raw/normalized/realm-retrieval-augmented-language-model-pre-training.md`
- `REALM: Retrieval-Augmented Language Model Pre-Training [TextUnit 0]` ← `raw/normalized/realm-retrieval-augmented-language-model-pre-training.md`
- `Latent Retrieval for Weakly Supervised Open Domain Question Answering [TextUnit ` ← `raw/normalized/latent-retrieval-for-weakly-supervised-open-domain-question-answe`

**GraphRAG:**
- `Open-Retrieval Question Answering (ORQA) Models: Wikipedia-based Learned Retriev` ← `graph://community_reports/48`
- `text_unit 115` ← `graph://text_units/115`
- `ORQA (Open-Retrieval Question Answering) Community: BERT-based Retriever-Reader ` ← `graph://community_reports/406`
- `ORQA (Open-Retrieval Question Answering) Community: Retriever-Reader Joint Learn` ← `graph://community_reports/161`
- `Open-Retrieval Question Answering (ORQA): BERT-Based Retriever-Reader Community` ← `graph://community_reports/56`
- `text_unit 0` ← `graph://text_units/0`
- `ORQA Open-Domain QA with DPR/BM25 and ICT Pretraining (latent retrieval communit` ← `graph://community_reports/482`
- `text_unit 114` ← `graph://text_units/114`

## rag_fid_comparison

**Question:** Compare RAG and FiD for how retrieved documents are used during generation.

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.6 | 1.0 |
| grounded_entity_rate | 0.0 | 1.0 |
| insufficient_evidence | True | False |
| citation_count | 1 | 1 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 3.74840982099704 | 223.29759963900142 |

### WGR answer

```text
## What the evidence says about RAG
The provided evidence describes **Retrieval-Augmented Generation (RAG)** as a knowledge-intensive generation framework where the system **retrieves documents** and then uses them during generation (e.g., for Jeopardy question generation and open-domain QA), and it emphasizes that retrieval provides more factual/specific outputs compared to a non-retrieval baseline.

However, the evidence **does not mention FiD** (Fusion-in-Decoder) or describe how FiD uses retrieved documents during generation, so a direct comparison to FiD cannot be grounded in the supplied sources.

## Comparison to FiD
**Insufficient evidence:** The retrieved contexts contain information about RAG, but **no information about FiD** (how it uses retrieved documents or any architectural details). Therefore, I cannot reliably compare RAG vs FiD based only on the evidence given.
```

### GraphRAG answer

```text
## High-level comparison: where retrieved documents enter generation
### RAG (Retrieval-Augmented Generation)
In the provided material, RAG uses retrieved documents as **conditioning evidence inside the generator**, but the *mathematical way* retrieval is coupled to generation depends on the RAG variant:
- **RAG-Token:** retrieval is integrated **at the token level** via a per-token marginalization over the top‑k retrieved documents, so different tokens can (in principle) be supported by different retrieved documents. [Data: retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks (28, 26, 14)]
- **RAG-Sequence:** retrieval is integrated **at the sequence level** by treating the retrieved document as a **single latent variable for the whole output**, then marginalizing over documents for the full sequence. Because the likelihood doesn’t factor into standard per-token terms, decoding uses special approximations (“Thorough” / “Fast” decoding). [Data: retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks (19, 28, 27)]
### FiD (Fusion-in-Decoder)
The provided material does not include a full FiD algorithmic description for “how retrieved documents are used during generation” in the same way it does for RAG. However, it *does* describe FiD-style fusion at the architecture level (from the ATLAS/FiD description):
- Retrieved documents are **encoded independently** in the encoder.
- Their encoder outputs are then **concatenated**, and the decoder performs **cross-a
```

### Retrieval — top 8 (title | path)

**WGR:**
- `In-Context Retrieval-Augmented Language Models [TextUnit 0]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 8]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `Key Excerpt` ← `wiki/sources/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 2]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 1]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 6]` ← `raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.`
- `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 3]` ← `raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.`

**GraphRAG:**
- `text_unit 27` ← `graph://text_units/27`
- `text_unit 18` ← `graph://text_units/18`
- `text_unit 60` ← `graph://text_units/60`
- `text_unit 17` ← `graph://text_units/17`
- `text_unit 77` ← `graph://text_units/77`
- `text_unit 19` ← `graph://text_units/19`
- `text_unit 16` ← `graph://text_units/16`
- `text_unit 13` ← `graph://text_units/13`

## realm_vs_rag

**Question:** How does REALM differ from RAG?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.6 | 0.5667 |
| grounded_entity_rate | 0.0 | 1.0 |
| insufficient_evidence | True | False |
| citation_count | 1 | 0 |
| citation_ref_valid_strict | 1.0 | 0.0 |
| latency_seconds | 4.820226597003057 | 181.423446847999 |

### WGR answer

```text
The provided evidence does **not** define or describe **RAG** (Retrieval-Augmented Generation), so I can’t directly explain how **REALM** differs from **RAG** based on this bundle.

What I *can* say from the evidence is how **REALM** is structured: it uses (1) a **knowledge retriever** that embeds documents/titles and retrieves candidate documents, and (2) a **knowledge-augmented encoder** that concatenates the input and retrieved document into a single sequence to enable **cross-attention** before predicting the output. REALM’s retriever and encoder are differentiable neural networks that are trained via log-likelihood optimization, and REALM involves pre-training where the retriever/encoder are jointly pre-trained, then fine-tuned on the target task.
```

### GraphRAG answer

```text
## High-level difference
Both **REALM** and **RAG** are retrieval-augmented approaches (they use an external corpus/index to ground a language model), but they differ in **where retrieval is integrated into the learning pipeline**:
- **RAG** is primarily a **retrieval-augmented generation** framework: retrieve documents, then **generate** conditioned on them.
- **REALM** is primarily a **retrieval-augmented pre-training** framework: retrieval is **inside the pre-training objective**, so the model learns to use retrieval during training (including learning signals that update the retriever). [Data: realm-retrieval-augmented-language-model-pre-training (159aa0b4-fc15-407c-88ce-1071c3fb1735); retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks (13, 14)]
## Where retrieval “lives” (training stage)
### RAG: retrieval used to condition generation
RAG couples retrieval to the **generation likelihood** by marginalizing over a **top‑K** set of retrieved documents during decoding/training of the generator. It has variants like:
- **RAG-Sequence** (one retrieved document choice for the whole output sequence)
- **RAG-Token** (document choice can vary per token) [Data: retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks (13, 14); retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks (19, 28, 17)]
### REALM: retrieval used inside the pre-training objective
REALM integrates retrieval into **language-model pre-training** as a latent-variable formulation: 
```

### Retrieval — top 8 (title | path)

**WGR:**
- `In-Context Retrieval-Augmented Language Models [TextUnit 0]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `Key Excerpt` ← `wiki/sources/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 2]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 1]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `In-Context Retrieval-Augmented Language Models [TextUnit 8]` ← `raw/normalized/in-context-retrieval-augmented-language-models.md`
- `REALM: Retrieval-Augmented Language Model Pre-Training [TextUnit 3]` ← `raw/normalized/realm-retrieval-augmented-language-model-pre-training.md`
- `REALM: Retrieval-Augmented Language Model Pre-Training [TextUnit 7]` ← `raw/normalized/realm-retrieval-augmented-language-model-pre-training.md`

**GraphRAG:**
- `Open-Domain QA Retrieval-Augmented Generation (RAG) — Fusion-in-Decoder, Self-RA` ← `graph://community_reports/167`
- `text_unit 134` ← `graph://text_units/134`
- `text_unit 17` ← `graph://text_units/17`
- `text_unit 140` ← `graph://text_units/140`
- `GraphRAG Community Evaluation Framework (GraphRAG vs Vector RAG/TS on Podcast & ` ← `graph://community_reports/532`
- `text_unit 7` ← `graph://text_units/7`
- `text_unit 18` ← `graph://text_units/18`
- `text_unit 139` ← `graph://text_units/139`

## replug_black_box

**Question:** How does REPLUG augment a black-box language model with retrieval?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 1.0 | 1.0 |
| grounded_entity_rate | 1.0 | 1.0 |
| insufficient_evidence | False | False |
| citation_count | 3 | 2 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 4.732945784002368 | 4.7177298850001534 |

### WGR answer

```text
REPLUG augments a black-box language model by adding a tuneable retrieval “plug” that retrieves relevant documents from an external corpus and then prepends those retrieved documents to the model’s input. Specifically, given an input context, REPLUG first retrieves a small set of relevant documents with a retriever, concatenates each retrieved document with the input context, feeds these reformulated inputs into the frozen black-box LM in parallel, and ensembles the resulting output probabilities to produce the final prediction. Because the LM is treated as a black box (no internal representations or parameter updates are required), this design is applicable to existing black-box LMs and can use any existing retrieval and language model combination.
```

### GraphRAG answer

```text
## REPLUG’s augmentation approach (black-box + retrieval)
REPLUG augments a language model that is treated as a **black box** by adding a **tunable retrieval component** outside the LM. In other words, instead of modifying the LM’s internal architecture or parameters, REPLUG retrieves relevant documents from an external corpus and then uses those documents to influence the LM’s prediction. [Data: Sources (46)]
## What happens at inference time
Given an input context, REPLUG:
1. **Uses a retriever to retrieve relevant documents** from an external corpus.
2. **Prepends the retrieved documents to the input context**.
3. **Feeds the augmented input to the frozen black-box LM** to produce the final prediction. [Data: Sources (46)]
Because the retrieved texts are added to the prompt, the method is designed to work even when internal LM representations are not accessible (e.g., when the LM is served via an API). [Data: Sources (46)]
## Training/tuning the retriever (LM remains a black box)
REPLUG can also **improve the retrieval model quality** using training where the LM remains frozen/black-box: it uses a training scheme (“LM-supervised retrieval”) that adapts the retriever based on language-model scoring signals (e.g., preferring retrieval that improves LM performance such as perplexity), without requiring changes to the LM itself. [Data: Sources (46); Sources (53)]
## Key takeaway
So, REPLUG’s core idea is: **keep the LM fixed and inaccessible, but use retrieval to supply suppor
```

### Retrieval — top 8 (title | path)

**WGR:**
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 7]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `Summary` ← `wiki/sources/replug-retrieval-augmented-black-box-language-models.md`
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 0]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 5]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 1]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 2]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 4]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `Key Excerpt` ← `wiki/sources/replug-retrieval-augmented-black-box-language-models.md`

**GraphRAG:**
- `text_unit 46` ← `graph://text_units/46`
- `text_unit 47` ← `graph://text_units/47`
- `text_unit 52` ← `graph://text_units/52`
- `text_unit 50` ← `graph://text_units/50`
- `text_unit 51` ← `graph://text_units/51`
- `REPLUG Retrieval-Augmented Black-Box Language Models (Li Bai rare-entity retriev` ← `graph://community_reports/363`
- `Retrieval-Augmented Black-Box Language Models (REPLUG) & REALM Retrieval-Augment` ← `graph://community_reports/129`
- `text_unit 53` ← `graph://text_units/53`

## self_rag_reflection

**Question:** How does Self-RAG use reflection or critique signals?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 1.0 | 1.0 |
| grounded_entity_rate | 1.0 | 1.0 |
| insufficient_evidence | False | False |
| citation_count | 6 | 2 |
| citation_ref_valid_strict | 1.0 | 1.0 |
| latency_seconds | 8.888360062002903 | 5.937372945998504 |

### WGR answer

```text
Self-RAG uses **reflection tokens** to represent both *retrieval* decisions and *critique* of content quality.

- **During training**, an LM is trained to generate both task text and reflection tokens as next-token predictions from an expanded vocabulary (original tokens + reflection tokens). The training data is **offline augmented** by a critic model that inserts these reflection tokens into the corpus. 
- **Critique signals** come from critique tokens that evaluate: (1) whether retrieved passages are relevant, and (2) whether the generated information is supported by the retrieved passage(s). 
- **During inference**, the model decodes reflection tokens so it can (a) decide whether retrieval is needed and (b) evaluate/re-rank or control segment generation using its own critique token probabilities. This makes the system controllable at inference time by adjusting how critique signals affect decoding (e.g., through weighted segment-level beam search).

In short: reflection/critique tokens are explicitly generated by the model, appended/used at segment level, and then used to guide which retrieved passages and generated segments should be selected as the final output.
```

### GraphRAG answer

```text
## How Self-RAG uses reflection / critique signals
Self-RAG trains its generator to emit **special “reflection tokens”** during generation. These tokens encode both (1) whether **retrieval is needed** and (2) **how well the generation matches the retrieved passages**, using critique-style judgments [Data: Sources (61, 62)].
### 1) Reflection tokens decide when to retrieve
During inference, the model produces a **retrieval token**. If the token indicates retrieval is helpful, Self-RAG calls the retriever; otherwise it continues generating without retrieval [Data: Sources (61)].
### 2) Critique tokens evaluate retrieved passages and segment support
When retrieval is used, Self-RAG generates additional **critique tokens** that:
- assess the **relevance** of retrieved passages, and
- judge whether each generated response **segment is supported** by the retrieved passage(s) [Data: Sources (61, 62)].
Self-RAG frames generation as **multiple segments**, and it produces critique signals for segments rather than only evaluating once at the end [Data: Sources (61, 62)].
### 3) A separate critic produces supervision offline
The system includes a critic model used **during training** to produce the reflection/critique token labels. The paper states that reflection tokens are inserted **offline** into training outputs, which “eliminates the need to host a critic model during training” (reducing overhead) [Data: Sources (61)].
### 4) The decoding algorithm uses reflection/critique signals 
```

### Retrieval — top 8 (title | path)

**WGR:**
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Key Points` ← `wiki/sources/self-rag-learning-to-retrieve-generate-and-critique-through-self-re`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`

**GraphRAG:**
- `Self-RAG Retrieval and Self-Reflection Evaluation Community (Evidence: self-rag-` ← `graph://community_reports/555`
- `text_unit 60` ← `graph://text_units/60`
- `text_unit 62` ← `graph://text_units/62`
- `SELF-RAG (self-rag-learning-to-retrieve-generate-and-critique-through-self-refle` ← `graph://community_reports/89`
- `Self-RAG Self-Reflection RAG System (Learning to Retrieve, Generate, and Critiqu` ← `graph://community_reports/467`
- `text_unit 61` ← `graph://text_units/61`
- `SELF-RAG (self-rag-learning-to-retrieve-generate-and-critique-through-self-refle` ← `graph://community_reports/463`
- `text_unit 67` ← `graph://text_units/67`

## unsupported_claim

**Question:** What does the corpus say about quantum-resistant retrieval memories?

| Metric | WGR | GraphRAG |
|---|---|---|
| answer_quality_score | 0.8 | 0.2 |
| grounded_entity_rate | 0.0 | 0.0 |
| insufficient_evidence | True | False |
| citation_count | 7 | 0 |
| citation_ref_valid_strict | 1.0 | 0.0 |
| latency_seconds | 1.6477107070022612 | 1.5633142290025717 |

### WGR answer

```text
<ANSWER_HELPFULNESS>5.0</ANSWER_HELPFULNESS>

The provided corpus evidence does **not** mention “quantum-resistant retrieval memories” (or any concept specifically framed as quantum-resistant retrieval memories).
```

### GraphRAG answer

```text
I can’t answer this from the provided corpus. The data you supplied discusses **retrieval-augmented language model** ideas (e.g., REALM-style retrieval as a learned, grounded memory; and RAG as retriever+generator with a dense vector index), but it contains **no information about “quantum-resistant” retrieval memories** or anything specific to quantum-resistance/anti-quantum security.
If you can share the relevant table rows/passages that mention *quantum-resistant* (or post-quantum / PQC) retrieval memories, I can summarize what the corpus says about them.
```

### Retrieval — top 8 (title | path)

**WGR:**
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection [` ← `raw/normalized/self-rag-learning-to-retrieve-generate-and-critique-through-self-`
- `REALM: Retrieval-Augmented Language Model Pre-Training [TextUnit 8]` ← `raw/normalized/realm-retrieval-augmented-language-model-pre-training.md`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `REPLUG: Retrieval-Augmented Black-Box Language Models [TextUnit 12]` ← `raw/normalized/replug-retrieval-augmented-black-box-language-models.md`
- `From Local to Global: A GraphRAG Approach to Query-Focused Summarization [TextUn` ← `raw/normalized/from-local-to-global-a-graphrag-approach-to-query-focused-summari`
- `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [TextUnit 8]` ← `raw/normalized/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.`

**GraphRAG:**
- `text_unit 88` ← `graph://text_units/88`
- `text_unit 49` ← `graph://text_units/49`
- `text_unit 137` ← `graph://text_units/137`
- `text_unit 82` ← `graph://text_units/82`
- `text_unit 85` ← `graph://text_units/85`
- `text_unit 77` ← `graph://text_units/77`
- `text_unit 4` ← `graph://text_units/4`
- `text_unit 96` ← `graph://text_units/96`

