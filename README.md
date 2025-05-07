# LLM handbook / Study materials
#### Here are some recommended materials for learning about large language models(LLM).

#### [My Glasp highlights](https://glasp.co/n5vm3r89druhaqmm)

## Recommended reading list

### Mainly and important concepts
- [Introduction of Generative AI; Hung-yi Lee; YouTube](https://www.youtube.com/watch?v=AVIKFXLCPY8&ab_channel=Hung-yiLee)
- [AI Agent](https://python.langchain.com/docs/modules/agents/)
- [S-BERT; AI-based search engine](https://subirverma.medium.com/semantic-search-with-s-bert-is-all-you-need-951bc710e160)
- [LLM Retrieval--Sentence Embedding](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/)
- [Here is all what you need about retriever](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/)
- [Using decoder model's embedding layer as a retriever; attention mechanism](https://arxiv.org/pdf/2202.08904)
- [One-to-many scoring function, multi-field embeddings; GCL](https://arxiv.org/pdf/2404.08535)

### Model training and tips
- [How and why to pre-train a LLM model](https://arxiv.org/abs/2004.10964)
- [LM-Cocktail; Merge models makes LLM stronger](https://arxiv.org/pdf/2311.13534)
- [Training master and student models](https://www.sbert.net/examples/training/multilingual/README.html)
- [MiniLM; Train a out-performance student model](https://arxiv.org/pdf/2002.10957)
- [A Few-Shot Method to Generate Synthetic Queries with GPT-3 for Training High-Performance Information Retrieval Models](https://arxiv.org/pdf/2202.05144)

### Evaluation matrix for retrieval and re-ranking
- [NDCG](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)
- [MRR](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr)
- [Ranking Metrics for Evaluating Question Answering Systems and Recommendation Systems](https://kaushikshakkari.medium.com/understanding-semantic-search-part-5-ranking-metrics-for-evaluating-question-answering-systems-f3150872d986)

### Pytorch and Cuda optimization
- [Parallelism Strategies (Data, Tensor, Context, Pipeline, Expert) and ZeRO Optimization for Large-Scale PyTorch Model Training](https://levelup.gitconnected.com/training-deep-learning-models-at-ultra-scale-using-pytorch-74c6cbaa814b)
- [Optimizing GPU Memory Usage in PyTorch: A Practical Guide to CUDA Caching Allocator, Gradient Accumulation, and In-Place Operations](https://levelup.gitconnected.com/mastering-gpu-memory-management-with-pytorch-and-cuda-94a6cd52ce54)

### GPT Inference optimization
- [Quantization of GPT model](https://towardsdatascience.com/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172)

### Inference Engine
- [GPU memory management optimization; Page attention and vLLM](https://arxiv.org/pdf/2309.06180)
- [Scheduling and interleaving the LLM generation requests in parallel](https://www.usenix.org/system/files/osdi22-yu.pdf)

### Vector store
- [Qdrant optimizer](https://qdrant.tech/documentation/concepts/optimizer/)
- [Qdrant index](https://qdrant.tech/documentation/concepts/indexing/)
- [Enhancing HNSW for Fast and Filterable Approximate Nearest Neighbor Search with Category, Label, and Range Constraints](https://blog.vasnetsov.com/posts/categorical-hnsw/)

### Prompt Engineer
- [Instruction-tuned model has better performance](https://openreview.net/pdf?id=gEZrGCozdqR)
- [Become a pro prompt engineer with LangGPT](https://github.com/langgptai/LangGPT)
- [Why prompt engineer is so important? What is "CO-STAR" prompt framework?](https://sydney-ai.notion.site/GPT-4-a868f41568174781b52a2f592fce7a58)
- [Prompt template for Llama2](https://arxiv.org/pdf/2402.07483.pdf)

### Function calling/ Tools for LLM
- [Overview of function calling](https://arxiv.org/pdf/2304.08354)
- [What is single-turn function calling](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- [What is multi-turn function calling](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
- [fine-tuning for single-turn function calling](https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060)
- [fine-tuning for multi-turn function calling](https://arxiv.org/pdf/2410.12952)
- [Generate training data for fine-tuning function calling by GPT-4](https://cookbook.openai.com/examples/fine_tuning_for_function_calling)
- [MCPO: Converting stdio-Based MCP Servers into RESTful APIs for Open-WebUI Integration and Custom Python Tool Support](https://mychen76.medium.com/mcpo-supercharge-open-webui-with-mcp-tools-4ee55024c371)

### Use cases
- [Completely local RAG; S-BERT(Retriever)+LangChain(RAG)+LlamaCpp(Llama2)](https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-langchain-llama2-%E5%89%B5%E9%80%A0%E4%BD%A0%E7%9A%84%E5%80%8B%E4%BA%BAllm-d6838febf8c4)
- [Build a local Co-pilot](https://blog.codegpt.co/free-and-private-copilot-the-future-of-coding-with-llms-in-vscode-372330c5b163)
- [LangChain Use cases](https://js.langchain.com/docs/use_cases)

---
## Seq2Seq (Encoder-Decoder) Series
- [Encoder-Decoder Transformer Models: BART and T5](https://medium.com/@lmpo/encoder-decoder-transformer-models-a-comprehensive-study-of-bart-and-t5-132b3f9836ed)
- [Generate Frequently Asked Questions from Presentation Transcripts](https://aclanthology.org/2023.inlg-main.35.pdf)

---
## User Behavior Detection Series
- [Forecasting Long-Tern Time Series Data by Transformer Model](https://arxiv.org/pdf/2211.14730)
- [Forecasting Long-Tern Time Series Data by Inverted Transformer Model](https://arxiv.org/pdf/2310.06625)

---

## Other handbook resources
- [Roadmap of becoming a LLM engineer](https://github.com/mlabonne/llm-course)