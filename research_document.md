# Research Document: Technology Selection for Construction Cost Estimation Chatbot

## 1. Vector Database Choice: Pinecone

Pinecone was selected over FAISS for the following reasons:

- **Low Latency**: Sub-100ms retrieval times at scale, which is critical for real-time query response in an interactive chatbot.
- **Fully Managed Infrastructure**: Requires no server maintenance, scaling, or monitoring.
- **High Availability**: 99.9% uptime with robust SLAs, ideal for production deployment.
- **Ease of Integration**: Native support via Python SDK, JSON-based metadata storage, and compatibility with LangChain.

While FAISS is open-source and suitable for local or research use, Pinecone is more suitable for cloud-based and scalable solutions.

**Reference**: [Pinecone vs FAISS Comparison](https://www.pinecone.io/learn/faiss-vs-pinecone/)

---

## 2. LLM Model Choice: Qwen3 via OpenRouter

Among the available free models on OpenRouter (DeepSeek, LLaMA 4, Mistral Nemo, Qwen3), **Qwen3** was selected due to:

- **Multimodal Reasoning**: Capable of understanding both text and visual layout features (advantageous for interpreting floorplans).
- **Spatial Awareness**: Superior performance in spatial and geometric reasoning — relevant for door/window positioning, area computation, and schedule layout analysis.
- **Mathematical Handling**: Handles measurement-based calculations and estimation breakdowns accurately.

**Provider**: [OpenRouter](https://openrouter.ai)  
**Model ID Used**: `qwen:Qwen1.5-7B-Chat`

---

## 3. RAG Architecture Overview

The system follows a hybrid Retrieval-Augmented Generation pipeline using LangChain:

1. **PDF Upload**
2. **OCR & Text Extraction** (via PyMuPDF + PaddleOCR)
3. **Door/Window Schedule Parsing**
4. **Text Embedding** using OpenCLIP
5. **Vector Storage** in Pinecone
6. **User Query**
7. **Relevant Chunks Retrieved from Pinecone**
8. **Prompted to Qwen3 via LangChain**
9. **Answer Generated and Returned**

**Diagram Reference**: See `Data Flow Diagram.pdf` (attached separately)

---

## 4. Summary

By combining **Pinecone** and **Qwen3**, the system achieves:

- Fast and scalable document processing
- High accuracy in layout and schedule interpretation
- Smooth RAG-based interaction via LangChain
- Production-readiness with minimal infrastructure overhead

This stack was selected specifically to balance performance, cost (free-tier LLM), and task suitability in the domain of architectural floorplan analysis.
