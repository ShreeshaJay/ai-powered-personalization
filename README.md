# AI-Powered Personalization (Working Draft)

This repository contains the working drafts, code examples, diagrams, and research literature for the book **"AI-Powered Personalization: An Industry Guide to Building Recommender Systems at Scale"** authored by Shreesha Jagadeesh.

## Context for this book

I have been in Machine Learning for nearly a decade. For the last 3+ years, I have been leading ML teams building ML/AI models at Best Buy focused on Recommender Systems, Adtech, Personalization, and Marketing use cases. My contribution as a leader spans both ML Science (algorithm development) and ML Engineering (building out the overall systems). At Best Buy, I am responsible for customer-facing models that serve 100+ million users annually.

When I entered RecSys/Personalization, I noticed that as an industry practitioner, I had to cobble together disparate sources of information — blogs, videos, research articles — to ramp up. This appears to be the case for many others I have spoken to. While there are excellent books on ML systems, they usually focus on general principles and do not go deep into the specifics of building RecSys engines. Nor do they guide engineering leaders on how to implement an enterprise strategy for a portfolio of RecSys/personalization use cases at their company. This book presents a unified narrative on the industry-standard ways of implementing RecSys/Personalization solutions and guides leaders on successful execution.

## 🚀 Book Description

This book is a practical guide to designing, building, and scaling real-world recommender systems. It is aimed at ML engineers and technical product leaders who want to understand modern multi-stage architectures, online deployment strategies, and the use of deep learning, LLMs, and vector databases in recommender systems.

The book includes extensive mini case studies both from the author's experience and from industry engineering blogs that provide insights into architectural design choices. There are additional code tutorials on selected model-related topics, although coding is not the main focus of the book. Hopefully, new entrants to the industry can ramp up much faster by reading this book.

Note: this book will not cover the mechanics of Data Science or Deep Learning fundamentals. Readers are assumed to be either practicing ML Engineers who know how to build supervised ML models in other domains, or engineering leaders looking for architectural patterns, reusability best practices, and what works in the industry.

## Book Outcomes

After reading this book, the reader will be able to:

- Recognize the day-to-day consumer applications where RecSys and Personalization are used and apply them to their own use case
- Understand the specialized requirements for collecting historical training data to train personalization-specific models, and influence enterprise data architecture decisions
- Make smart design choices when building RecSys models to avoid data leakage, train-serve skew, and common pitfalls
- Choose the right personalization ML model for their use case
- Select architectural patterns that minimize inference latency through multi-region deployment, caching, and model optimization
- Develop embeddings to represent customers and items within their domain
- Lead ML teams to build, deploy, and iterate specialized RecSys models for both batch and online inference
- Collaborate with platform teams to effectively A/B test and measure the performance of deployed models in production
- Craft an effective ML strategy that minimizes bespoke model development and redirects investment into common reusable assets

While this is not a code-oriented book, there is just enough code within each chapter to get the gist of each topic, with accompanying tutorials in the repository containing far more details that readers are encouraged to run in their own machines.

## 📚 Table of Contents (Draft — 16 Chapters)

### Part I: Foundations
1. **Introduction to Recommender Systems**
2. **Multi-Stage Architectures and Common Tradeoffs**
3. **Training Data Collection, Setup, and Evaluation**
4. **Scalable Data Pipelines and Feature Management**

### Part II: The Core Pipeline
5. **Retrieval Stage**
6. **Ranking Stage (Basics)**
7. **Advanced Ranking** — DCN, DLRM, MMoE
8. **Value Functions and Re-ranking** — business rules, diversity, MMR
9. **Adtech-specific Recommender Systems** — CTR/CVR, delayed feedback, calibration

### Part III: Representation Learning
10. **Item Embeddings** — from Item2Vec to multimodal fusion
11. **User Embeddings** — sequence models, graph-based approaches (LightGCN)
12. **LLMs for Recommenders** — Semantic IDs, LLMs as direct recommenders, LLM-augmented RecSys

### Part IV: Production & Operations
13. **Online Deployment**
14. **Latency Optimization**
15. **Monitoring and Retraining**
16. **A/B Testing, Experimentation, and Bandits**

### Appendices
- **Appendix A: RexBERT** — a domain-specialized SBERT model for e-commerce
- **Appendix B: Superlinked** — vector compute framework for personalization
- **Appendix C: Hybrid Vector Databases** — combining dense + sparse retrieval at scale

## 📂 Repository Structure

| Folder | Description |
|--------|-------------|
| `chapter5_retrieval/` through `chapter13_deployment/` | Python code tutorials for each chapter |
| `appendix_rexbert/`, `appendix_superlinked/`, `Appendix Hybrid Search/` | Code for the appendices |
| Chapter-specific `README.md` files | Setup instructions, dataset notes, and expected outputs per chapter |

Each chapter's code folder is self-contained with its own requirements, data loaders, training scripts, and evaluation utilities.

## 🛠 How to Contribute

If you are a reviewer, please use Pull Requests or Issues to suggest changes. I am looking for feedback from three reader personas:

1. **Practicing ML Engineers / Data Scientists not yet in RecSys or Personalization** — to see if I am able to convey concepts clearly to other ML engineers.
2. **Experienced ML Engineers already in RecSys / Personalization** — to identify where I can improve the book's depth, fact-check claims, and potentially contribute case studies or fireside stories.
3. **Engineering managers / leaders from other ML or AI domains who lead (or will lead) teams building recommendation use cases** — to see if the chapters around enterprise adoption and strategy resonate.

If you'd like early access to the chapters, please email me directly or open an issue.

---

*Book publication is planned via self-publishing channels (Gumroad / Leanpub). A free online version may be offered in the future.*
