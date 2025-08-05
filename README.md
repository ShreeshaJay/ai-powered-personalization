# ai-powered-personalization



\# AI-Powered Personalization (Working Draft)



This repository contains the working drafts, code examples, diagrams, and research literature for the book "AI-Powered Personalization: An Industry Guide to Building Recommender Systems at Scale" authored by Shreesha Jagadeesh.



\## Context for this book

I have been in Machine Learning for nearly a decade. Since the last 3 years, I have been leading ML teams building ML/AI models for Best Buy focusing on Recommender Systems, Adtech, Personalization \& Marketing use cases. My contribution as a leader is both on the ML Science (algorithm development) as well as ML Engineering to build out the overall systems. At Best Buy, I am responsible for customer-facing models that serve 100Million+ users annually. 



when I entered Recsys/Personalization, I noticed that as an industry practitioner, I had to cobble together disparate sources of information including blogs, videos, research articles to ramp up. This appears to be the case with a lot of others I have spoken to. While there are excellent books on ML systems, they usually focus on the general principles and do not go too deep into the specifics on how to build RecSys Engines in particular. Nor do they guide engineering leaders on how to implement an enterprise strategy for implementing a portfolio of Recsys/personalization use cases at their company.  So, this book presents a unified narrative on the industry standard ways of implementing RecSys/Personalization solutions and also guides leaders on successful execution. 



\## 🚀 Book Description



This book is a practical guide to designing, building, and scaling real-world recommender systems. It is aimed at ML engineers and technical product leaders who want to understand modern multi-stage architectures, online deployment strategies, and use of deep learning, LLMs, and vector databases in recommender systems.

There will be extensive mini case studies both from the author’s experience as well as industry’s engineering blogs that provide insights into the architectural design choices. There will be additional code repositories as well on selective model related topics but coding is not the main focus of the book. Hopefully, new entrants to the industry can now ramp up much faster by reading this book.



Note that this book will not cover the mechanics of Data Science, Deep Learning. The readers are assumed to be either practicing ML Engineers who know how to build supervised ML models in other domains or Engineering leaders who are looking more for the architectural patterns, best practices for reusability and what works in the industry. 



\## Book Outcomes 



After reading this book, the reader will be able to:

•	Recognize the day-to-day consumer applications where RecSys and Personalization are used and be able to apply it to their own use case. 

•	Understand specialized requirements to collect historical training data to train Personalization specific model and influence enterprise data architecture decisions. 

•	Make smart design choices when building RecSys models to avoid data leakage, train-serve skew, etc.

•	Make suitable choices on choosing the right Personalization ML model for their use case. 

•	Choose architectural patterns that minimize inference latencies through multi-region deployment, caching, etc.

•	Develop Embeddings for representing customers and items within their domain

•	Lead ML teams to build, deploy and iterate specialized RecSys models for batch \& online inference use cases.

•	Collaborate with platform teams to effectively A/B test and measure the performance of the deployed models in PROD. 

•	Technology leaders in this domain will be able to craft an effective ML strategy to minimize bespoke model development and instead redirect their investments into common reusable assets. 



While this is not a code oriented book, there will be just enough code to get the gist of certain topics. 



\## 📚 Table of Contents (Draft)



1\. Introduction to Recommender Systems  

2\. Multi-Stage Architectures and Common Tradeoffs

3\. Training Data Collection, Setup and Evaluation  

4\. Scalable Data Pipelines and Feature Management  

5\. Retrieval Stage  

6\. Ranking Stage   

7\. Re-ranking Stage Representation Learning for Users and Items  

8\. Online Deployment Incorporating LLMs in Recommenders  

9\. Latency Optimization   

10\. Monitoring and Re-training   

11\. A/B Testing and Experimentation 

12\. AutoML for RecSys

13.Product Embeddings 

14\. Customer Embeddings 

15\. LLMs for Recsys and Future Directions



Appendix



\## 📂 Repository Structure



| Folder | Description |

|--------|-------------|

| `chapters/` | Markdown drafts of each book chapter |

| `code/`     | Python scripts and notebooks for examples and case studies |

| `images/`   | Diagrams, charts, and figures used in the book |

| `literature/` | PDFs and notes of key academic and industry papers |



---



\## 🛠 How to Contribute



If you are a reviewer, please use Pull Requests or Issues to suggest changes.  I am looking for primarily three personas 

1\) Practicing ML Engineer or Data Scientist but not yet in RecSys/Personalization domain -> to see if I am able to convey concepts to other ML Engineers layman

2\) Experienced ML Engineer already in RecSys/Personalization -> to see where I can improve the book's depth, fact check and potentially contribute case studies or fireside stories.

3\) Engineering manager/leader from another ML/AI domain who lead or about to lead ML teams that develop these Recommendations use caes -> to see if my chapters around the enterprise adoption resonate. 



If you'd like early access to the chapters, please email me directly or open a request.



---



