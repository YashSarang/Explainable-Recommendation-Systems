# Explainable Recommendation Systems
## A brief basic level implementation

An explainable recommendation model provides personalized suggestions while transparently articulating the reasoning behind each recommendation to enhance user trust and understanding.
Explainable recommendation models can be model intrinsic or model agnostic.

Basic techniques for recommendation models are:
1. Content Based Filtering
2. Collaborative Filtering (Collaborative Filtering is Less Intuitive)
3. Hybrid Recommendation System (Content + Collaborative)
4. Hidden Factors and Topic model (HFT) based

Refer the Presentation for further explaination & outputs over the above approaches,
Codes for the same are available in the corresponding ipynb files.

## Further complex approaches to implement Explainable Recommendations:

Latent Factor Approach,
Knowledge Graph Approach
Deep Learning Based Approach, 
Reinforcement Learning based Approach,
LLM based or LLM Integration Approach.

Future Unexplored Viable Approach:
Recognising Important Features in the model which lead to Recommendation output, and integrating it with a custom fine-tuned LLM to provide verbal reasoning.

### Previosly Implemented working approaches:

* Reinforcement Knowledge Graph Reasoning for Explainable Recommendation
https://github.com/orcax/PGPR 
Xian, Y., Fu, Z., Muthukrishnan, S., De Melo, G., & Zhang, Y. (2019). Reinforcement Knowledge Graph Reasoning for Explainable Recommendation. ArXiv. https://doi.org/10.1145/3331184.3331203 

* KBE4ExplainableRecommendation
https://github.com/evison/KBE4ExplainableRecommendation/ 
Qingyao Ai, Vahid Azizi, Xu Chen, Yongfeng Zhang.
Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation
Algorithms. 2018, 11(9). Special Issue Collaborative Filtering and Recommender Systems. https://doi.org/10.3390/a11090137 

Note: Convert the codes to be suitable for cuda-cores to save time exponentially on the above mentioned codebases.

### Further Research Papers to look into:

1. A Survey of Explainable E-Commerce Recommender Systems - https://ieeexplore.ieee.org/document/10101904
2. Explainable Recommendation: A Survey and New Perspectives - https://arxiv.org/abs/1804.11192
3. Faithfully Explainable Recommendation via Neural Logic Reasoning - https://arxiv.org/abs/2104.07869
4. Generate Natural Language Explanations for Recommendation - https://arxiv.org/abs/2101.03392 