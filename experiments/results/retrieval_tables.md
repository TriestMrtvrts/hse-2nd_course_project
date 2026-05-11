# Retrieval Experiments

## preprocessing
| experiment                  | family        | split      |   document_hit@1 |   document_hit@3 |   document_mrr |   document_ndcg@3 |   section_hit@1 |   section_hit@3 |
|:----------------------------|:--------------|:-----------|-----------------:|-----------------:|---------------:|------------------:|----------------:|----------------:|
| preprocess_raw_validation   | preprocessing | validation |                1 |                1 |              1 |                 1 |          0.7917 |          0.8333 |
| preprocess_clean_validation | preprocessing | validation |                1 |                1 |              1 |                 1 |          0.7917 |          0.8333 |

## chunk_size
| experiment              | family     | split      |   document_hit@1 |   document_hit@3 |   document_mrr |   document_ndcg@3 |   section_hit@1 |   section_hit@3 |
|:------------------------|:-----------|:-----------|-----------------:|-----------------:|---------------:|------------------:|----------------:|----------------:|
| chunk_medium_validation | chunk_size | validation |           1      |           1      |         1      |            1      |          0.7917 |          0.8333 |
| chunk_small_validation  | chunk_size | validation |           0.9583 |           1      |         0.9792 |            0.9846 |          0.75   |          0.875  |
| chunk_large_validation  | chunk_size | validation |           0.9167 |           0.9583 |         0.9458 |            0.943  |          0.7083 |          0.7917 |

## chunking
| experiment                        | family   | split      |   document_hit@1 |   document_hit@3 |   document_mrr |   document_ndcg@3 |   section_hit@1 |   section_hit@3 |
|:----------------------------------|:---------|:-----------|-----------------:|-----------------:|---------------:|------------------:|----------------:|----------------:|
| chunk_plain_recursive_validation  | chunking | validation |                1 |                1 |              1 |                 1 |          0.25   |          0.25   |
| chunk_header_recursive_validation | chunking | validation |                1 |                1 |              1 |                 1 |          0.7917 |          0.8333 |

## embedder
| experiment                 | family   | split      |   document_hit@1 |   document_hit@3 |   document_mrr |   document_ndcg@3 |   section_hit@1 |   section_hit@3 |
|:---------------------------|:---------|:-----------|-----------------:|-----------------:|---------------:|------------------:|----------------:|----------------:|
| embedder_deepvk_validation | embedder | validation |           1      |           1      |         1      |            1      |          0.7917 |          0.8333 |
| embedder_e5_validation     | embedder | validation |           1      |           1      |         1      |            1      |          0.7917 |          0.8333 |
| embedder_minilm_validation | embedder | validation |           0.7917 |           0.9583 |         0.8833 |            0.8968 |          0.5    |          0.625  |

## retriever
| experiment                           | family    | split      |   document_hit@1 |   document_hit@3 |   document_mrr |   document_ndcg@3 |   section_hit@1 |   section_hit@3 |
|:-------------------------------------|:----------|:-----------|-----------------:|-----------------:|---------------:|------------------:|----------------:|----------------:|
| retriever_dense_validation           | retriever | validation |           1      |           1      |         1      |             1     |          0.7917 |          0.8333 |
| retriever_ensemble_validation        | retriever | validation |           1      |           1      |         1      |             1     |          0.7083 |          0.875  |
| retriever_ensemble_rerank_validation | retriever | validation |           1      |           1      |         1      |             1     |          0.75   |          0.7917 |
| retriever_bm25_validation            | retriever | validation |           0.9167 |           0.9583 |         0.9458 |             0.943 |          0.5    |          0.7083 |

## held_out_test
| experiment          | family    | split   |   document_hit@1 |   document_hit@3 |   document_mrr |   document_ndcg@3 |   section_hit@1 |   section_hit@3 |
|:--------------------|:----------|:--------|-----------------:|-----------------:|---------------:|------------------:|----------------:|----------------:|
| baseline_dense_test | reference | test    |                1 |                1 |              1 |                 1 |          0.7333 |          0.8333 |
