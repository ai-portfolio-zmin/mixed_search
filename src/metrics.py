import numpy as np



def ndcg_per_query(ranked_relevance, k):
    ranked_relevance = np.array(ranked_relevance)
    if len(ranked_relevance) < k:
        k = len(ranked_relevance)
    ranked_relevance = ranked_relevance[:k]

    dcg = 0
    for i, rel in enumerate(ranked_relevance,start=1):
        dcg+=(2**rel - 1)/ np.log2(i+1)

    relevance_sorted = np.sort(ranked_relevance)[::-1]
    idcg = 0
    for i, rel in enumerate(relevance_sorted,start=1):
        idcg+=(2**rel - 1)/ np.log2(i+1)

    if idcg ==0:
        return 0
    else:
        return dcg/idcg

def ndcg(ranked_relevance_list, k):
    ranked_relevance_list = np.array(ranked_relevance_list)
    result = []
    for ranked_relevance in ranked_relevance_list:
        result.append(ndcg_per_query(ranked_relevance, k))
    return np.mean(result)

def mrr(ranked_relevance_list, k):
    ranked_relevance_list = np.array(ranked_relevance_list)
    assert len(ranked_relevance_list)>0
    result = 0
    valid = 0
    for ranked_relevance in ranked_relevance_list:
        if len(ranked_relevance) < k:
            k = len(ranked_relevance)
        ranked_relevance = ranked_relevance[:k]
        if max(ranked_relevance) ==0:
            continue
        else:
            valid+=1
            for i, v in enumerate(ranked_relevance):
                if v>0:
                    result += 1/(i+1)
                    break
    return result/valid if valid>0 else 0

def recall_per_query(ranked_relevance, k):
    ranked_relevance = np.array(ranked_relevance)
    if len(ranked_relevance) < k:
        k = len(ranked_relevance)
    ranked_relevance_k = ranked_relevance[:k]
    if (ranked_relevance>0).sum() ==0:
        return 0
    else:
        return (ranked_relevance_k>0).sum() / (ranked_relevance>0).sum()

def recall(ranked_relevance_list, k):
    ranked_relevance_list = np.array(ranked_relevance_list)
    result = []
    for ranked_relevance in ranked_relevance_list:
        result.append(recall_per_query(ranked_relevance, k))
    return np.mean(result)