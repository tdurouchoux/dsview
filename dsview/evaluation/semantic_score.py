import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine

from dsview.config import ModelType, load_model_config
from dsview.models import get_model_provider

nltk.download("stopwords")

semantic_score_model_config = load_model_config(ModelType.SEMANTIC_SCORE)
model_provider = get_model_provider(semantic_score_model_config)

stop_words = set(stopwords.words("english"))


def tokenize(sentence: str) -> list[str]:
    word_list = word_tokenize(sentence)

    return [word for word in word_list if word.lower() not in stop_words]


def semantic_score(
    reference: str,
    candidate: str,
) -> dict[str, float]:
    """Calculate BERTScore-like metric using embeddings"""
    # Split texts into words
    candidate_words = tokenize(candidate)
    reference_words = tokenize(reference)

    # Get embeddings for each word
    candidate_embeddings = [model_provider.embed(word) for word in candidate_words]
    reference_embeddings = [model_provider.embed(word) for word in reference_words]

    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(candidate_words), len(reference_words)))

    for i, cand_emb in enumerate(candidate_embeddings):
        for j, ref_emb in enumerate(reference_embeddings):
            similarity_matrix[i, j] = cosine(cand_emb, ref_emb)

    # Calculate precision (max similarity for each candidate word)
    precision = np.mean(np.max(similarity_matrix, axis=1))

    # Calculate recall (max similarity for each reference word)
    recall = np.mean(np.max(similarity_matrix, axis=0))

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
