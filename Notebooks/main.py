import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from datasets import load_dataset

dataset = load_dataset("JuliaWolken/chunks.csv", split="train")
chunks_df = dataset.to_pandas()


model = SentenceTransformer('JuliaWolken/fine_tuned_model_with_triplets')


original_chunks = chunks_df['Chunk'].tolist()


chunk_embeddings = model.encode(original_chunks, convert_to_tensor=True)


tokenizer = AutoTokenizer.from_pretrained('DiTy/cross-encoder-russian-msmarco')
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained('DiTy/cross-encoder-russian-msmarco')


def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=True)

def find_relevant_chunks(question_embedding, top_k=5):
    cosine_similarities = cosine_similarity(question_embedding.cpu().numpy(), chunk_embeddings.cpu().numpy()).flatten()
    num_candidates = top_k * 10  # Adjust to get more candidates for re-ranking
    top_indices = cosine_similarities.argsort()[-num_candidates:][::-1]
    return [original_chunks[i] for i in top_indices]


def re_rank(question, candidate_chunks):
    inputs = tokenizer([question] * len(candidate_chunks), candidate_chunks, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        scores = cross_encoder_model(**inputs).logits.squeeze()
    ranked_indices = scores.argsort(descending=True)
    return [candidate_chunks[i] for i in ranked_indices]


def find_relevant_chunks_with_reranking(question, top_k=5):
    question_embedding = embed_texts([question])
    candidate_chunks = find_relevant_chunks(question_embedding, top_k=top_k)
    ranked_chunks = re_rank(question, candidate_chunks) if len(candidate_chunks) > 1 else candidate_chunks
    return ranked_chunks[:top_k]


def answer_question(question):

    if not question or len(question) < 10:
        return "Пожалуйста, задайте вопрос. Количество символов должно превышать 10."


    if not re.search(r'[а-яА-Я]', question):
        return "Простите, на этом языке я пока не говорю. Попробуем еще раз?"


    top_chunks = find_relevant_chunks_with_reranking(question, top_k=5)


    if not top_chunks:
        return "Ничего не нашлось. Я только учусь, сформулируйте вопрос иначе, пожалуйста"


    return "\n\n".join([f"Answer {i+1}: {chunk}" for i, chunk in enumerate(top_chunks)])

# Set up Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="Question Answering Model",
    description="Здравствуйте! Задайте мне вопрос на русском о работе пунктов выдачи WB, и я постараюсь найти самые лучшие ответы."
)

# Launch the Gradio interface with shareable link
iface.launch()
