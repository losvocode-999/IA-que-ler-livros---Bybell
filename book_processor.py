import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import torch

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_text(text):
    logger.debug("Iniciando pré-processamento do texto")
    sentences = sent_tokenize(text)
    logger.info(f"Texto tokenizado em {len(sentences)} sentenças")
    return sentences

class AIProcessor:
    def __init__(self):
        logger.info("Inicializando o processador de IA")
        self.model_name = "unicamp-dl/ptt5-base-portuguese-vocab"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.nlp = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer)
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        logger.info(f"Modelos carregados: {self.model_name} e SentenceTransformer")

    def get_relevant_context(self, sentences, question, max_tokens=1000):
        logger.debug(f"Buscando contexto relevante para a pergunta: {question}")
        
        # Encode the question and all sentences
        question_embedding = self.sentence_model.encode(question, convert_to_tensor=True)
        sentence_embeddings = self.sentence_model.encode(sentences, convert_to_tensor=True)

        # Compute cosine similarities
        cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

        # Sort sentences by similarity score
        sorted_sentences = sorted(zip(sentences, cosine_scores), key=lambda x: x[1], reverse=True)

        context = []
        total_tokens = 0
        for sentence, score in sorted_sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if total_tokens + sentence_tokens <= max_tokens:
                context.append(sentence)
                total_tokens += sentence_tokens
                logger.debug(f"Adicionada sentença com score {score:.4f}: {sentence[:50]}...")
            else:
                break

        logger.info(f"Contexto selecionado com {total_tokens} tokens")
        return " ".join(context)

    def generate_answer(self, question, context):
        logger.debug("Gerando resposta detalhada")
        input_text = f"""Gere uma resposta detalhada e abrangente para a seguinte pergunta, usando o contexto fornecido.
Pergunta: {question}
Contexto: {context}
Instruções:
1. Forneça uma resposta completa, bem estruturada e coerente.
2. Inclua informações relevantes do contexto e faça inferências quando apropriado.
3. Divida a resposta em tópicos ou parágrafos para melhor compreensão.
4. Use exemplos, analogias ou explicações adicionais para esclarecer conceitos complexos.
5. Se houver incertezas ou informações faltantes, mencione isso na resposta e sugira possíveis interpretações.
6. Relacione a resposta com conhecimentos gerais relevantes, quando aplicável.
7. Conclua com um breve resumo ou reflexão sobre a pergunta e a resposta.
Resposta:"""

        result = self.nlp(input_text, max_length=800, min_length=300, num_beams=5, length_penalty=2.0, no_repeat_ngram_size=3, do_sample=True, temperature=0.8)
        answer = result[0]['generated_text']
        logger.info(f"Resposta gerada com {len(answer)} caracteres")
        logger.debug(f"Resposta: {answer}")
        return answer

    def process_book_and_answer(self, book_text, question):
        logger.info(f"Processando livro e gerando resposta para: {question}")
        
        processed_text = preprocess_text(book_text)
        context = self.get_relevant_context(processed_text, question)
        answer = self.generate_answer(question, context)
        
        return answer

# Exemplo de uso
if __name__ == "__main__":
    processor = AIProcessor()
    book_text = "Texto do livro aqui..."  # Substitua pelo texto real do livro
    question = "Qual é a principal ideia deste livro e como ela se relaciona com o contexto atual?"
    answer = processor.process_book_and_answer(book_text, question)
    print(f"Resposta: {answer}")
    logger.info("Processo concluído")