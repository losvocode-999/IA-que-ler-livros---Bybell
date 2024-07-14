import flask
from flask import request, jsonify
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

app = flask.Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='bybel_server_log.txt', filemode='w')
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)

class AIProcessor:
    def __init__(self):
        logger.info("Inicializando o processador de IA")
        self.model_name = "t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.nlp = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer)
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.reasoning_model_name = "gpt2"
        self.reasoning_tokenizer = AutoTokenizer.from_pretrained(self.reasoning_model_name)
        self.reasoning_model = AutoModelForCausalLM.from_pretrained(self.reasoning_model_name)
        logger.info(f"Modelos carregados: {self.model_name}, SentenceTransformer, GPT-2")

    def preprocess_text(self, text):
        logger.debug("Iniciando pré-processamento do texto")
        sentences = sent_tokenize(text)
        logger.info(f"Texto tokenizado em {len(sentences)} sentenças")
        return sentences

    def get_relevant_context(self, sentences, question, max_tokens=1000):
        logger.debug(f"Buscando contexto relevante para a pergunta: {question}")
        
        question_embedding = self.sentence_model.encode(question, convert_to_tensor=True)
        sentence_embeddings = self.sentence_model.encode(sentences, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

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
        input_text = f"Pergunta: {question}\nContexto: {context}\n"
        reasoning_input = self.reasoning_tokenizer.encode(input_text, return_tensors="pt")
        reasoning_output = self.reasoning_model.generate(reasoning_input, max_length=800, num_beams=5, length_penalty=2.0, no_repeat_ngram_size=3, do_sample=True, temperature=0.8)
        answer = self.reasoning_tokenizer.decode(reasoning_output[0], skip_special_tokens=True)
        logger.info(f"Resposta gerada com {len(answer)} caracteres")
        logger.debug(f"Resposta: {answer}")
        return answer

    def process_book_and_answer(self, book_text, question):
        logger.info(f"Processando livro e gerando resposta para: {question}")
        
        processed_text = self.preprocess_text(book_text)
        context = self.get_relevant_context(processed_text, question)
        answer = self.generate_answer(question, context)
        
        return answer

ai_processor = AIProcessor()

@app.route('/process', methods=['POST'])
def process_question():
    data = request.json
    book_text = data['book_text']
    question = data['question']
    
    try:
        answer = ai_processor.process_book_and_answer(book_text, question)
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}")
        return jsonify({'error': 'Erro ao processar a pergunta. Tente novamente mais tarde.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)