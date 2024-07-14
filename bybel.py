import sys
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar, QFileDialog,
                             QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='bibliai_log.txt', filemode='w')
logger = logging.getLogger(__name__)

def preprocess_text(text):
    logger.debug("Iniciando pré-processamento do texto")
    sentences = sent_tokenize(text)
    logger.info(f"Texto tokenizado em {len(sentences)} sentenças")
    return sentences

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
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        logger.info(f"Modelos carregados: {self.model_name}, SentenceTransformer, GPT-2, e BERT")

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
        
        processed_text = preprocess_text(book_text)
        context = self.get_relevant_context(processed_text, question)
        answer = self.generate_answer(question, context)
        
        return answer

class BibliAIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bybel - Sua Biblioteca Inteligente")
        self.setGeometry(100, 100, 1000, 800)
        self.setup_ui()
        self.book_text = ""
        self.ai_processor = AIProcessor()
        logger.info("Interface gráfica inicializada.")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Estilo geral
        self.setStyleSheet("""
            QWidget {
                background-color: #2C3E50;
                color: #ECF0F1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QLineEdit, QTextEdit {
                background-color: #34495E;
                border: 1px solid #7F8C8D;
                border-radius: 5px;
                padding: 8px;
                color: #ECF0F1;
                font-size: 14px;
            }
            QProgressBar {
                border: 1px solid #7F8C8D;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #27AE60;
                border-radius: 5px;
            }
            QLabel {
                font-size: 16px;
            }
        """)

        # Header
        header_layout = QVBoxLayout()
        
        header_label = QLabel("Bybel - Sua Biblioteca Inteligente")
        header_label.setStyleSheet("font-size: 32px; font-weight: bold; margin-bottom: 20px; color: #E74C3C;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(header_label)
        
        logo_label = QLabel(self)
        pixmap = QPixmap("artwork.png")
        logo_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(logo_label)
        
        main_layout.addLayout(header_layout)

        # Stacked Widget for different pages
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Page 1: File Selection
        file_page = QWidget()
        file_layout = QVBoxLayout(file_page)

        file_label = QLabel("Selecione um arquivo PDF para começar:")
        file_layout.addWidget(file_label)

        file_selection_layout = QHBoxLayout()
        self.file_path = QLineEdit(self)
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("Nenhum arquivo selecionado")
        file_selection_layout.addWidget(self.file_path)

        select_button = QPushButton("Selecionar PDF", self)
        select_button.clicked.connect(self.select_file)
        file_selection_layout.addWidget(select_button)

        file_layout.addLayout(file_selection_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        file_layout.addWidget(self.progress_bar)

        self.stacked_widget.addWidget(file_page)

        # Page 2: Q&A
        qa_page = QWidget()
        qa_layout = QVBoxLayout(qa_page)

        question_label = QLabel("Digite sua pergunta:")
        qa_layout.addWidget(question_label)

        self.question_input = QLineEdit(self)
        qa_layout.addWidget(self.question_input)

        ask_button = QPushButton("Perguntar", self)
        ask_button.clicked.connect(self.ask_question)
        qa_layout.addWidget(ask_button)

        self.answer_output = QTextEdit(self)
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("A resposta aparecerá aqui...")
        qa_layout.addWidget(self.answer_output)

        self.stacked_widget.addWidget(qa_page)

    def select_file(self):
        logger.info("Selecionando arquivo PDF")
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Selecionar Arquivo PDF", "", "PDF Files (*.pdf);;All Files (*)", options=options)
        if file_name:
            logger.info(f"Arquivo selecionado: {file_name}")
            self.file_path.setText(file_name)
            self.load_pdf(file_name)
            self.stacked_widget.setCurrentIndex(1)
        else:
            logger.warning("Nenhum arquivo selecionado")

    def load_pdf(self, file_name):
        try:
            logger.info(f"Carregando conteúdo do PDF: {file_name}")
            with open(file_name, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                self.book_text = ""
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    self.progress_bar.setValue(int((page_num / num_pages) * 100))
                    page = reader.pages[page_num]
                    self.book_text += page.extract_text()
            self.progress_bar.setValue(100)
            logger.info("PDF carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar PDF: {str(e)}")
            self.file_path.setText("Erro ao carregar arquivo PDF")

    def ask_question(self):
        question = self.question_input.text()
        if question:
            logger.info(f"Pergunta recebida: {question}")
            self.answer_output.setText("Processando...")
            self.process_question(question)
        else:
            logger.warning("Pergunta vazia")

    def process_question(self, question):
        def background_task():
            try:
                answer = self.ai_processor.process_book_and_answer(self.book_text, question)
                self.worker.update_answer.emit(answer)
            except Exception as e:
                logger.error(f"Erro ao processar pergunta: {str(e)}")
                self.worker.update_answer.emit("Erro ao processar a pergunta. Tente novamente mais tarde.")

        self.thread = QThread()
        self.worker = Worker(background_task)
        self.worker.update_answer.connect(self.display_answer)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def display_answer(self, answer):
        self.answer_output.setText(answer)
        logger.info("Resposta atualizada na interface")

class Worker(QObject):
    finished = pyqtSignal()
    update_answer = pyqtSignal(str)

    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        self.task()
        self.finished.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BibliAIGUI()
    window.show()
    sys.exit(app.exec())