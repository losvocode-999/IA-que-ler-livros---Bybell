import sys
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar, QFileDialog,
                             QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import PyPDF2
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='bybel_client_log.txt', filemode='w')
logger = logging.getLogger(__name__)

class BibliAIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bybel - Sua Biblioteca Inteligente")
        self.setGeometry(100, 100, 1000, 800)
        self.setup_ui()
        self.book_text = ""
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
                response = requests.post('http://localhost:5000/process', 
                                         json={'book_text': self.book_text, 'question': question})
                if response.status_code == 200:
                    answer = response.json()['answer']
                else:
                    answer = "Erro ao processar a pergunta. Tente novamente mais tarde."
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