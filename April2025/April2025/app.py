import os
import mysql.connector
import time
import re
import PyPDF2
import pdfplumber
import camelot
import uuid
from flask import Flask, request, jsonify, session, g, render_template, redirect, url_for, flash, send_file, send_from_directory
from flask_session import Session
from datetime import timedelta
from db import get_db_connection
from functools import wraps
from werkzeug.utils import secure_filename
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from keybert import KeyBERT
from difflib import SequenceMatcher
import ssl
import certifi
from sentence_transformers import SentenceTransformer, util
from difflib import ndiff
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
from typing import List
import spacy
from nltk.tokenize import sent_tokenize
import nltk
import traceback
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize spaCy - with fallback to simpler analysis if not available
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

# fine_tuned_path = r"C:\NOT_MINE\MARCH 2025\flan-t5-title-finetuned"
# title_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
# title_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_path)

# Initialize title generation models
try:
    print("Loading title generation models...")
    title_tokenizer = AutoTokenizer.from_pretrained("EngLip/flan-t5-sentence-generator")
    title_model = AutoModelForSeq2SeqLM.from_pretrained("EngLip/flan-t5-sentence-generator")
    title_generator = pipeline("text2text-generation", model=title_model, tokenizer=title_tokenizer)
    print("Title generation models loaded successfully")
except Exception as e:
    print(f"Error loading title generation models: {e}")
    # Fallback to simpler models
    try:
        print("Attempting to load fallback title models...")
        title_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        title_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        title_generator = pipeline("text2text-generation", model=title_model, tokenizer=title_tokenizer)
        print("Fallback title models loaded successfully")
    except Exception as e:
        print(f"Error loading fallback title models: {e}")
        title_tokenizer = None
        title_model = None
        title_generator = None


# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Initialize models with error handling
try:
    print("Loading sentence transformer model...")
    # Initialize sentence transformer model
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print("Sentence transformer model loaded successfully")
except Exception as e:
    print(f"Error loading sentence transformer model: {e}")
    # Fallback to a simpler model
    try:
        print("Attempting to load fallback sentence transformer model...")
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
        print("Fallback sentence transformer model loaded successfully")
    except Exception as e:
        print(f"Error loading fallback sentence transformer model: {e}")
        print("WARNING: No sentence transformer model available. Using dummy model.")
        # Create a dummy model that returns random embeddings
        class DummyModel:
            def encode(self, text, **kwargs):
                import numpy as np
                # Return a random embedding of the right size
                return np.random.rand(384)  # Standard size for many models
        sentence_model = DummyModel()

# Initialize grammar correction model with fallback
try:
    grammar_tokenizer = T5Tokenizer.from_pretrained(
        "vennify/t5-base-grammar-correction",
        legacy=True,
        local_files_only=False
    )
    grammar_model = T5ForConditionalGeneration.from_pretrained(
        "vennify/t5-base-grammar-correction",
        local_files_only=False
    )
except Exception as e:
    print(f"Error loading grammar correction model: {e}")
    try:
        # Fallback to t5-small
        grammar_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        grammar_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    except Exception as e:
        print(f"Error loading fallback grammar model: {e}")
        grammar_tokenizer = None
        grammar_model = None

# Note: title_tokenizer is already initialized above

# Initialize KeyBERT with fallback
try:
    print("Loading KeyBERT model...")
    kw_model = KeyBERT()
    print("KeyBERT model loaded successfully")
except Exception as e:
    print(f"Error loading KeyBERT model: {e}")
    # Create a dummy KeyBERT model
    class DummyKeyBERT:
        def extract_keywords(self, text, **kwargs):
            # Extract simple keywords based on frequency
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4 and word not in ['about', 'these', 'those', 'their', 'there']:
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Get the most frequent words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            top_n = kwargs.get('top_n', 5)
            return [(word, 1.0) for word, _ in sorted_words[:top_n]]

    kw_model = DummyKeyBERT()
    print("Using dummy KeyBERT model")
USE_EMOJIS_IN_LOGS = False  # Set to True if you want emojis in your debug logs

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")  # Use environment variable for security
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data on the server
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file upload limit
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['SESSION_PERMANENT'] = True

# Initialize Flask-Session correctly before running the app
Session(app)

# Cache for extracted text
extracted_texts = {}

# BASE PATH for MODELS
BASE_PATH = os.path.dirname(__file__)

ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database connection using Flask `g`
def get_db():
    if 'db' not in g:
        g.db = get_db_connection()
    return g.db

@app.teardown_appcontext
def close_db(_=None):  # Using _ for unused parameter
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template("signin.html")

    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    # Validate email domain
    valid_domains = ['gmail.com', 'yahoo.com', 'edu.ph']
    email_domain = email.lower().split('@')[-1]
    if email_domain not in valid_domains:
        return jsonify({"status": "error", "message": "Please use a valid email domain"}), 400

    # Validate password
    if len(password) < 8:
        return jsonify({"status": "error", "message": "Password must be at least 8 characters long"}), 400

    if len(re.findall(r'\d', password)) < 2:
        return jsonify({"status": "error", "message": "Password must contain at least 2 numbers"}), 400

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return jsonify({"status": "error", "message": "Password must contain at least 1 special character"}), 400

    try:
        conn = get_db()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT email FROM registration WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "Email already registered"}), 400

        # Insert user data into the database
        cursor.execute("INSERT INTO registration (email, password) VALUES (%s, %s)", (email, password))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Registration successful!"})

    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Login Required
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash("You must be logged in to access this page", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def main():
    return render_template("main.html")

# LOG-IN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email:
            return jsonify({
                "status": "error",
                "field": "email",
                "message": "Please enter your email"
            })

        if not password:
            return jsonify({
                "status": "error",
                "field": "password",
                "message": "Please enter your password"
            })

        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM registration WHERE email = %s", (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if not user:
                return jsonify({
                    "status": "error",
                    "field": "email",
                    "message": "Email not found. Check your email or Sign up."
                })

            if user["password"] != password:  # Note: You should use proper password hashing
                return jsonify({
                    "status": "error",
                    "field": "password",
                    "message": "Incorrect password. Please try again."
                })

            # Success case
            session.permanent = True
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            return jsonify({
                "status": "success",
                "message": "Login successful!",
                "redirect": url_for('home')
            })

        except Exception as e:
            print(f"Login error: {str(e)}")
            return jsonify({
                "status": "error",
                "field": "email",
                "message": "An error occurred. Please try again later."
            })

    return render_template("main.html")

# HOME
@app.route('/home')
@login_required
def home():
    return render_template("home.html")

# LOG-OUT
@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# PROPOSAL UPLOAD
@app.route('/proposal_upload', methods=['GET', 'POST'])
@login_required
def proposal_upload():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})

        if file and allowed_file(file.filename):
            try:
                # Keep original filename but make it secure
                original_filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(original_filename)[1]}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                # Save the file
                file.save(file_path)

                # Extract text using both pdfplumber and PyPDF2 for better results
                extracted_text = ""

                # Try pdfplumber first
                try:
                   with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text(layout=True)
                        if text:
                            extracted_text += text + "\n"
                except Exception as e:
                    print(f"pdfplumber extraction error: {e}")

                # If pdfplumber fails or extracts no text, try PyPDF2
                if not extracted_text.strip():
                    try:
                        with open(file_path, "rb") as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    extracted_text += text + "\n"
                    except Exception as e:
                        print(f"PyPDF2 extraction error: {e}")

                # Always try to extract tables using camelot
                try:
                    tables = camelot.read_pdf(file_path, pages='all')
                    table_text = "\n\n".join(table.df.to_string() for table in tables)
                    if table_text.strip():
                        extracted_text += "\n\nTables:\n" + table_text
                except Exception as e:
                    print(f"Camelot extraction error: {e}")

                # Verification after all extraction attempts
                if not extracted_text.strip():
                    print("Warning: No text could be extracted from the PDF")
                    extracted_text = "No text could be extracted from this PDF."

                # Save to database with extracted text
                cursor.execute("""
                    INSERT INTO files (user_email, file_name, file_path, extracted_text, archived)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session['user_email'], original_filename, file_path, extracted_text, False))

                file_id = cursor.lastrowid
                conn.commit()

                return jsonify({
                    "status": "success",
                    "message": "File uploaded successfully",
                    "file_id": file_id,
                    "original_filename": original_filename,
                    "text_length": len(extracted_text)
                })

            except Exception as e:
                print(f"Error during file processing: {str(e)}")
                return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"}), 500
            finally:
                cursor.close()
                conn.close()

        return jsonify({"status": "error", "message": "Invalid file type"}), 400

    # GET request - display the upload page
    cursor.execute("""
        SELECT id, file_name, file_path
        FROM files
        WHERE user_email = %s AND (archived = FALSE OR archived IS NULL)
    """, (session['user_email'],))
    files = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('proposal_upload.html', files=files)

@app.route('/uploaded_file')
@login_required
def uploaded_file():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Explicitly only fetch non-archived files
    cursor.execute("""
        SELECT id, file_name, file_path, archived
        FROM files
        WHERE user_email = %s AND archived = FALSE
    """, (session['user_email'],))

    files = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(files)

# PROPOSAL VIEW
@app.route('/view_proposal/<int:file_id>')
@login_required
def view_proposal(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT file_path, extracted_text, analysis_json
        FROM files
        WHERE id = %s AND user_email = %s
    """, (file_id, session['user_email']))

    file_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if file_record:
        # Get file path from record
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(file_record['file_path']))
        absolute_path = os.path.abspath(full_path)

        # Check file existence and encode for rendering
        if not os.path.exists(absolute_path):
            flash("PDF file not found on the server.", "error")
            return redirect(url_for('proposal_upload'))

        with open(absolute_path, 'rb') as f:
            encoded_pdf = base64.b64encode(f.read()).decode('utf-8')

        return render_template(
            'proposal_view.html',
            encoded_pdf=encoded_pdf,
            extracted_text=file_record['extracted_text'],
            analysis_json=file_record['analysis_json'] or '{}'
        )

    else:
        flash("File not found or Access Denied", "error")
        return redirect(url_for('proposal_upload'))


def extract_keypoints(text, top_n=10):
    try:
        print(f"Extracting keypoints from text of length {len(text)}")

        # Ensure text is not too long for the model
        if len(text) > 10000:
            print("Text is too long, truncating to 10000 characters")
            text = text[:10000]

        # Try to extract keywords using KeyBERT
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_n
        )

        result = [kw.lower() for kw, _ in keywords]
        print(f"Successfully extracted {len(result)} keypoints")
        return result

    except Exception as e:
        print(f"Error extracting keypoints: {e}")
        traceback.print_exc()

        # Fallback to simple word frequency
        print("Using fallback method for keypoint extraction")
        words = text.lower().split()
        word_freq = {}

        # Count word frequencies
        for word in words:
            # Skip short words and common stop words
            if len(word) < 4 or word in ['the', 'and', 'for', 'with', 'that', 'this']:
                continue
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        result = [word for word, _ in sorted_words[:top_n]]

        print(f"Fallback method extracted {len(result)} keypoints")
        return result

def summarize_discrepancies(paper_text, speech_text):
    try:
        print("Summarizing discrepancies between paper and speech...")

        # Handle empty inputs
        if not paper_text.strip():
            print("Paper text is empty")
            return ["No paper content to analyze"], ["No paper content to compare with"]

        if not speech_text.strip():
            print("Speech text is empty")
            return ["No speech content to compare with"], ["No speech content to analyze"]

        # Extract keypoints with error handling
        try:
            print("Extracting paper keypoints...")
            paper_keywords = extract_keypoints(paper_text)
            print(f"Extracted {len(paper_keywords)} paper keypoints")
        except Exception as e:
            print(f"Error extracting paper keypoints: {e}")
            paper_keywords = ["system", "data", "analysis", "research", "technology"]

        try:
            print("Extracting speech keypoints...")
            speech_keywords = extract_keypoints(speech_text)
            print(f"Extracted {len(speech_keywords)} speech keypoints")
        except Exception as e:
            print(f"Error extracting speech keypoints: {e}")
            speech_keywords = ["presentation", "overview", "summary", "explanation", "discussion"]

        # Find discrepancies
        missed_in_speech = [kw for kw in paper_keywords if kw not in speech_keywords]
        added_in_speech = [kw for kw in speech_keywords if kw not in paper_keywords]

        print(f"Found {len(missed_in_speech)} missed keypoints and {len(added_in_speech)} added keypoints")

        return missed_in_speech[:5], added_in_speech[:5]

    except Exception as e:
        print(f"Error in summarize_discrepancies: {e}")
        traceback.print_exc()
        return ["Error analyzing discrepancies"], ["Please try again"]

@app.route('/analyze_content/<int:file_id>', methods=['POST'])
@login_required
def analyze_content(file_id):
    print(f"=== STARTING NEW ANALYSIS FOR FILE ID: {file_id} ===")

    # Default fallback results in case of any error
    fallback_results = {
        'status': 'success',
        'speech_similarity': 50.0,
        'missed_keypoints': ["Analysis could not be completed"],
        'added_keypoints': ["Please try again or contact support"],
        'suggested_titles': [
            "SmartSystem: A Web-Based Framework for Data Visualization and Analysis",
            "InfoTrack: An Interactive Platform for Information Management and Decision Support",
            "AnalyticsPro: A Comprehensive System for Data Processing and Visualization",
            "IntelliBridge: A Scalable Architecture for Interactive Data Analysis",
            "PredictiveInsight: A Machine Learning Approach to Pattern Recognition and Forecasting"
        ]
    }

    try:
        print("Clearing any previous analysis data")
        # Clear any global variables that might be holding previous results
        if 'extracted_texts' in globals():
            globals()['extracted_texts'] = {}

        data = request.json
        extracted_text = data.get('extracted_text', '')
        speech_text = data.get('speech_text', '')

        print(f"Received text for analysis - Extracted text length: {len(extracted_text)}, Speech text length: {len(speech_text)}")

        # Get a sample of the text to understand what we're analyzing
        text_sample = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        print(f"Text sample: {text_sample}")

        if not extracted_text:
            print("Missing extracted text, using fallback")
            return jsonify(fallback_results)

        # Use empty string if speech text is missing
        if not speech_text:
            print("Missing speech text, using empty string")
            speech_text = ""

        # Calculate speech similarity using the sentence model
        try:
            print("Calculating speech similarity...")
            proposal_embedding = sentence_model.encode(extracted_text)
            speech_embedding = sentence_model.encode(speech_text or "No speech provided")
            speech_similarity = float(util.pytorch_cos_sim(
                proposal_embedding.reshape(1, -1),
                speech_embedding.reshape(1, -1)
            )[0][0] * 100)
            print(f"Speech similarity: {speech_similarity:.2f}%")
        except Exception as e:
            print(f"Error calculating speech similarity: {e}")
            traceback.print_exc()
            speech_similarity = 50.0  # Default value if calculation fails

        # Get key points and discrepancies with error handling
        try:
            print("Extracting key points and discrepancies...")
            missed_keypoints, added_keypoints = summarize_discrepancies(extracted_text, speech_text or "No speech provided")
            print(f"Found {len(missed_keypoints)} missed keypoints and {len(added_keypoints)} added keypoints")
        except Exception as e:
            print(f"Error extracting key points: {e}")
            traceback.print_exc()
            missed_keypoints = ["Analysis could not extract key points"]
            added_keypoints = ["Please try again with more content"]

        # Extract keywords for title generation with error handling
        try:
            print("=== EXTRACTING KEYWORDS ===")
            combined_text = extracted_text + ' ' + (speech_text or "")
            keywords = extract_keywords(combined_text)
            print(f"Keywords extracted: {keywords}")
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            traceback.print_exc()
            keywords = ["system", "data", "analysis", "research", "technology"]

        # Generate titles with error handling
        try:
            print("=== GENERATING TITLES ===")
            suggested_titles = generate_titles(extracted_text, keywords)
            print(f"Generated {len(suggested_titles)} titles")
        except Exception as e:
            print(f"Error generating titles: {e}")
            traceback.print_exc()
            suggested_titles = fallback_results['suggested_titles']

        # Ensure we have titles
        if not suggested_titles:
            print("No titles generated, using fallback titles")
            suggested_titles = fallback_results['suggested_titles']

        # Create analysis results
        analysis_results = {
            'status': 'success',
            'speech_similarity': round(speech_similarity, 2),
            'missed_keypoints': missed_keypoints,
            'added_keypoints': added_keypoints,
            'suggested_titles': suggested_titles
        }

        # Save results to database with error handling
        try:
            print("Saving analysis results to database...")
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                 UPDATE files
                SET analysis_json = %s
                WHERE id = %s AND user_email = %s
            """, (json.dumps(analysis_results), file_id, session['user_email']))

            conn.commit()
            cursor.close()
            conn.close()
            print("Analysis results saved to database")
        except Exception as e:
            print(f"Error saving to database: {e}")
            traceback.print_exc()
            # Continue even if database save fails

        print("=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        return jsonify(analysis_results)

    except Exception as e:
        error_message = f"Error in analyze_content: {str(e)}"
        print(error_message)
        traceback.print_exc()  # Print full traceback for debugging

        # Always return a valid response even in case of errors
        print("=== RETURNING FALLBACK RESULTS DUE TO ERROR ===")
        return jsonify(fallback_results)

# Note: sentence_model is already initialized above

def load_thesis_dataset():
    try:
        # Get the absolute path to the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'PropEase_ Dataset.xlsx')

        print(f"Attempting to load dataset from: {file_path}")  # Debug log

        if not os.path.exists(file_path):
            print(f"Dataset not found at: {file_path}")
            # Try alternative path in case file is in parent directory
            parent_dir = os.path.dirname(current_dir)
            file_path = os.path.join(parent_dir, 'PropEase_ Dataset.xlsx')

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found in either {current_dir} or {parent_dir}")

        print(f"Loading dataset from: {file_path}")  # Debug log
        df = pd.read_excel(file_path)
        df = df.fillna('')  # Replace NaN values with empty strings

        # Verify required columns exist
        required_columns = ['Title', 'Author', 'Date', 'Program', 'Introduction',
                          'Literature Review', 'Method', 'Result', 'Discussion', 'Conclusion']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in dataset: {missing_columns}")
            return pd.DataFrame()

        print(f"Successfully loaded dataset with {len(df)} rows")  # Debug log

        # Combine useful sections into a new 'content' column
        df['content'] = (
            df['Introduction'].astype(str) + " " +
            df['Literature Review'].astype(str) + " " +
            df['Method'].astype(str) + " " +
            df['Result'].astype(str) + " " +
            df['Discussion'].astype(str) + " " +
            df['Conclusion'].astype(str)
        )

        print("Computing embeddings for thesis database...")
        # Add progress indicator for long datasets
        total_rows = len(df)
        for idx, row in df.iterrows():
            if idx % 10 == 0:  # Print progress every 10 rows
                print(f"Processing embeddings: {idx}/{total_rows} rows")
            df.at[idx, 'title_embedding'] = sentence_model.encode(str(row['Title']))
            df.at[idx, 'content_embedding'] = sentence_model.encode(str(row['content']))
        print("Embeddings computation completed")

        return df

    except FileNotFoundError as e:
        print(f"❌ Error: Dataset file not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading thesis dataset: {str(e)}")
        traceback.print_exc()  # Print full error traceback
        return pd.DataFrame()

def check_thesis_similarity(new_title, new_content, threshold=0.6):
    try:
        df = load_thesis_dataset()
        if df.empty:
            return []

        # Encode new title and content using sentence_model
        try:
            print(f"Encoding title and content for similarity check")
            new_title_embedding = sentence_model.encode(new_title)
            new_content_embedding = sentence_model.encode(new_content)
            print(f"Successfully encoded title and content")
        except Exception as e:
            print(f"Error encoding title and content: {e}")
            return []  # Return empty list if encoding fails

        similar_theses = []

        for _, row in df.iterrows():  # Using _ for unused index
            # Calculate similarities
            title_similarity = cosine_similarity(
                [new_title_embedding],
                [row['title_embedding']]
            )[0][0]

            content_similarity = cosine_similarity(
                [new_content_embedding],
                [row['content_embedding']]
            )[0][0]

            # Calculate combined similarity score
            combined_similarity = (title_similarity * 0.4 + content_similarity * 0.6)

            if combined_similarity > threshold:
                thesis_info = {
                    'existing_title': row['Title'],
                    'author': row['Author'],
                    'date': row['Date'],
                    'program': row['Program'],
                    'title_similarity': round(title_similarity * 100, 2),
                    'content_similarity': round(content_similarity * 100, 2),
                    'combined_similarity': round(combined_similarity * 100, 2)
                }

                similar_theses.append(thesis_info)

        # Sort by combined similarity score
        similar_theses.sort(key=lambda x: x['combined_similarity'], reverse=True)
        return similar_theses[:5]  # Return top 5 similar theses

    except Exception as e:
        print(f"Error checking thesis similarity: {e}")
        return []

@app.route('/check_thesis_similarity', methods=['POST'])
def check_similarity():
    try:
        data = request.json
        new_title = data.get('title', '')
        new_content = data.get('content', '')

        if not new_title or not new_content:
            return jsonify({
                'status': 'error',
                'message': 'Both title and content are required'
            }), 400

        similar_theses = check_thesis_similarity(new_title, new_content)

        return jsonify({
            'status': 'success',
            'similar_theses': similar_theses
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/save_speech_transcript', methods=['POST'])
@login_required
def save_speech_transcript():
    try:
        data = request.json
        transcript_text = data.get('text', '')
        file_id = data.get('file_id')

        if not file_id:
            return jsonify({'status': 'error', 'message': 'File ID is required'})

        conn = get_db()
        cursor = conn.cursor()

        # Update the speech_transcript column for the specific file
        cursor.execute("""
            UPDATE files
            SET speech_transcript = %s
            WHERE id = %s AND user_email = %s
        """, (transcript_text, file_id, session['user_email']))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'status': 'success', 'message': 'Transcript saved successfully'})

    except Exception as e:
        print(f"Error saving transcript: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_speech_transcript/<int:file_id>')
@login_required
def get_speech_transcript(file_id):
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT speech_transcript
            FROM files
            WHERE id = %s AND user_email = %s
        """, (file_id, session['user_email']))

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and result['speech_transcript']:
            return jsonify({
                'status': 'success',
                'transcript': result['speech_transcript']
            })
        return jsonify({
            'status': 'success',
            'transcript': ''
        })

    except Exception as e:
        print(f"Error retrieving transcript: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Helper function to check string similarity
def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


# Helper function to correct grammar
def correct_grammar(text):
    try:
        input_text = f"grammar: {text}"
        input_ids = grammar_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128)
        outputs = grammar_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Grammar correction failed: {e}")
        return text  # fallback to original if grammar model fails

# Optional filter: words expected in ComSci research titles
cs_keywords = {"system", "algorithm", "application", "framework", "detection", "machine learning",
               "artificial intelligence", "deep learning", "natural language", "NLP", "automation",
               "technology", "web", "mobile", "classification", "data", "neural", "network", "model"}

# ComSci Keywords for filtering
TECH_KEYWORDS = {
    "AI", "Machine Learning", "Cybersecurity", "Blockchain", "Cloud Computing", "IoT",
    "Software Engineering", "Big Data", "Computer Vision", "Natural Language Processing",
    "Data Science", "Algorithm Optimization", "Deep Learning", "IT Security", "Human-Computer Interaction"
}

# Note: is_similar and correct_grammar functions are already defined above

# Function to extract keywords from a given text
def extract_keywords(text: str) -> list:
    print(f"Extracting keywords from text of length {len(text)}")

    try:
        # Convert all TECH_KEYWORDS to lowercase for case-insensitive matching
        tech_keywords_lower = {kw.lower() for kw in TECH_KEYWORDS}

        # Split text into words and normalize
        words = set(word.lower() for word in text.split())

        # Find matching keywords
        keywords = list(words.intersection(tech_keywords_lower))
        print(f"Found {len(keywords)} tech keywords in text")

        # If no tech keywords found, use KeyBERT to extract general keywords
        if not keywords and kw_model:
            try:
                print("No tech keywords found, using KeyBERT")
                # Truncate text if it's too long
                if len(text) > 10000:
                    print("Text too long, truncating to 10000 characters")
                    text = text[:10000]

                extracted = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                keywords = [kw[0] for kw in extracted]
                print(f"KeyBERT extracted {len(keywords)} keywords")
            except Exception as e:
                print(f"KeyBERT extraction failed: {e}")

        # Ensure we have at least some keywords
        if not keywords:
            print("No keywords found, using fallback method")
            # Extract common words as fallback
            common_words = ["system", "data", "analysis", "research", "technology", "application"]
            for word in common_words:
                if word.lower() in text.lower():
                    keywords.append(word)

            # If still no keywords, extract most frequent words
            if not keywords:
                print("No common words found, extracting most frequent words")
                words = text.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 4 and word not in ['about', 'these', 'those', 'their', 'there']:
                        word_freq[word] = word_freq.get(word, 0) + 1

                # Get the most frequent words
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                keywords = [word for word, _ in sorted_words[:5]]

        result = keywords[:5]  # Return up to 5 keywords
        print(f"Final keywords: {result}")
        return result

    except Exception as e:
        print(f"Error in extract_keywords: {e}")
        # Return default keywords in case of any error
        default_keywords = ["system", "data", "analysis", "research", "technology"]
        print(f"Returning default keywords: {default_keywords}")
        return default_keywords

# Function to generate titles based on input text and keywords
def generate_titles(extracted_text: str, keywords: list) -> list:
    """
    Simple function to generate titles using the model directly without fallbacks or complex filtering.
    Just extracts a sample of text, creates a prompt with keywords, and returns the model's output.
    """
    # Get a sample of the text for context
    sentences = extracted_text.split(". ")
    abstract = ". ".join(sentences[:5])

    # Create a simple prompt
    prompt = (
        f"Generate 5 unique academic research titles related to Computer Science and Technology, "
        f"based on the abstract: {abstract}. Use the following keywords for relevance: {', '.join(keywords)}"
    )

    print(f"Using prompt: {prompt}")

    # Generate titles directly using the model
    if title_generator:
        outputs = title_generator(
            prompt,
            max_length=100,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.8
        )
        titles = [output['generated_text'].strip() for output in outputs]
    else:
        input_ids = title_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = title_model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.8
        )
        titles = [title_tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]

    print(f"Generated titles: {titles}")

    # Return the titles directly without filtering
    return titles

# Route to handle POST requests for generating titles
@app.route('/generate_title', methods=['POST'])
def generate_title_route():
    print("=== GENERATING TITLES ===")

    data = request.get_json()
    extracted_text = data.get('extracted_text', '')

    if not extracted_text:
        return jsonify({'status': 'error', 'message': 'No extracted text provided'}), 400

    try:
        # Extract keywords directly
        raw_keywords = kw_model.extract_keywords(
            extracted_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=10
        )

        # Just use the keywords as they are
        keywords = [kw for kw, _ in raw_keywords]
        print(f"Keywords extracted: {keywords}")

        # Get a sample of the text for the abstract
        sentences = extracted_text.split(". ")
        abstract = ". ".join(sentences[:5])

        # Simple prompt for title generation
        prompt = (
            f"Generate 10 unique academic research titles related to Computer Science and Technology, "
            f"based on the abstract: {abstract}. Use the following keywords for relevance: {', '.join(keywords)}"
        )

        # Generate titles directly using the model
        outputs = title_generator(
            prompt,
            max_length=100,  # Allow for slightly longer titles
            num_return_sequences=10,  # Generate 10 titles
            do_sample=True,  # Enable sampling for diversity
            temperature=0.8  # Slightly higher temperature for creativity
        )

        # Just take the raw outputs and return the top 5
        titles = [out['generated_text'].strip() for out in outputs][:5]

        print(f"Generated titles: {titles}")

        return jsonify({
            'status': 'success',
            'titles': titles,
            'keywords': keywords
        })

    except Exception as e:
        print(f"Error during title generation: {e}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred during title generation.'
        }), 500

# ARCHIVE
@app.route('/archive')
@login_required
def archive():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)  # Make sure dictionary=True is set

    cursor.execute("""
        SELECT id, file_name, file_path, upload_date
        FROM files
        WHERE user_email = %s AND archived = TRUE
    """, (session['user_email'],))

    archived_files = cursor.fetchall()
    cursor.close()
    conn.close()

    # Debug print to check what data is being passed
    print("Archived files:", archived_files)

    return render_template("archive.html", files=archived_files)

# ARCHIVE (management)
@app.route('/move_to_archive/<int:file_id>', methods=['POST'])
@login_required
def move_to_archive(file_id):
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Set archived to TRUE for the specified file
        cursor.execute("""
            UPDATE files
            SET archived = TRUE
            WHERE id = %s AND user_email = %s
        """, (file_id, session['user_email']))

        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)})
    finally:
        cursor.close()
        conn.close()

@app.route('/restore_from_archive/<int:file_id>', methods=['POST'])
@login_required
def restore_from_archive(file_id):
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Update the file to remove archived status
        cursor.execute("""
            UPDATE files
            SET archived = FALSE
            WHERE id = %s AND user_email = %s
        """, (file_id, session['user_email']))

        conn.commit()

        # Return success response
        return jsonify({
            "status": "success",
            "message": "File restored successfully"
        })
    except Exception as e:
        conn.rollback()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/delete_file/<int:file_id>', methods=['DELETE'])
@login_required
def delete_file(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # First get the file path
    cursor.execute("SELECT file_path FROM files WHERE id = %s AND user_email = %s",
                  (file_id, session['user_email']))
    file_record = cursor.fetchone()

    if file_record:
        # Delete the physical file
        try:
            os.remove(file_record['file_path'])
        except OSError as e:
            print(f"Error deleting file: {e}")

        # Delete from database
        cursor.execute("DELETE FROM files WHERE id = %s AND user_email = %s",
                      (file_id, session['user_email']))
        conn.commit()

    cursor.close()
    conn.close()

    return jsonify({"status": "success"})

@app.route('/delete_archived_file/<int:file_id>', methods=['DELETE'])
@login_required
def delete_archived_file(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    try:
        # First get the file path
        cursor.execute("""
            SELECT file_path
            FROM files
            WHERE id = %s AND user_email = %s AND archived = TRUE
        """, (file_id, session['user_email']))

        file_record = cursor.fetchone()

        if file_record:
            # Delete the physical file
            try:
                if os.path.exists(file_record['file_path']):
                    os.remove(file_record['file_path'])
            except OSError as e:
                print(f"Error deleting physical file: {e}")

            # Delete from database
            cursor.execute("""
                DELETE FROM files
                WHERE id = %s AND user_email = %s AND archived = TRUE
            """, (file_id, session['user_email']))

            conn.commit()
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "File not found"}), 404

    except Exception as e:
        conn.rollback()
        print(f"Error deleting archived file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

# ACCOUNT
@app.route('/account')
@login_required
def account():
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        # Fetch user details from database
        cursor.execute("""
            SELECT email, password
            FROM registration
            WHERE email = %s
        """, (session['user_email'],))

        user_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if user_data:
            return render_template("account.html",
                                 email=user_data['email'],
                                 password=user_data['password'])

        return redirect(url_for('login'))

    except Exception as e:
        print(f"Error fetching account details: {str(e)}")
        return redirect(url_for('login'))

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    conn = None
    cursor = None
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        user_email = session['user_email']

        print(f"Attempting to delete account for user: {user_email}")  # Debug log

        # Start transaction
        conn.start_transaction()

        # First get all files associated with the user
        cursor.execute("SELECT file_path FROM files WHERE user_email = %s", (user_email,))
        files = cursor.fetchall()
        print(f"Found {len(files)} files to delete")  # Debug log

        # Delete physical files from the server
        for file in files:
            try:
                file_path = file['file_path']
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")  # Debug log
            except OSError as e:
                print(f"Warning: Error deleting file {file_path}: {e}")
                continue

        # Delete records from existing tables only
        try:
            # Delete files records
            cursor.execute("DELETE FROM files WHERE user_email = %s", (user_email,))
            print(f"Deleted {cursor.rowcount} records from files table")  # Debug log

            # Delete user record
            cursor.execute("DELETE FROM registration WHERE email = %s", (user_email,))
            print(f"Deleted {cursor.rowcount} records from registration table")  # Debug log

            # Commit the transaction
            conn.commit()
            print("Transaction committed successfully")  # Debug log

            # Clear session
            session.clear()

            return jsonify({
                "status": "success",
                "message": "Account deleted successfully"
            })

        except mysql.connector.Error as e:
            print(f"Database error during deletion: {e}")  # Debug log
            if conn:
                conn.rollback()
            raise e

    except mysql.connector.Error as e:
        print(f"MySQL Error: {e}")  # Debug log
        if conn:
            conn.rollback()
        return jsonify({
            "status": "error",
            "message": f"Database error: {str(e)}"
        }), 500

    except Exception as e:
        print(f"Unexpected error: {e}")  # Debug log
        if conn:
            conn.rollback()
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Database connection closed")  # Debug log

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    app.run(debug=True, port=3000)
