import os
import mysql.connector
import time
import re
import PyPDF2
import pdfplumber
import camelot
import uuid
import pickle
from flask import Flask, request, jsonify, session, g, render_template, redirect, url_for, flash, send_file, send_from_directory
from flask_session import Session
from datetime import timedelta
from db import get_db_connection
from functools import wraps
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer 

from flask import Flask, request, jsonify
from t5_title_generator import generate_title  # This imports your function
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from flask import Flask, request, jsonify
from keybert import KeyBERT
import pdfplumber
import os
from difflib import SequenceMatcher

# Load model and tokenizer once
model_path = r"C:\Users\JaneBenneth\OneDrive\Documents\THESIS\Propease_March2025\t5_title_generator_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
kw_model = KeyBERT() 
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
BASE_PATH = r"C:\Users\JaneBenneth\OneDrive\Documents\THESIS\Propease_March2025"

# SVM Model
svm_model_path = os.path.join(BASE_PATH, "svm_model.pkl")
if not os.path.exists(svm_model_path):
    raise FileNotFoundError(f"Model file not found: {svm_model_path}")
with open(svm_model_path, "rb") as model_file:
    svm_model = pickle.load(model_file)

# TF-IDF Vectorizer
vectorizer_path = os.path.join(BASE_PATH, "tfidf_vectorizer.pkl")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)
# Check if vectorizer is properly fitted
if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_:
    print("Vectorizer is fitted. Ready to use.")
else:
    print("Vectorizer is not fitted. You need to train it.")

ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database connection using Flask `g`
def get_db():
    if 'db' not in g:
        g.db = get_db_connection()
    return g.db

@app.teardown_appcontext
def close_db(error=None):
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

# Load grammar correction model
grammar_tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
grammar_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def correct_grammar(text):
    try:
        input_text = f"grammar: {text}"
        input_ids = grammar_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128)
        outputs = grammar_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception:
        return text  # fallback to original if grammar model fails

# Optional filter: words expected in ComSci research titles
cs_keywords = {"system", "algorithm", "application", "framework", "detection", "machine learning", 
               "artificial intelligence", "deep learning", "natural language", "NLP", "automation", 
               "technology", "web", "mobile", "classification", "data", "neural", "network", "model"}

@app.route('/generate_title', methods=['POST'])
def generate_title():
    data = request.get_json()
    extracted_text = data.get('extracted_text')

    if not extracted_text:
        return jsonify({'status': 'error', 'message': 'No extracted text provided'}), 400

    try:
        # ✅ Extract keywords
        keywords = kw_model.extract_keywords(
            extracted_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=5
        )
        keyword_list = [kw for kw, _ in keywords]
        keyword_str = ', '.join(keyword_list)

        # ✅ Stronger prompt with CS emphasis
        prompt = (
            f"Generate 5 academic research project titles related to computer science. "
            f"Context: {extracted_text.strip()}. Use keywords: {keyword_str}."
        )

        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

        outputs = model.generate(
            inputs,
            max_length=32,
            min_length=8,
            num_beams=20,
            num_return_sequences=20,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            early_stopping=True
        )

        raw_titles = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

        # ✅ Filter by length and uniqueness
        filtered_titles = []
        for title in raw_titles:
            if len(title.split()) < 4:
                continue
            if all(not is_similar(title, existing) for existing in filtered_titles):
                filtered_titles.append(title)
            if len(filtered_titles) == 5:
                break

        # ✅ Fill remaining if under 5
        if len(filtered_titles) < 5:
            for title in raw_titles:
                if title not in filtered_titles and len(title.split()) >= 4:
                    filtered_titles.append(title)
                if len(filtered_titles) == 5:
                    break

        # ✅ Correct grammar
        corrected_titles = [correct_grammar(title) for title in filtered_titles]

        # 🔍 Optional CS filter: keep only if it mentions common CS terms (loose check)
        final_titles = [title for title in corrected_titles if any(cs_word.lower() in title.lower() for cs_word in cs_keywords)]
        if len(final_titles) < 5:
            final_titles += [t for t in corrected_titles if t not in final_titles][:5 - len(final_titles)]

        # ✅ Debug
        print("[DEBUG] Extracted Keywords:", keyword_list)
        print("[DEBUG] Final Titles:")
        for t in final_titles:
            print(f"- {t}")

        return jsonify({
            'status': 'success',
            'titles': final_titles,
            'keywords': keyword_list
        })

    except Exception as e:
        print(f"❌ Error during title generation: {e}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred during title generation.'
        }), 500
    
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
@app.route('/proposal_view/<int:file_id>')
@login_required
def proposal_view(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    
    # Fetch file details based on the logged-in user
    cursor.execute("SELECT file_path, extracted_text FROM files WHERE id = %s AND user_email = %s", 
                   (file_id, session['user_email']))
    
    file_record = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if file_record:
        extracted_text = file_record.get('extracted_text', '')  # Ensure it’s not None
        print("Extracted Text:", extracted_text.encode('cp1252', errors='ignore').decode('cp1252'))

        paragraphs = [p.strip() for p in extracted_text.split('\n\n') if p.strip()]
        return render_template('proposal_view.html', extracted_text='\n\n'.join(paragraphs))
    else:  
        flash("File not found or Access Denied", "error")
        return redirect(url_for('proposal_upload'))

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
    app.run(debug=True, port=3000)  