from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import pymysql  # Use pymysql for MySQL connection

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Initialize global variables
vector_store = None
embeddings = OpenAIEmbeddings()
current_user = None  # Track the currently logged-in user

# In-memory database for demonstration purposes
db = {
    'users': [],
    'documents': [],
    'sessions': [],

    'addUser ': lambda user: db['users'].append(user),
    'addDocument': lambda doc: db['documents'].append(doc),
    'createSession': lambda userId: db['sessions'].append({'userId': userId, 'token': os.urandom(16).hex()})
}


def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_connection():
    """Create a connection to the MySQL database using pymysql."""
    connection = None
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',  # Replace with your MySQL username
            password='root',  # Replace with your MySQL password
            database='chatbot'  # Replace with your database name
        )
        print("Connection to MySQL DB successful")
    except pymysql.MySQLError as e:
        print(f"The error '{e}' occurred")
    return connection


def process_document(filepath, extension):
    """Process the uploaded document and create embeddings."""
    global vector_store

    if extension == 'pdf':
        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(pages)
    else:  # For txt files
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(text)

    # Create embeddings and store them in the vector store
    vector_store = FAISS.from_documents(texts, embeddings)


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if current_user is None:
        return jsonify({'error': 'User  not authenticated'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the document
        extension = filename.rsplit('.', 1)[1].lower()
        process_document(filepath, extension)

        # Insert document metadata into MySQL
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("INSERT INTO documents (filename, user_id) VALUES (%s, %s)", (filename, current_user['id']))
        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({'message': 'File successfully uploaded and processed'}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle user queries based on uploaded documents."""
    if current_user is None:
        return jsonify({'error': 'User  not authenticated'}), 401

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if vector_store is None:
        return jsonify({'error': 'No documents uploaded yet'}), 400

    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        temperature=0.7
    )

    return jsonify({
        'answer': response.choices[0].message.content,
        'sources': [doc.metadata for doc in docs]
    })


@app.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    global current_user
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = next((u for u in db['users'] if u['email'] == email and u['password'] == password), None)
    if user:
        current_user = user
        return jsonify({'message': 'Login successful', 'username': user['name']}), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401


@app.route('/signup', methods=['POST'])
def signup():
    """Handle user signup."""
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if any(u['email'] == email for u in db['users']):
        return jsonify({'error': 'User  already exists'}), 400

    user = {
        'id': len(db['users']) + 1,
        'name': name,
        'email': email,
        'password': password
    }
    db['addUser '](user)
    return jsonify({'message': 'Signup successful! Please login.'}), 200


@app.route('/logout', methods=['POST'])
def logout():
    """Handle user logout."""
    global current_user
    current_user = None
    return jsonify({'message': 'Logout successful'}), 200


@app.route('/documents', methods=['GET'])
def get_documents():
    """Fetch the list of uploaded documents from the database."""
    if current_user is None:
        return jsonify({'error': 'User  not authenticated'}), 401

    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM documents WHERE user_id = %s", (current_user['id'],))
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
