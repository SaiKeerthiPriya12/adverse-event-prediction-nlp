from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Load model and vectorizer
with open('adverse_event_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Predict adverse event
def predict_adverse_event(text_input, model, vectorizer):
    preprocessed_text = preprocess_text(text_input)
    X_test = vectorizer.transform([preprocessed_text])
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    predicted_class = y_pred[0]
    probability = y_prob[0][predicted_class]
    return predicted_class, probability

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        print("Hi")
        # Simple authentication (replace with proper auth in production)
        if username == 'admin' and password == 'admin':
            session['user'] = username
            return redirect(url_for('predict'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = ''
    if request.method == 'POST':
        text_input = request.form['text_input']
        predicted_class, probability = predict_adverse_event(text_input, model, vectorizer)
        if predicted_class == 1:
            prediction = f"The input text indicates a potential adverse event with a probability of {probability:.2f}"
        else:
            prediction = "The input text does not indicate an adverse event."
            print("predicition is: ",prediction)
        session['score'] = probability

    return render_template('prediction.html', prediction=prediction)

@app.route('/graph')
def graph():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('performance.html')

@app.route('/score')
def score():
    if 'user' not in session:
        return redirect(url_for('login'))
    result = session.get('score', 0)
    return render_template('score.html', result=result)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
