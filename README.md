# 🎓 College Info Chatbot

A simple AI-powered chatbot that uses Natural Language Processing (NLP) with NLTK and Deep Learning with TensorFlow to answer queries related to **Bharati Vidyapeeth College of Engineering, Pune**. The bot understands user input, classifies it using a neural network, and provides relevant responses from predefined intents.

## 📚 Features

### Chatbot Capabilities
* **College Information**: Answers questions about:
  * Available courses (BTech, BBA, MBA) and their durations
  * College location and infrastructure
  * Admission requirements and student eligibility
  * Teaching style (British LTW methodology) and class schedule
  * Exam patterns and assessment structure
  * Extracurricular activities and facilities
  * Detailed fee structure for all programs

### NLP & AI Features
* **Text Preprocessing**: NLTK tokenization and lemmatization
* **Intent Classification**: Custom-built Neural Network with TensorFlow
* **Bag of Words Model**: Feature extraction from text
* **Pattern Matching**: Trained on multiple patterns per intent
* **Web Interface**: Modern, responsive chat UI
* **Real-time Responses**: Fast inference with typing indicators

---

## 🛠️ Tech Stack

### Backend
* **Python 3.8+**
* **Flask 2.3** - Web framework
* **TensorFlow/Keras 2.13** - Deep learning model
* **NLTK** - Natural language processing
* **NumPy** - Numerical computations

### Frontend
* **HTML5/CSS3** - Modern chat interface
* **JavaScript** - Real-time interactions
* **Responsive Design** - Mobile-friendly UI

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Kunal6156/NLP-Chatbot.git
cd NLP-Chatbot
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (Auto-downloaded on first run)
If needed, manually download:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### 5. Run the Application
```bash
python app.py
```

The chatbot will be available at: **http://localhost:5000**

---

## 📁 Project Structure

```
NLP-Chatbot/
│
├── app.py                  # Flask web application with integrated NLP
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Chat interface (HTML/CSS/JS)
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

---

## 🚀 How to Use

### Web Interface
1. Run `python app.py`
2. Open browser to `http://localhost:5000`
3. Start chatting with Ribot!

### Example Queries
```
👤 User: Hello
🤖 Ribot: Hello! What can I do for you?

👤 User: What courses are available?
🤖 Ribot: [Lists all available courses]

👤 User: What is the fee structure?
🤖 Ribot: [Provides detailed fee breakdown]

👤 User: Where is the college located?
🤖 Ribot: [Gives location details]
```

---

## 🧠 How It Works

### 1. Data Preprocessing
* **Tokenization**: Splits text into individual words
* **Lemmatization**: Converts words to base form (e.g., "running" → "run")
* **Cleaning**: Removes punctuation and normalizes text

### 2. Feature Extraction
* **Bag of Words (BoW)**: Converts text into numerical vectors
* Each word in vocabulary becomes a feature
* Binary representation (1 if word present, 0 otherwise)

### 3. Model Training
* **Architecture**: Feed-forward Neural Network
  ```
  Input (BoW) → Dense(128, ReLU) → Dropout(0.5) 
              → Dense(64, ReLU) → Dropout(0.3) 
              → Dense(Output, Softmax)
  ```
* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam (learning_rate=0.01)
* **Training**: 200-500 epochs with shuffled data

### 4. Intent Classification
* User input → Preprocessed → BoW vector
* Model predicts intent probabilities
* Threshold filtering (confidence > 0.2)
* Highest probability intent selected

### 5. Response Generation
* Intent mapped to response list
* Random response selected for variety
* Returned to user via web interface

---

## 🎯 Training Details

### Model Architecture
| Layer | Type | Units | Activation | Dropout |
|-------|------|-------|------------|---------|
| Input | Dense | 128 | ReLU | 50% |
| Hidden | Dense | 64 | ReLU | 30% |
| Output | Dense | N (intents) | Softmax | - |

### Hyperparameters
* **Epochs**: 200-500
* **Optimizer**: Adam
* **Learning Rate**: 0.01
* **Loss**: Categorical Crossentropy
* **Batch Size**: Full dataset (small dataset)

### Performance
* **Training Accuracy**: ~98-100%
* **Inference Time**: <50ms per query
* **Model Size**: ~500KB

---

## 🌐 Deployment

### Quick Deploy to Cloud

#### Option 1: Render (Recommended)
1. Add `gunicorn` to requirements:
   ```bash
   pip install gunicorn
   pip freeze > requirements.txt
   ```
2. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Push to GitHub
4. Go to [render.com](https://render.com)
5. Create new Web Service → Connect GitHub repo
6. Deploy! ✨

#### Option 2: Railway
1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Deploy from GitHub → Select repo
4. Auto-deployed! 🚀

#### Option 3: Heroku
```bash
# Create Procfile and runtime.txt
echo "web: gunicorn app:app" > Procfile
echo "python-3.11.9" > runtime.txt

# Deploy
heroku login
heroku create nlp-chatbot-app
git push heroku master
heroku open
```

### Environment Variables
No environment variables required for basic deployment!

---

## 📊 Intent Categories

The chatbot handles **15+ intent categories**:

| Category | Examples |
|----------|----------|
| Greetings | "hello", "hi", "hey" |
| Identity | "what is your name", "who are you" |
| Courses | "what courses available", "programs offered" |
| Duration | "how long is BTech", "course length" |
| Location | "where is college", "location" |
| Admission | "entry requirements", "admission criteria" |
| Classes | "class schedule", "daily classes" |
| Teaching | "teaching style", "methodology" |
| Exams | "exam pattern", "assessment" |
| Facilities | "college infrastructure", "amenities" |
| Activities | "extracurricular", "sports" |
| Fees | "fee structure", "cost" |
| Thanks | "thank you", "thanks" |
| Goodbye | "bye", "see you" |

---

## 🔄 Customization

### Adding New Intents

Edit the `data` dictionary in `app.py`:

```python
{"tag": "scholarships",
 "patterns": [
     "do you offer scholarships",
     "financial aid available",
     "scholarship programs"
 ],
 "responses": [
     "Yes! We offer merit-based and need-based scholarships...",
     "Our college provides various scholarship options..."
 ]
}
```

Restart the app to retrain the model with new intents.

### Modifying UI
Edit `templates/index.html` to customize:
* Colors and theme
* Chat bubble styles
* Header text
* Animations

---

## 🔄 Future Improvements

### Planned Features
- [ ] **Model Persistence**: Save/load trained model to avoid retraining
- [ ] **Context Handling**: Multi-turn conversations with memory
- [ ] **Advanced NLP**: TF-IDF or Word2Vec embeddings
- [ ] **Voice Input**: Speech-to-text integration
- [ ] **Analytics Dashboard**: Track popular queries
- [ ] **Multi-language Support**: Hindi and regional languages
- [ ] **Database Integration**: Store conversation history
- [ ] **Admin Panel**: Update intents without code changes

### Optimization Ideas
* Cache trained model in production
* Use lighter models (DistilBERT) for faster inference
* Implement RAG for dynamic information retrieval
* Add spell correction for typos

---

## ⚠️ Troubleshooting

### Common Issues

**1. Model training takes too long**
```python
# Reduce epochs in app.py
model.fit(x=train_X, y=train_y, epochs=100, verbose=0)  # Instead of 200
```

**2. Memory issues**
```bash
# Use CPU-only TensorFlow
pip install tensorflow-cpu
```

**3. NLTK download errors**
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**4. Port already in use**
```python
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=8080)
```

---

## 📈 System Requirements

### Minimum
* **Python**: 3.8+
* **RAM**: 2GB
* **Storage**: 500MB
* **Internet**: For NLTK downloads

### Recommended
* **Python**: 3.10+
* **RAM**: 4GB
* **Storage**: 1GB
* **CPU**: Multi-core for faster training

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** changes: `git commit -m 'Add feature'`
4. **Push** to branch: `git push origin feature-name`
5. **Open** a Pull Request

### Contribution Ideas
* Add new intents and responses
* Improve model architecture
* Enhance UI/UX
* Add unit tests
* Improve documentation

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Kunal**
* GitHub: [@Kunal6156](https://github.com/Kunal6156)
* Project: [NLP-Chatbot](https://github.com/Kunal6156/NLP-Chatbot)

---

## 🙏 Acknowledgments

* **NLTK** for natural language processing tools
* **TensorFlow** for deep learning framework
* **Flask** for web framework
* **Bharati Vidyapeeth College of Engineering** for domain inspiration

---

## 📞 Support

If you have any questions or issues:
* Open an [Issue](https://github.com/Kunal6156/NLP-Chatbot/issues)
* Star ⭐ the repo if you found it helpful!

---

<div align="center">

**Made with ❤️ and 🤖 by Kunal**

[⬆ Back to Top](#-college-info-chatbot)

</div>