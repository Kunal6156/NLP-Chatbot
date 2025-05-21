# 🎓 College Info Chatbot

A simple AI-powered chatbot that uses Natural Language Processing (NLP) with NLTK and Deep Learning with TensorFlow to answer queries related to **Bharati Vidyapeeth College of Engineering, Pune**. The bot understands user input, classifies it using a neural network, and provides relevant responses from predefined intents.

---

## 📚 Features

* Answers questions about:

  * Available courses and their durations
  * College location and infrastructure
  * Admission requirements
  * Teaching style and class schedule
  * Exam patterns and extracurricular activities
  * Fee structure and more
* Text preprocessing using **NLTK** (tokenization and lemmatization)
* Intent classification using a custom-built **Neural Network**
* Bag of Words model for feature representation
* Easily extendable and customizable JSON-based intent structure

---

## 🛠️ Tech Stack

* **Python 3.8+**
* **NLTK** for natural language processing
* **TensorFlow / Keras** for training the intent classification model
* **NumPy** for numerical computations
* **JSON** for intent data storage

---

## 🔧 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/college-chatbot.git
cd college-chatbot
```

2. **Install required dependencies**

```bash
pip install nltk tensorflow numpy
```

3. **Download NLTK data (if not already installed)**

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

---

## 📁 Project Structure

```
.
├── chatbot.py         # Main script with training + chat loop
├── intents.json       # Intents file with tags, patterns, and responses
├── README.md          # Project documentation
└── model.h5           # Saved trained model (optional if saved)
```

---

## 🚀 How to Run

```bash
python chatbot.py
```

The chatbot will start and prompt for input. Type a message like:

```
> what is your name
> what courses are available
> how long is the BTech course
> bye
```

---

## 💡 How It Works

1. **Data Preprocessing**:

   * Tokenizes and lemmatizes the input patterns.
   * Converts patterns to a Bag-of-Words representation.

2. **Model Training**:

   * A feed-forward neural network is trained on the bag-of-words inputs to classify into one of the predefined tags (intents).

3. **Inference**:

   * User input is tokenized, cleaned, and converted into BoW.
   * The trained model predicts the most likely intent.
   * A random response associated with the intent is returned.

---

## 🧠 Training Details

* Model: Feed-forward neural network
* Layers: `128 → Dropout → 64 → Dropout → Output`
* Activation: `ReLU` for hidden layers, `Softmax` for output
* Loss: `Categorical Crossentropy`
* Optimizer: `Adam`
* Epochs: `500`

---

## 🔄 Future Improvements

* Integrate GUI or Web App using **Tkinter**, **Flask**, or **Streamlit**
* Use **TF-IDF** or **Word Embeddings** instead of Bag-of-Words
* Add **context handling** for better conversation flow
* Save and load model weights for faster startup

---

## 🤝 Contributing

Pull requests and feature additions are welcome! Please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is licensed under the MIT License.
