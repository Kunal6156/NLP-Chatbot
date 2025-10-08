from flask import Flask, render_template, request, jsonify
import json
import string
import random
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

# Load your existing chatbot data
data = {"intents": [
  {"tag": "greetings",
  "patterns": ["hello","hey","hi","good day","Greetings","what's up?","how is it going?"],
  "responses": ["Hello!","Hey!","What can I do for you?"]
  },
  {"tag": "name",
  "patterns": ["what is your name","name","what's your name","who are you","what should I call you"],
  "responses": ["You can call me Ribot","I'm Ribot","I'm Ribot your virtual assistant"]
  },
  {"tag": "courses",
  "patterns": ["what courses are available", "how many courses are there in this college"],
  "responses": ["Bharati Vidyapeeth College of engineering Pune has been in direct partnership with London Metropolitan University, \nUK to provide enviable higher education in IT and Business to students in Pokhara.\nFor Bachelors Degree in Information Technology we have been offering the specialization in BSc (Hons) Computing.\nFor Bachelors in Business Administration we have been offering the followings:\n\n1. BBA (Marketing) with International Business \n\n2. BBA (Accounting & Finance) with International Business\n\n3. BBA (International Business)"]
  },
  {"tag": "courseDuration",
  "patterns": ["how long will be Btech or MBA course", "how long will it take to complete Btech or MBA course"],
  "responses": ["Our college offers 4 year long Btech course and 1 and half year long MBA course."]
  },
  {"tag": "Location",
  "patterns": ["location","where is it located","what is the location of the college"],
  "responses": ["Bharati Vidyapeeth College of engineering Pune is located in Mharashtra Pune, near Hospital."]
  },
  {"tag": "semesters",
  "patterns": ["how many semesters are there in a year","how many semesters one should study in a year"],
  "responses": ["There are two semesters in a year."]
  },
  {"tag": "semDuration",
  "patterns": ["how many months are there in a semester","how long will be a single semester"],
  "responses": ["The single semester will be around 4 months."]
  },
  {"tag": "studentRequirements",
  "patterns": ["what are the student requirements for admission","entry requirements","admission requirements"],
  "responses": ["Academic Level\nNEB +2 overall aggregate of 2.2 CGPA (55%) or above with each subject (theory and practical) grade D+ or above, and SEE Mathematics score of C+ ( 50%) or above.\nFor A-Levels, a minimum of 3.5 credits and atleast a grade of D and above.\n\nEnglish Proficiency\nEnglish NEB XII Marks greater or equals to 60% or 2.4 GPA\nFor Level 4 or Year 1 BIT\nPass in General Paper or English Language or IELTS 5.5 or PTE 47/ Meeting UCAS Tariff points of 80.\nFor Level 4 or Year 1 BBA\nPass in General Paper or English Language or IELTS 5.5 or PTE 47/ Meeting UCAS Tariff points of 96."]
  },
  {"tag": "classes",
  "patterns": ["how many classes will be there in a day","how long are the classes?"],
  "responses": ["There may be two or three classes per day. Each class will be of 1 hour and 30 minutes."]
  },
  {"tag": "teachingStyle",
  "patterns": ["what is the teaching style of this college?","Is the teaching pattern different from other college?","what is the teaching format?"],
  "responses": ["Our college has different teaching patterns than other colleges of Mharastra. We adopt a British teaching methodology, following the LTW techniques which stands for Lecture, Tutorial and Workshop.\nYou can provide us with your contact details and our counselors shall reach out to you and provide you with further details."]
  },
  {"tag": "exams",
  "patterns": ["what are the exams like?","What is the exam pattern"],
  "responses": ["There are assignments which carry more weight than your written exams. The assignments have deadlines which you should not exceed if you want to get better marks."]
  },
  {"tag": "hours",
  "patterns": ["what are your hours","when are you guys open","what your hours of operation"],
  "responses": ["You can message us here at any hours. But our college premises will be open from 7:00 am to 5:00 pm only."]
  },
  {"tag": "funActivities",
  "patterns": ["will there be any extra curriculum activities?","does the college conducts any fun program"],
  "responses": ["Yes, Of course. Our college not only provides excellent education but also encourage students to take part in different curriculum activities. The college conducts yearly programs like Sports meet, Carnival, Holi festival, and Christmas. \n Also our college has basketball court, badminton court, table tennis, chess, carrom board and many more refreshment zones."]
  },
  {"tag": "facilities",
  "patterns": ["what facilities are provided by the college?","what are the facilities of college for students", "what are the college infrastructures "],
  "responses": ["With excellent education facilities, Our College provides various other facilities like 24 hours internet, library, classes with AC, discusson room, canteen, parking space, and student service for any students queries."]
  },
  {"tag": "fee",
  "patterns": ["how much is the college fee","what is the fee structure"],
  "responses": ["Course BIT\nAdmission fee=RS 96,000\nYear 1\nUniversity and Exam fee= RS 100,000 Each semester fee=RS 69,000 Total fee= RS 334,000\nYear 2\nUniversity and Exam fee= RS 100,000 Each semester fee=RS 69,000 Total fee= RS 238,000\nYear 3\nUniversity and Exam fee= RS 100,000 Each semester fee=RS 69,000 Total fee= RS 238,000\nGrandTotal fee= RS 810,000\n\nCourse BBA\nAdmission fee=RS 96,000\nYear 1\nUniversity and Exam fee= RS 100,000 Each semester fee=RS 52,000 Total fee= RS 300,000\nYear 2\nUniversity and Exam fee= RS 100,000 Each semester fee=RS 52,000 Total fee= RS 204,000\nYear 3\nUniversity and Exam fee= RS 100,000 Each semester fee=RS 52,000 Total fee= RS 204,000\nYear 4\nUniversity and Exam fee= RS 50,000 Semester fee=RS 52,000 Total fee= RS 102,000\nGrandTotal fee= RS 810,000"]
  },
  {"tag": "goodbye",
  "patterns": ["cya","See you later","Goodbye","I am leaving","Have a Good Day","bye","see ya"],
  "responses": ["Sad to see you go :(","Talk you later","Goodbye"]
  },
  {"tag": "invalid",
    "patterns": ["","gvsd","asbhk"],
    "responses": ["Sorry, can't understand you", "Please give me more info", "Not sure I understand"]
  },
  {"tag": "thanks",
    "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
    "responses": ["Happy to help!", "Any time!", "My pleasure"]
  }
]}

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the pre-trained model and preprocessed data
print("Loading model and data...")
try:
    model = load_model("chatbot_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open("words.pkl", "rb") as f:
        words = pickle.load(f)
    with open("classes.pkl", "rb") as f:
        classes = pickle.load(f)
    print("Words and classes loaded successfully!")
except Exception as e:
    print(f"Error loading words/classes: {e}")
    words = []
    classes = []

# Prediction functions
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]), verbose=0)[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    intents = pred_class(message, words, classes)
    response = get_response(intents, data)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)