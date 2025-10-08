data={"intents": [
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
import json
import string
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout

lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
doc_x=[]
doc_y=[]
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens=nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words=sorted(set(words))
classes=sorted(set(classes))
print(words)
print(classes)
print(doc_x)
print(doc_y)
training=[]
out_empty=[0]*len(classes)



for idx, doc in enumerate(doc_x):
    bow=[]
    text=lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row=list(out_empty)
    output_row[classes.index(doc_y[idx])]=1

    training.append([bow, output_row])

random.shuffle(training)

training=np.array(training,dtype=object)

train_X=np.array(list(training[:,0]))
train_y=np.array(list(training[:,1]))
input_shape=(len(train_X[0]),)
output_shape=len(train_y[0])

epochs=500
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation='softmax'))


adam = tf.keras.optimizers.Adam(learning_rate=0.01)


model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())
model.fit(x=train_X, y=train_y, epochs=500, verbose=1)
# After model.fit(...)
model.save("chatbot_model.h5")
print("Model saved!")

import pickle
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

def clean_text(text):
    tokens=nltk.word_tokenize(text)
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text,vocab):
    tokens=clean_text(text)
    bow=[0]*len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word==w:
                bow[idx]=1
    return np.array(bow)
def pred_class(text, vocab,labels):
    bow=bag_of_words(text, vocab)
    result=model.predict(np.array([bow]))[0]
    thresh=0.2
    y_pred=[[idx,res] for idx, res in enumerate(result) if res>thresh]

    y_pred.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    tag=intents_list[0]
    list_of_intents=intents_json["intents"]
    for i in list_of_intents:
        if i["tag"]==tag:
            result=random.choice(i["responses"])
            break
    return result
while True:
    message=input("")
    intents=pred_class(message, words, classes)
    result=get_response(intents,data)
    print(result)