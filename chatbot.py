import numpy as np
import random
import json
import pickle
import nltk.tokenize
from stempel import StempelStemmer
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, request

app = Flask(__name__)
nltk.download('punkt')
stemmer = StempelStemmer.default()

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(w.lower()) for w in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag, dtype='float32'))


def predict_class(sentence, interpreter):
    p = bow(sentence, words, False)
    p = p.reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], p)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append(
            {"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    ints = predict_class(userText, interpreter)
    res = get_response(ints, intents)
    return res

@app.route('/about/')
def about():
    return render_template('about.html')

#userText = 'Test'
# ints = predict_class(userText, interpreter)
# res = get_response(ints, intents)
# print(res)
