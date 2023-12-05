import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import numpy as np
import warnings
import sentencepiece
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'koni@4045'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = '/Users/konishbharathrajjonnalagadda/Desktop/UNH/NLP/final project/saved_models'
model = BertForSequenceClassification.from_pretrained(save_path)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/process', methods=['GET', 'POST'])
def process():
    user_input = request.form.get('userInput')
    predicted_sarcasm = predict_sarcasm(user_input, model)
    predicted_sarcasm = predicted_sarcasm.numpy()
    sarcasm_meter = predicted_sarcasm[0][1]
    french_translated_sentence = french_translation(user_input)
    german_translated_sentence = german_translation(user_input)
    spanish_translated_sentence = spanish_translation(user_input)
    return render_template('sarcasm.html', predicted_sarcasm=sarcasm_meter,
                           french_translated_sentence=french_translated_sentence,german_translated_sentence=german_translated_sentence
                           ,spanish_translated_sentence=spanish_translated_sentence)


def predict_sarcasm(statement, model):
    # Tokenize the statement
    inputs = tokenizer(statement, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # predicted_label = torch.argmax(logits).item()
    predicted_label = torch.nn.Softmax()(logits)

    return predicted_label


def french_translation(sentence):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    # Tokenize the sentences
    encoded_english = tokenizer.prepare_seq2seq_batch(sentence, return_tensors="pt")

    # Perform translation
    with torch.no_grad():
        translations = model.generate(**encoded_english)

    # Decode and display translations
    translated_sentence = ''
    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    for french_translation in decoded_translations:
        translated_sentence = translated_sentence + french_translation
        return translated_sentence


def german_translation(sentence):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    # Tokenize the sentences
    encoded_english = tokenizer.prepare_seq2seq_batch(sentence, return_tensors="pt")

    # Perform translation
    with torch.no_grad():
        translations = model.generate(**encoded_english)

    translated_sentence = ''
    # Decode and display translations
    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    for german_translation in decoded_translations:
        translated_sentence=translated_sentence + german_translation
    return translated_sentence


def spanish_translation(sentence):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    # Tokenize the sentences
    encoded_french = tokenizer.prepare_seq2seq_batch(sentence, return_tensors="pt")

    # Perform translation
    with torch.no_grad():
        translations = model.generate(**encoded_french)

    # Decode and display translations
    translated_sentence=''
    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    for spanish_translation in decoded_translations:
        translated_sentence=translated_sentence+spanish_translation

    return translated_sentence


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = '/Users/konishbharathrajjonnalagadda/Desktop/UNH/NLP/final project/saved_models'
    model = BertForSequenceClassification.from_pretrained(save_path)

    statement = "Breaking News: World Shocked to Discover That Mondays Are, in Fact, Everyone's Favorite Day!"
    print(predict_sarcasm(statement, model))

    app.run(host='0.0.0.0', port=8080, debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
