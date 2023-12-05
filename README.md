# translation-and-sarcasm-detection
Here is a draft README.md for the sarcasm detection web app code:


This is a web app that can detect sarcasm in text and provide translations to French, German, and Spanish using Transformer models.

## Features

- Takes user input text and makes sarcasm predictions using a fine-tuned BERT model
- Displays a sarcasm meter showing the probability of the text being sarcastic
- Translates the input text to French, German, and Spanish using OPUS-MT models
- Built with Flask as a web framework and runs as local web server

## Usage

The main file to run is `main.py`. This will launch the Flask web server. Navigate to `http://localhost:8080` to access the web interface.

Type or paste text into the input box and click Submit. The app will analyze the text and display the sarcasm meter reading and translations.

## Models

The sarcasm classifier is a fine-tuned BERT base uncased model saved in `saved_models`. This provides context-based understanding to detect sarcasm.

The translation models are from the OPUS-MT project by Helsinki NLP. These provide high quality neural machine translation for a variety of language pairs.

## Installation

### Requirements

- Python 3.6+
- PyTorch 
- Transformers
- Flask

Install requirements with:


pip install -r requirements.txt  

## Credits

Sarcasm classifier based on research from paper: [Contextual Sarcasm Detection in Online Discussion Forums](https://ojs.aaai.org/index.php/ICWSM/article/view/3246)

Translation models from [Helsinki-NLP/Opus-MT](https://github.com/Helsinki-NLP/Opus-MT)

## License

MIT
