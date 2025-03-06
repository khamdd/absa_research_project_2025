import re
import emoji
import pandas as pd
from unidecode import unidecode
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches
from nltk.corpus import words
import nltk
import os

nltk.data.path.append("nltk_data")

class TextCleaner:
    @staticmethod
    def remove_html_tags(text):
        return re.sub(r'<.*?>', '', text)
    
    @staticmethod
    def remove_emoji(text):
        return emoji.replace_emoji(text, replace='')
    
    @staticmethod
    def remove_urls(text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    @staticmethod
    def remove_special_chars_and_numbers(text):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    @staticmethod
    def remove_hashtags(text):
        return re.sub(r'#\w+', '', text)
    
    @staticmethod
    def remove_mentions(text):
        return re.sub(r'@\w+', '', text)
    
    @staticmethod
    def process(text):
        text = TextCleaner.remove_html_tags(text)
        text = TextCleaner.remove_emoji(text)
        text = TextCleaner.remove_urls(text)
        text = TextCleaner.remove_special_chars_and_numbers(text)
        text = TextCleaner.remove_hashtags(text)
        text = TextCleaner.remove_mentions(text)
        return text
        
class Normalizer:
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "I'm": "I am",
        "I've": "I have",
        "it's": "it is",
        "he's": "he is",
        "she's": "she is",
        "they're": "they are",
        "we're": "we are",
        "you've": "you have",
        "you're": "you are",
        "wasn't": "was not",
        "weren't": "were not",
        "isn't": "is not",
        "aren't": "are not"
    }

    abbreviation_dict = {
        "pls": "please",
        "plz": "please",
        "u": "you",
        "ur": "your",
        "b4": "before",
        "gr8": "great",
        "cya": "see you",
        "bff": "best friends forever",
        "omg": "oh my god",
        "lol": "laugh out loud",
        "brb": "be right back",
        "gtg": "got to go",
        "np": "no problem",
        "idk": "I don't know",
        "smh": "shaking my head",
        "thx": "thanks",
        "ttyl": "talk to you later",
        "omw": "on my way",
        "fyi": "for your information",
        "yolo": "you only live once",
        "ez": "easy",
        "ty": "thank you"
    }

    english_words = set(words.words())

    @staticmethod
    def normalize_unicode(text):
        return unidecode(text)

    @staticmethod
    def expand_contractions(text):
        for contraction, expansion in Normalizer.contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    @staticmethod
    def expand_abbreviations(text):
        for abbr, full in Normalizer.abbreviation_dict.items():
            text = text.replace(abbr, full)
        return text
    
    @staticmethod
    def correct_wrong_words(text):
        def is_english_word(word):
            return word.lower() in Normalizer.english_words

        def correct_word(word):
            if is_english_word(word):
                return word
            suggestions = get_close_matches(word, Normalizer.english_words, n=1, cutoff=0.8)
            return suggestions[0] if suggestions else word

        corrected_words = [correct_word(word) for word in text.split()]
        return " ".join(corrected_words)

    # @staticmethod
    # def lemmatize_text(text):
    #     lemmatizer = WordNetLemmatizer()
    #     words = word_tokenize(text)
    #     return ' '.join([lemmatizer.lemmatize(word) for word in words])

    @staticmethod
    def process(text):
        text = Normalizer.normalize_unicode(text)
        text = Normalizer.expand_contractions(text)
        text = Normalizer.expand_abbreviations(text)
        # text = Normalizer.lemmatize_text(text)
        text = Normalizer.correct_wrong_words(text)
        return text

class TextPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}. Shape: {df.shape}")

        df['normalized_text'] = df['Sentence'].apply(lambda x: Normalizer.process(TextCleaner.process(x)))
        df = df[['normalized_text']]
        return df
    
    
if __name__ == '__main__':
    file_path = './data/raw/Restaurants_Test_Data_PhaseA.csv'
    output_file_path = './data/processed/Processed_Restaurants_Test_Data_PhaseA.csv'

    text_preprocessor = TextPreprocessor(file_path)
    df = text_preprocessor.load_and_preprocess_data()

    print("\nPreprocessed Data:")
    print(df.head())

    df.to_csv(output_file_path, index=False)
    print("\nCleaned data saved as 'cleaned_restaurant_data.csv'.")