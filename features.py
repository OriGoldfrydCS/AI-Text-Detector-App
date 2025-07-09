# ============================
# Import Setup
# ============================

import nltk
import numpy as np
from collections import Counter
from tqdm import tqdm
import spacy
from textblob import TextBlob
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================
# NLTK Resource Downloads
# ============================

nltk.download('punkt', quiet=True)          # Tokenizer models
nltk.download('punkt_tab', quiet=True)      # Extra tokenization models
nltk.download('words', quiet=True)          # Word list corpus
nltk.download('averaged_perceptron_tagger_eng', quiet=True) # POS tagger
from nltk.corpus import words
import logging

# ============================
# Logging Setup
# ============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# External Tool Initialization
# ============================

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as e:
    logger.error(f"Failed to load spaCy model 'en_core_web_sm': {e}")
    logger.error("Please run: python -m spacy download en_core_web_sm")
    raise

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# ============================
# Feature Extraction Functions
# ============================

def extract_statistical_features(texts):
    """
    Extracts statistical text features such as sentence length, word length, punctuation ratio, etc.

    Args:
        texts (List[str]): List of input texts.

    Returns:
        np.ndarray: Array of shape (N, 6) with features:
            [avg_sent_len, avg_word_len, sent_len_var,
             total_word_count, punct_to_word_ratio, uppercase_to_word_ratio]
    """
    features = []
    for text in tqdm(texts, desc="Extracting statistical features"):
        
        # Sentence and word tokenization
        sentences = nltk.sent_tokenize(text)
        words_list = nltk.word_tokenize(text)
        
        # Average sentence length (in tokens)
        avg_sent_len = np.mean([len(nltk.word_tokenize(sent)) for sent in sentences]) if sentences else 0
        
        # Average word length (in characters)
        avg_word_len = np.mean([len(word) for word in words_list]) if words_list else 0
        
        # Sentence length variance (burstiness of ideas)
        sent_len_var = np.var([len(nltk.word_tokenize(sent)) for sent in sentences]) if sentences else 0
        
        # Total number of words in the text
        total_word_count = len(words_list)
        
        # Count punctuation marks (basic punctuation)
        punctuation_count = len([c for c in text if c in '.,!?;:"\''])
        
        # Ratio of punctuation to total words
        punct_to_word_ratio = punctuation_count / total_word_count if total_word_count > 0 else 0
        
        # Number of all-uppercase words (e.g., emphasis, acronyms)
        uppercase_words = len([w for w in words_list if w.isupper() and w.isalpha()])
        uppercase_to_word_ratio = uppercase_words / total_word_count if total_word_count > 0 else 0
        
        # Add computed features for this text
        features.append([avg_sent_len, avg_word_len, sent_len_var, total_word_count, 
                        punct_to_word_ratio, uppercase_to_word_ratio])
    return np.array(features)

def extract_stylistic_features(texts):
    """
    Extracts stylistic features like TTR, repetition rate, rare words, and sentiment.

    Args:
        texts (List[str]): List of input texts.

    Returns:
        np.ndarray: Array of shape (N, 6) with features:
            [TTR, rare_word_freq, repetition_rate, discourse_freq, emotional_score, spelling_error_rate]
    """
    word_set = set(words.words()) # English vocabulary for filtering rare words
    discourse_markers = {'however', 'although', 'because', 'therefore', 'moreover', 'nevertheless', 
                        'thus', 'meanwhile', 'consequently', 'furthermore'}
    
    features = []
    for text in tqdm(texts, desc="Extracting stylistic features"):
        tokens = nltk.word_tokenize(text.lower())
        # Debug: Print tokens
        print(f"Tokens: {tokens}")
        
        # Type-Token Ratio: vocabulary diversity as percentage
        ttr = (len(set(tokens)) / len(tokens) * 100) if tokens else 0
        print(f"TTR calculation: {len(set(tokens))} unique / {len(tokens)} total = {ttr}%")
        
        # Rare word frequency
        rare_words = [w for w in tokens if w not in word_set]
        rare_word_freq = len(rare_words) / len(tokens) if tokens else 0
        
        # Repetition rate (token-level repetition)
        word_counts = Counter(tokens)
        print(f"Word counts: {word_counts}")
        repeated_words = sum(1 for count in word_counts.values() if count >= 2)
        total_repeated_tokens = sum(count - 1 for count in word_counts.values() if count >= 2)
        print(f"Repeated words count: {repeated_words}")
        print(f"Total repeated tokens: {total_repeated_tokens}")
        repetition_rate = (total_repeated_tokens / len(tokens) * 100) if tokens else 0
        print(f"Repetition Rate calculation: {total_repeated_tokens} / {len(tokens)} = {repetition_rate}%")
        
        # Discourse marker frequency
        discourse_count = sum(1 for token in tokens if token in discourse_markers)
        discourse_freq = discourse_count / len(tokens) if tokens else 0
        
        # Emotional score using VADER
        sentiment_scores = sid.polarity_scores(text)
        emotional_score = abs(sentiment_scores['pos'] - sentiment_scores['neg'])
        
        # Spelling error rate using TextBlob
        blob = TextBlob(text)
        spelling_errors = sum(1 for word in blob.words if word not in word_set and word.isalpha())
        spelling_error_rate = spelling_errors / len(tokens) if tokens else 0
        features.append([ttr, rare_word_freq, repetition_rate, discourse_freq, emotional_score, spelling_error_rate])
    return np.array(features)

def extract_syntactic_features(texts):
    """
    Extracts syntactic features such as POS tag ratios and dependency parse depth.

    Args:
        texts (List[str]): List of input texts.

    Returns:
        np.ndarray: Array of shape (N, 6) with features:
            [noun_ratio, verb_ratio, adj_ratio, avg_parse_depth, passive_ratio, pronoun_ratio]
    """
    
    # Ensure POS tagger is available
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        logger.error("NLTK averaged_perceptron_tagger_eng not found. Please run: python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")
        raise
    
    features = []
    for text in tqdm(texts, desc="Extracting syntactic features"):
        doc = nlp(text)  # spaCy parsed document
        tokens = nltk.word_tokenize(text)
        
        # POS tagging
        try:
            pos_tags = nltk.pos_tag(tokens)
        except Exception as e:
            logger.error(f"POS tagging failed for text: {e}")
            pos_tags = []
        pos_counts = Counter(tag for word, tag in pos_tags) if pos_tags else Counter()
        total_tags = len(pos_tags) if pos_tags else 1
        
        # Calculate noun, verb, and adjective tag ratios        
        noun_ratio = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0)) / total_tags
        verb_ratio = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + 
                     pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_tags
        adj_ratio = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)) / total_tags
        
        # Calculate average dependency parse depth (measure of syntactic complexity)
        parse_depths = [len(list(token.ancestors)) for token in doc]
        avg_parse_depth = np.mean(parse_depths) if parse_depths else 0  # New syntactic complexity feature
        
        # Passive voice ratio: percentage of passive subjects in sentences
        passive_count = sum(1 for sent in doc.sents for token in sent if token.dep_ == 'nsubjpass')
        passive_ratio = passive_count / len(list(doc.sents)) if doc.sents else 0
        
        # Pronoun ratio: personal/possessive references to total tokens
        pronoun_count = sum(1 for token in doc if token.pos_ == 'PRON')
        pronoun_ratio = pronoun_count / len(tokens) if tokens else 0
        features.append([noun_ratio, verb_ratio, adj_ratio, avg_parse_depth, passive_ratio, pronoun_ratio])  # Added avg_parse_depth
    return np.array(features)

def extract_meta_features(texts):
    """
    Extracts meta-linguistic features such as burstiness (sentence length variance).

    Args:
        texts (List[str]): List of input texts.

    Returns:
        np.ndarray: Array of shape (N, 1) with feature:
            [burstiness]
    """
    features = []
    for text in tqdm(texts, desc="Extracting meta features"):
        sentences = nltk.sent_tokenize(text)
        sent_lengths = [len(nltk.word_tokenize(sent)) for sent in sentences]
        
        # Burstiness: how much sentence lengths vary relative to their mean
        burstiness = np.var(sent_lengths) / np.mean(sent_lengths) if sent_lengths and np.mean(sent_lengths) > 0 else 0
        features.append([burstiness])
    return np.array(features)