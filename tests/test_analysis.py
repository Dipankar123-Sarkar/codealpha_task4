import os
from src.analysis import ensure_nltk_resources, preprocess_text, sentiment_vader, sentiment_textblob, emotion_from_lexicon
from src.lexicons import load_sample_lexicon


def test_preprocess():
    assert preprocess_text('  Hello   world\n') == 'Hello world'


def test_sentiment_vader_and_textblob():
    ensure_nltk_resources()
    texts = ['I love this', 'I hate this', 'It is fine']
    v = sentiment_vader(texts)
    tb = sentiment_textblob(texts)
    assert len(v) == 3
    assert len(tb) == 3
    # check that at least one positive and one negative label appear
    assert 'positive' in v or 'positive' in tb
    assert 'negative' in v or 'negative' in tb


def test_emotion_lexicon():
    lex = load_sample_lexicon()
    res = emotion_from_lexicon(['I am very happy and excited', 'This is bad and sad'], lex)
    assert isinstance(res, list) and len(res) == 2
    assert res[0].get('joy', 0) >= 1
    assert res[1].get('sadness', 0) >= 1
