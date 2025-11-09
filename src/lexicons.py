"""Small sample lexicon module.
Contains a tiny NRC-like lexicon mapping words to emotions.
This is only a demo subset; replace or extend with full NRC Emotion Lexicon for production.
"""

def load_sample_lexicon():
    # words should be lowercase
    return {
        'happy': ['joy'],
        'joy': ['joy'],
        'love': ['joy'],
        'excited': ['joy'],
        'good': ['joy'],
        'great': ['joy'],
        'sad': ['sadness'],
        'disappointed': ['sadness'],
        'bad': ['sadness'],
        'angry': ['anger'],
        'hate': ['anger'],
        'frustrated': ['anger'],
        'fear': ['fear'],
        'scared': ['fear'],
        'surprised': ['surprise'],
        'surprise': ['surprise'],
        'bored': ['sadness'],
        'amazed': ['surprise']
    }
