import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import simpleaudio as sa
import numpy as np

def text_to_sound(user_text):

    A4 = 440
    sample_rate = 44100
    T = 1
    t = np.linspace(0, T, T * sample_rate, False)

    nltk_analyzer = SentimentIntensityAnalyzer()
    all_sentences = nltk.sent_tokenize(user_text)

    sin_waves = []

    for each_sentence in all_sentences:
        score = round((nltk_analyzer.polarity_scores(each_sentence)['compound'] + 1) * 57)

        frequency = A4 * (2 ** ((score + 1 - 49) / 12))
        sin_waves.append(np.sin(2 * np.pi * frequency * t))

    
    # concatenate notes
    audio = np.hstack(sin_waves)
    # normalize to 16-bit range
    audio *= 32767 / np.max(np.abs(audio))
    # convert to 16-bit data
    audio = audio.astype(np.int16)

    # playing sound
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()

if __name__ == "__main__":
    
    user_text = input("Enter some text that you want to hear: ")
    print("Playing music...")
    text_to_sound(user_text)