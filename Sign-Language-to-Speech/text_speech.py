import pyttsx3
from googletrans import Translator
from gtts import gTTS
import os
import pygame
import time
import tempfile


def text_to_speech(text, gender='Male', language='en'):
    if language != 'en':
        # no male voices for other languages except english
        translator = Translator()
        trans = translator.translate(text, src='en', dest=language)
        text = trans.text


        tts = gTTS(text=text, lang=language)


        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        tts.save(temp_audio_path)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(1)


        pygame.mixer.music.stop()
        pygame.mixer.quit()


        os.remove(temp_audio_path)
    else:

        voice_dict = {'Male': 0, 'Female': 1}
        code = voice_dict[gender]

        engine = pyttsx3.init()
        engine.setProperty('rate', 125)


        engine.setProperty('volume', 0.8)

        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[code].id)

        engine.say(text)
        engine.runAndWait()



