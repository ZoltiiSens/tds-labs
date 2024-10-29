import asyncio
import os
import requests
import pytesseract

from speech_recognition import Recognizer, AudioFile
from bs4 import BeautifulSoup
from g4f.client import Client
from g4f.Provider import RetryProvider, Liaobots, AiMathGPT, AmigoChat, Blackbox, ChatGptEs, DarkAI, Editee, Pizzagpt
import g4f.debug
from PIL import Image


# Необхідна конфігурація
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# g4f.debug.version_check = False
pytesseract.pytesseract.tesseract_cmd = r'E:\Programs\Tesseract-OCR\tesseract.exe'
speech_recognizer = Recognizer()


def main():
    user_answer = input('Оберіть джерело тексту:\n * 0 - ввести вручну\n * 1 - файл(txt)\n * 2 - фото(png)\n * 3 - аудіо-файл(wav)\n * 4 - парсинг\nEnter: ')
    if user_answer == '0':
        input_text = ask_user_for_text_from_input()
    elif user_answer == '1':
        input_text = ask_user_for_text_from_file()
    elif user_answer == '2':
        input_text = ask_user_for_text_from_photo()
    elif user_answer == '3':
        input_text = ask_user_for_text_from_audio()
    elif user_answer == '4':
        input_text = ask_user_for_text_from_web()
    else:
        print('Помилка при введенні!')
        return
    input_text = input_text.replace('\n', ' ').replace('\t', '').lower()
    print(f'Отриманий текст:\n{input_text}')
    print('Запит до LLM...')
    classification_result = ask_LLM_to_classify_text(input_text)
    print(f'Результат класифікації: {classification_result}')


def ask_user_for_text_from_input() -> str:
    return input('Введіть текст(без переходів на нову строку): ')


def ask_user_for_text_from_file() -> str:
    path = get_path_by_user('txt')
    with open(path, 'r') as f:
        text = f.read()
    return text


def ask_user_for_text_from_photo() -> str:
    path = get_path_by_user('png')
    text = pytesseract.image_to_string(Image.open(path), lang='ukr')
    return text


def ask_user_for_text_from_audio() -> str:
    path = get_path_by_user('wav')
    audio_file = AudioFile(path)
    with audio_file as source:
        audio = speech_recognizer.record(source)
    text = 'None'
    try:
        text = speech_recognizer.recognize_google(audio, language="uk-UA")
    except Exception as e:
        print("Exception: " + str(e))
    return text


def ask_user_for_text_from_web() -> str:
    url = input("Введіть URL інтернет-ресурсу: ")
    try:
        response = requests.get(url)
        response.raise_for_status()
        css_selector = input("Введіть CSS-Селектор елемента, з якого отримувати дані (до прикладу, 'section.main'): ")

        soup = BeautifulSoup(response.text, 'html.parser')
        element = soup.select_one(css_selector)
        if element:
            return element.get_text(strip=True)
        else:
            print("Заданий селектор не знайдено!")
            exit(-1)
    except requests.exceptions.RequestException as e:
        print('Помилка під час парсингу!')
        exit(-1)


def get_path_by_user(extension):
    user_answer = input(f'Введіть шлях до файлу {extension}(відносний або абсолютний): ')
    if user_answer.split('.')[-1] != extension:
        print('Некоректний тип файлу!')
        exit(-1)
    path = os.path.normpath(user_answer)
    if not os.path.exists(path):
        print('No file by given path!')
        exit(-1)
    return path


def ask_LLM_to_classify_text(text):
    client = Client(
        provider=RetryProvider([Liaobots, AmigoChat, Blackbox, ChatGptEs, DarkAI, Editee, AiMathGPT, Pizzagpt],
                               shuffle=False)
    )

    question = '''Below I'll send text. Classify it. Don't use commas and dots.
    Your answer should be in Ukrainian and have structure(Don't write any other words except of this):
    {Type of literature} {Genre(if Художній текст) or branch(if other)}
    
    Types of literature:
    - Науковий текст
    - Довідковий текст
    - Технічний текст
    - Художній текст    
    
    Example:
    - технічний текст в галузі машинного навчання
    - художній текст у детективному жанрі
    - науковий текст в галузі природніх копалин
    - художній текст у науково-фантастичному жанрі
    
    Text:
    
    '''
    question += text

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )
    print()
    return response.choices[0].message.content


# Вхід у програму
if __name__ == '__main__':
    main()
