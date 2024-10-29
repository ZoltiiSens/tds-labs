# Інструкція з інсталяції

1. Завантажити та встановити tesseract на пк
2. Додати шлях до tesseract у змінну Path (Якщо ОС Windows)
3. Створити віртуальне середовище: ```python -m venv venv```
4. Активувати віртуальне середовище: ```venv\Scripts\activate``` (Якщо ОС Windows)
5. Завантажити залежності з requirements.txt: ```pip install -r requirements.txt```
6. Замінити значення з рядку ```pytesseract.pytesseract.tesseract_cmd = r'E:\Programs\Tesseract-OCR\tesseract.exe'``` на шлях до tesseract'у на ПК
7. Запустити скрипт на виконання