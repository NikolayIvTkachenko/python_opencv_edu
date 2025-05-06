
# # Пример отображения видеопотока с веб-камеры
# import cv2
#
# # Создание объекта для захвата видео
# cap = cv2.VideoCapture(0)  # 0 - индекс веб-камеры
#
# while cap.isOpened():
#     # Считывание кадров
#     read_ok, frame = cap.read()
#
#     if not read_ok:
#         break
#
#     # Отображение обработанного кадра
#     cv2.imshow('Video Stream', frame)
#
#     # Завершение работы по нажатию клавиши 'Esc'
#     if cv2.waitKey(20) == 27:
#         break
#
# # Освобождение ресурсов
# cap.release()
# cv2.destroyAllWindows()

# Пример отображения видеофайла
# import cv2
#
# # Создание объекта для захвата видео
# cap = cv2.VideoCapture('example_video.mp4')  # Путь к видеофайлу
#
# while cap.isOpened():
#     # Считывание кадров
#     read_ok, frame = cap.read()
#
#     if not read_ok:
#         break
#
#     # Отображение обработанного кадра
#     cv2.imshow('Video File', frame)
#
#     # Завершение работы по нажатию клавиши 'Esc'
#     if cv2.waitKey(20) == 27:
#         break
#
# # Освобождение ресурсов
# cap.release()
# cv2.destroyAllWindows()



# Проверка задания в системе Stepik
# import cv2
# import numpy as np
# import base64
# import os
#
#
# def cv2_VideoCapture(b64_string):
#     decoded_video_bytes = base64.b64decode(b64_string)
#     temp_filename = f'{b64_string[:10]}.mp4'
#     with open(temp_filename, 'wb') as video_file:
#         video_file.write(decoded_video_bytes)
#     return cv2.VideoCapture(temp_filename)
#
#
# cap = cv2_VideoCapture(input())
#
# while cap.isOpened():
#     read_ok, frame = cap.read()
#     if not read_ok:
#         break
#
#     # ваш код для обработки видео
#
# cap.release()


# import cv2
# import numpy as np
#
# import imageio
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from skimage.transform import resize
# from IPython.display import HTML
#
# def display_video(path):
#     video = imageio.mimread(path)  # Загрузка видео
#     fig = plt.figure(figsize=(3, 3))  # Размер дисплея
#
#     mov = []
#     for i in range(len(video)):  # Добавление кадров видео по одному
#         img = plt.imshow(video[i], animated=True)
#         plt.axis('off')
#         mov.append([img])
#
#     # Создание анимации
#     anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)
#
#     plt.close()
#     return HTML(anime.to_html5_video())  # Отображение видео в HTML5
#
#
# filename = 'example_video.mp4'
#
#
# cap = cv2.VideoCapture(filename)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(frame_width)
# print(frame_height)
# print(total_frames)

# while cap.isOpened():
#     read_ok, frame = cap.read()
#
#     if not read_ok:
#         break
#     print(read_ok)
#
#     # здесь код для обработки видео
#
# cap.release()
# display_video(filename)


#============
import cv2
import numpy as np
import base64
import os

def cv2_VideoCapture(b64_string):
    decoded_video_bytes = base64.b64decode(b64_string)
    temp_filename = f'{b64_string[:10]}.mp4'
    with open(temp_filename, 'wb') as video_file:
        video_file.write(decoded_video_bytes)
    return cv2.VideoCapture(temp_filename)

cap = cv2_VideoCapture(input())

while cap.isOpened():
    read_ok, frame = cap.read()
    if not read_ok:
        break

# ваш код для обработки видео
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Разрешение видео: {frame_width}*{frame_height}")
print(f"Количество кадров: {total_frames}")
cap.release()


#----------------
# colors = {
#     (255, 255, 255): "белый",             # White
#     (0, 0, 255): "синий",                 # Blue
#     (0, 255, 0): "зеленый",               # Green
#     (255, 0, 0): "красный",               # Red
#     (0, 255, 255): "желтый",              # Yellow
#     (255, 0, 255): "пурпурный",           # Magenta
#     (255, 255, 0): "голубой",             # Cyan
#     (0, 0, 0): "черный",                  # Black
#     (128, 128, 128): "серый",             # Gray
#     (0, 0, 128): "темно-красный",         # Dark Red
#     (0, 128, 0): "темно-зеленый",         # Dark Green
#     (128, 0, 0): "темно-синий",           # Dark Blue
#     (0, 128, 128): "оливковый",           # Olive
#     (128, 0, 128): "темно-пурпурный",     # Dark Magenta
#     (128, 128, 0): "темно-голубой",       # Dark Cyan
#     (255, 128, 128): "светло-синий",      # Light Blue
#     (128, 128, 255): "розовый",           # Pink
#     (128, 255, 128): "салатовый",         # Light Green
#     (0, 128, 255): "оранжевый",           # Orange
#     (128, 255, 0): "бирюзовый",           # Turquoise
#     (128, 0, 255): "фиолетовый",          # Violet
#     (255, 128, 0): "аквамарин",           # Aquamarine
#     (255, 255, 128): "бледно-голубой",    # Light Cyan
#     (255, 0, 128): "индиго",              # Indigo
# }


# Методы нахождения белого пикселя на изображении
# 1. Перебор пикселей с помощью циклов
# for y in range(image.shape[0]):  # Перебор строк изображения
#     for x in range(image.shape[1]):  # Перебор столбцов изображения
#         if (image[y, x] == [255, 255, 255]).all():  # Проверка, является ли пиксель белым
#             print (y, x)  # Выводим координаты белого пикселя
# 2. Использование функции np.where
# boolean_matrix = np.all(image == [255, 255, 255], axis=-1)
# # white_pixel = np.where(boolean_matrix)
# # if white_pixel[0].size > 0:
# #     print (white_pixel[0][0], white_pixel[1][0])
# # # Вывод: 3 7

# 3. Использование функции np.argwhere
# Функция np.argwhere сразу возвращает индексы элементов массива, которые
# удовлетворяют условию. Она также быстрее, чем использование циклов.
# boolean_matrix = np.all(image == [255, 255, 255], axis=-1)
# white_pixel = np.argwhere(boolean_matrix)
# if white_pixel.size > 0:
#     print(tuple(white_pixel[0]))
# # Вывод: (3, 7)

# 4. Использование функции cv2.findNonZero
# Функция cv2.findNonZero из библиотеки OpenCV находит все ненулевые пиксели
# в бинаризованном изображении. Этот метод требует предварительного преобразования
# изображения в бинарное, т.е. одноканальное изображение,
# где каждый пиксель представлен одним числом - значением 0 или 255.
# image_bin = np.where(np.sum(image, axis=-1) > 30, 255, 0).astype(np.uint8)
# print(image_bin)
# non_zero_pixels = cv2.findNonZero(image_bin)
# if non_zero_pixels is not None:
#     print(tuple(non_zero_pixels[0][0]))
# # Вывод: (7, 3)



# Вам предстоит обработать видео, состоящее из 10 кадров размером 10х10 пикселей, попытаться найти в каждом кадре белый пиксель и вывести ответ в формате:
#
# В кадре №1 белый пиксель расположен в координатах: X, Y
# В кадре №2 белый пиксель расположен в координатах: X, Y
# В кадре №3 белый пиксель не обнаружен
# ...
# В кадре №10 белый пиксель расположен в координатах: X, Y
# Файлы для самостоятельно обработки: Скачать архив
# Google Colab: Ссылка
#
# Примечание: вы, наверное, уже заметили, что видео в формате MP4 сжимаются также, как и JPG- изображения. Это стоит учитывать при обработке!
#
# Начните решение, используя шаблон:
# import cv2
# import numpy as np
# import base64
# import os
#
#
# def cv2_VideoCapture(b64_string):
#     decoded_video_bytes = base64.b64decode(b64_string)
#     temp_filename = f'{b64_string[:10]}.mp4'
#     with open(temp_filename, 'wb') as video_file:
#         video_file.write(decoded_video_bytes)
#     return cv2.VideoCapture(temp_filename)
#
#
# cap = cv2_VideoCapture(input())
#
# while cap.isOpened():
#     read_ok, frame = cap.read()
#     if not read_ok:
#         break
#
#     # ваш код для обработки видео
#
# cap.release()