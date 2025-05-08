import numpy as np
import cv2


# # Параметры видео
# width, height = 600, 600
# num_frames = 300
#
#
# filename = 'moving_circle.mp4'
#
# # Создаем объект записи видео
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 20
# out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
#
#
# for i in range(num_frames):
#     frame = np.zeros((height, width, 3), dtype=np.uint8)
#     x_position = int(600 * (i / num_frames))
#     y_position = 300
#     cv2.circle(frame, (x_position, y_position), 30, (0, 255, 0), -1)
#     out.write(frame)
# out.release()
#
#
# cap = cv2.VideoCapture(filename)
#
# while cap.isOpened():
#     read_ok, frame = cap.read()
#     if not read_ok:
#         break
#     cv2.imshow('Moving circle', frame)
#     if cv2.waitKey(50) == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# Цикл for создает изображения с кругом, каждый раз в новом месте с небольшим смещением. С ним,
# все достаточно просто, а вот с функциями для записи видео разберемся подробнее:
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 20
# out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v'):
#
# fourcc (Four Character Code) — это кодек, который определяет формат видео.
# В данном случае используется кодек mp4v, что соответствует формату MP4.
# out = cv2.VideoWriter('wave_effect.mp4', fourcc, fps, (width, height)):
#
# Создается объект VideoWriter для записи видеофайла. Аргументы:
# 'wave_effect.mp4': Имя выходного файла.
# fourcc: Кодек для записи.
# fps: Частота кадров (20 кадров в секунду).
# (width, height): Размер кадра.
# Запись каждого кадра в видеофайл:
# out.write(frame): Записывает текущий кадр frame в видеофайл.
# Освобождение ресурсов после завершения записи:
# out.release():Освобождает объект VideoWriter и завершает запись видео.

# import cv2
# from PIL import Image
#
# filename = 'moving_circle.mp4'
# # Чтение видео и создание GIF
# cap = cv2.VideoCapture(filename)
# frames = []
#
# while cap.isOpened():
#     read_ok, frame = cap.read()
#     if not read_ok:
#         break
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame_rgb)
#     frames.append(img)
#
# cap.release()

# Сохранение в GIF
# frames[0].save('moving_circle.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
#
# OpenCV не предоставляет прямого метода для создания GIF-файлов. Однако, вы можете использовать
# OpenCV совместно с библиотекой Pillow (PIL) для конвертации кадров в GIF.
# Вот как это можно сделать, как в вашем примере:
#
# Чтение видео с помощью OpenCV.
# cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) Преобразование каждого кадра из BGR в RGB.
# Image.fromarray(frame_rgb) Создание объекта Image из массива.
# frames.append(img) Добавление всех кадров в список.
# frames[0].save('wave_effect.gif', save_all=True, append_images=frames[1:], duration=50, loop=0):
# Сохраняет список изображений в GIF с заданной длительностью кадров и зацикливанием.


# Сохранение кадров видео в отдельные файлы
import cv2
import os
import time

# Создаем директорию для сохранения изображений, если ее нет
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Открываем видеофайл
video_path = 'example_video_001.mp4'
# video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    read_ok, frame = cap.read()
    if not read_ok:
        break

    # Формируем имя файла
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:06d}.png')

    # Сохраняем кадр в файл
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
print(f"Сохранено {frame_count} кадров в директорию {output_dir}")

# Сохранение кадров видео в отдельные файлы
# Создание директории: Проверяем наличие директории images, если ее нет, создаем.
# Открытие видеофайла: Используем cv2.VideoCapture для открытия видео.
# Чтение кадров: В цикле читаем кадры с помощью cap.read(). Если кадры закончились, выходим из цикла.
# Сохранение кадров: Сохраняем каждый кадр с именем формата frame_000001.png, чтобы обеспечить правильную сортировку.
# Закрытие видео: cap.release().


# Создание видео из набора изображений
# import cv2
# import os
#
# # Параметры видео
# frame_rate = 20
# output_video_path = 'output_video.mp4'
#
# # Директория с изображениями
# input_dir = 'images'
#
# # Получаем список файлов изображений
# image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
#
# # Проверяем, есть ли изображения
# if not image_files:
#     raise ValueError("Нет изображений в директории")
#
# # Читаем первое изображение для получения размеров кадра
# first_frame = cv2.imread(os.path.join(input_dir, image_files[0]))
# height, width, layers = first_frame.shape
#
# # Создаем объект записи видео
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
#
# # Записываем каждое изображение в видеофайл
# for image_file in image_files:
#     frame = cv2.imread(os.path.join(input_dir, image_file))
#     out.write(frame)
#
# out.release()
# print(f"Видео сохранено в {output_video_path}")


# Создание видео из набора изображений
# Параметры видео: Устанавливаем частоту кадров и имя выходного видео.
# Получение списка изображений: Сортируем список файлов изображений в директории images.
# Чтение первого изображения: Используем первое изображение для получения размеров кадра.
# Создание объекта записи видео: Используем cv2.VideoWriter для создания видеофайла.
# Запись кадров в видеофайл: В цикле читаем и записываем каждое изображение в видеофайл.
# Закрытие видео: out.release().