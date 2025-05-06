

# import cv2
# import numpy as np
# image = cv2.imread('pixel3x3.png')
# print(image)
#
# height, width, channels = image.shape
# print("Высота:", height)      # Вывод: 3
# print("Ширина:", width)       # Вывод: 3
# print("Каналы:", channels)    # Вывод: 3
#
# # [0, 0, 0] — черный цвет.
# # [128, 128, 128] — серый цвет (среднее значение между черным и белым).
# # [255, 255, 255] — белый цвет.
# # [0, 0, 255] — красный цвет.
# # [0, 255, 0] — зеленый цвет.
# # [255, 0, 0] — синий цвет.
# # [255, 255, 0] — бирюзовый цвет.
# # [0, 255, 255] — желтый цвет.
# # [255, 0, 255] — фиолетовый цвет.
#
# print(image.shape)  # Вывод: (3, 3, 3)

#
# import cv2
# import numpy as np
#
# # Чтение изображения
# image = cv2.imread('pixel3x3.png')
#
# # Получение цвета пикселя в координате (1, 2)
# pixel_color = image[1, 2]
# print(pixel_color)  # Выводит: [255, 0, 0] (синий пиксель)


# import cv2
# import numpy as np
# import base64
#
# colors = {(255,255,255):"white",
#           (0,0,255):"red",
#           (0,255,0):"green",
#           (255,0,0):"blue",
#           (0,255,255):"yellow",
#           (255,0,255):"purple",
#           (255,255,0):"aqua",
#           (0,0,0):"black",
#           (128,128,128):"gray",}
#
# def read_image(input_text):
#     img = cv2.imdecode(np.frombuffer(base64.b64decode(input_text), dtype=np.uint8), cv2.IMREAD_COLOR)
#     return img
#
# image = read_image(input())

# import cv2
# import numpy as np
#
# # Чтение изображения
# image = cv2.imread('image.jpg')
#
# # Округление значений цветов
# image = np.where(image > 200, 255, 0)
#
# # Сохранение результата
# cv2.imwrite('result_image.png', image)
#
# import cv2
# import numpy as np
# import base64
#
# colors = {(255,255,255):"white",
#           (0,0,255):"red",
#           (0,255,0):"green",
#           (255,0,0):"blue",
#           (0,255,255):"yellow",
#           (255,0,255):"purple",
#           (255,255,0):"aqua",
#           (0,0,0):"black"}
#
# def read_image(input_text):
#     img = cv2.imdecode(np.frombuffer(base64.b64decode(input_text), dtype=np.uint8), cv2.IMREAD_COLOR)
#     return img
#
# image = read_image(input())
# pos=np.array(list(map(int, input().split())))
# print(pos)
# print(image[pos[0]][pos[1]])
# color=image[pos[0]][pos[1]]
# colorimage = np.where(color > 200, 255, 0)
# print(colorimage)
# key=(colorimage[0], colorimage[1], colorimage[2])
# print(colors[key])

# import cv2
# import numpy as np
#
# img = cv2.imread('RGB-image.png')
# cv2.imshow('Image',img) #'Image' - заголовок окна, img - переменная, содержащая изображение
#
# cv2.waitKey(0) #ожидает нажатия любой клавиши
# cv2.destroyAllWindows() #закроет окно перед завершением скрипта
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# #%matplotlib inline
#
#
# image = cv2.imread('RGB-image.png')
# plt.imshow(image)
# plt.show()
#
# image2show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image2show)
# plt.show()
#
# blue_channel = image[:, :, 0]  # Синий канал
# green_channel = image[:, :, 1]  # Зеленый канал
# red_channel= image[:, :, 2]  # Красный канал
#
# # Отображение окон
# cv2.imshow('Original image', image)
# cv2.imshow('Red channel', red_channel)
# cv2.imshow('Green channel', green_channel)
# cv2.imshow('Blue channel', blue_channel)
# #cv2.imwrite(filename, image)
#
# # Ожидание нажатия клавиши для закрытия окон
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-------------------------------------------------------------
# import numpy as np
#
# # Пример изображения (4x4 пикселя, 3 канала - BGR)
# image = np.array([[[120, 120, 120],
#                    [121, 121, 122],
#                    [122, 122, 123],
#                    [123, 124, 124]],
#
#                   [[125, 125, 125],
#                    [126, 126, 127],
#                    [127, 128, 128],
#                    [128, 129, 129]],
#
#                   [[130, 130, 131],
#                    [131, 131, 132],
#                    [132, 133, 133],
#                    [134, 134, 134]],
#
#                   [[135, 135, 136],
#                    [136, 137, 137],
#                    [137, 138, 138],
#                    [139, 139, 140]]])
#
#
# # Среднее значение по всем пикселям и каналам
# avg_color_all = np.mean(image)
# print(image)
# print("Среднее значение всех пикселей изображения:", avg_color_all)
# # Вывод: Среднее значение всех пикселей изображения: 129.52083333333334
#
# import numpy as np
#
# image = np.array([
#     [[250,  41,  51], [248,  61,  46], [252,  56,  61], [251,  71,  56]],
#     [[249,  66,  41], [250,  76,  46], [247,  81,  51], [248,  86,  61]],
#     [[251,  91,  41], [250,  96,  56], [249, 101,  61], [248, 106,  51]],
#     [[252, 111,  61], [251, 116,  46], [250, 121,  56], [249, 126,  41]]
# ])
#
# # Среднее значение по всем пикселям и каналам
# avg_color_all = np.mean(image)
# print("Среднее значение всех пикселей изображения:", avg_color_all)
# # Вывод: Среднее значение всех пикселей изображения: 129.72916666666666

import numpy as np


image_blue = np.array([
    [[250,  41,  51], [248,  61,  46], [252,  56,  61], [251,  71,  56]],
    [[249,  66,  41], [250,  76,  46], [247,  81,  51], [248,  86,  61]],
    [[251,  91,  41], [250,  96,  56], [249, 101,  61], [248, 106,  51]],
    [[252, 111,  61], [251, 116,  46], [250, 121,  56], [249, 126,  41]]
])

image_gray = np.array([
    [[120, 120, 120], [121, 121, 122], [122, 122, 123], [123, 124, 124]],
    [[125, 125, 125], [126, 126, 127], [127, 128, 128], [128, 129, 129]],
    [[130, 130, 131], [131, 131, 132], [132, 133, 133], [134, 134, 134]],
    [[135, 135, 136], [136, 137, 137], [137, 138, 138], [139, 139, 140]]])

# Среднее значение по каждому каналу
avg_color_channels = np.mean(image_blue, axis=(0, 1), dtype=np.int32)
print("Среднее значение по каналам B, G, R:", avg_color_channels)
# Вывод: Среднее значение по каналам B, G, R: [249  87  51]
#print(np.all(np.abs(avg_color - color) < tolerance))
# Среднее значение по каждому каналу
avg_color_channels = np.mean(image_gray, axis=(0, 1), dtype=np.int32)
print("Среднее значение по каналам B, G, R:", avg_color_channels)
# Вывод: Среднее значение по каналам B, G, R: [129 129 129]

import numpy as np

# Средний цвет части изображения
avg_color = np.array([105, 145, 195])

# Эталонный цвет
color = np.array([100, 150, 200])

# Допустимая погрешность
tolerance = 10

# Вычисление абсолютной разности
abs_diff = np.abs(avg_color - color)
print("Абсолютная разность:", abs_diff)
# Вывод: Абсолютная разность: [5 5 5]

# Проверка, все ли разности меньше погрешности
comparison = abs_diff < tolerance
print("Сравнение с погрешностью:", comparison)
# Вывод: Сравнение с погрешностью: [ True  True  True]

# Проверка, все ли значения True
result = np.all(comparison)
print("Результат проверки:", result)
# Вывод: Результат проверки: True

# В этом примере все компоненты среднего цвета отличаются от соответствующих
# компонентов эталонного цвета на значения, меньшие допустимой погрешности (10).
# Поэтому функция возвращает True, указывая на то, что средний цвет совпадает с эталонным в пределах заданной погрешности.

def find_closest_color(color, color_dict):
    closest_color = None
    min_distance = float('inf')
    for key in color_dict.keys():
        distance = np.linalg.norm(np.array(color) - np.array(key))  # Вычисление евклидова расстояния
        if distance < min_distance:
            min_distance = distance
            closest_color = color_dict[key]
    return closest_color

# Эталонные цвета
colors = {
    (255, 255, 255): "белый",             # White
    (255, 0, 0): "синий",                 # Blue
    (0, 255, 0): "зеленый",               # Green
    (0, 0, 255): "красный",               # Red
    (255, 255, 0): "голубой",             # Cyan
    (255, 0, 255): "пурпурный",           # Magenta
    (0, 255, 255): "желтый",              # Yellow
    (0, 0, 0): "черный",                  # Black
    (128, 128, 128): "серый",             # Gray
    (128, 0, 0): "темно-синий",           # Dark Blue
    (0, 128, 0): "темно-зеленый",         # Dark Green
    (0, 0, 128): "темно-красный",         # Dark Red
    (128, 128, 0): "оливковый",           # Olive
    (255, 128, 128): "розовый",           # Pink
    (255, 128, 0): "оранжевый"            # Orange
}

# Цвет, для которого нужно найти ближайший эталонный цвет
color = [100, 150, 200]

closest_color = find_closest_color(color, colors)
print("Ближайший цвет:", closest_color)  # Вывод: Ближайший цвет: голубой

# Задача усложняется, цветов стало намного больше!
# Ваша задача прежняя — определить цвет данного пикселя и вывести его название.

colors = {
    (255, 255, 255): "белый",             # White
    (0, 0, 255): "красный",               # Red
    (0, 255, 0): "зеленый",               # Green
    (255, 0, 0): "синий",                 # Blue
    (0, 255, 255): "желтый",              # Yellow
    (255, 0, 255): "пурпурный",           # Magenta
    (255, 255, 0): "голубой",             # Cyan
    (0, 0, 0): "черный",                  # Black
    (128, 128, 128): "серый",             # Gray
    (0, 0, 128): "темно-красный",         # Dark Red
    (0, 128, 0): "темно-зеленый",         # Dark Green
    (128, 0, 0): "темно-синий",           # Dark Blue
    (0, 128, 128): "оливковый",           # Olive
    (128, 0, 128): "темно-пурпурный",     # Dark Magenta
    (128, 128, 0): "темно-голубой",       # Dark Cyan
    (255, 128, 255): "розово-фиолетовый", # Purple-pink
    (255, 128, 128): "светло-синий",      # Light Blue
    (128, 255, 255): "светло-желтый",     # Light yellow
    (128, 128, 255): "розовый",           # Pink
    (128, 255, 128): "салатовый",         # Lime
    (0, 128, 255): "оранжевый",           # Orange
    (0, 255, 128): "светло-зеленый",      # Light green
    (128, 255, 0): "бирюзовый",           # Turquoise
    (128, 0, 255): "фиолетовый",          # Violet
    (255, 128, 0): "аквамарин",           # Aquamarine
    (255, 255, 128): "лавандовый",        # Lavender
    (255, 0, 128): "индиго",              # Indigo
}
#--------
#
# import cv2
# import numpy as np
# import base64
#
# colors = {
#     (255, 255, 255): "белый",             # White
#     (0, 0, 255): "красный",               # Red
#     (0, 255, 0): "зеленый",               # Green
#     (255, 0, 0): "синий",                 # Blue
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
#     (255, 128, 255): "розово-фиолетовый", # Purple-pink
#     (255, 128, 128): "светло-синий",      # Light Blue
#     (128, 255, 255): "светло-желтый",     # Light yellow
#     (128, 128, 255): "розовый",           # Pink
#     (128, 255, 128): "салатовый",         # Lime
#     (0, 128, 255): "оранжевый",           # Orange
#     (0, 255, 128): "светло-зеленый",      # Light green
#     (128, 255, 0): "бирюзовый",           # Turquoise
#     (128, 0, 255): "фиолетовый",          # Violet
#     (255, 128, 0): "аквамарин",           # Aquamarine
#     (255, 255, 128): "лавандовый",        # Lavender
#     (255, 0, 128): "индиго",              # Indigo
# }
#
# def read_image(input_text):
#     img = cv2.imdecode(np.frombuffer(base64.b64decode(input_text), dtype=np.uint8), cv2.IMREAD_COLOR)
#     return

# import cv2
# import numpy as np
# import base64
#
# def read_image(input_text):
#     img = cv2.imdecode(np.frombuffer(base64.b64decode(input_text), dtype=np.uint8), cv2.IMREAD_COLOR)
#     return img
#
# image = read_image(input())