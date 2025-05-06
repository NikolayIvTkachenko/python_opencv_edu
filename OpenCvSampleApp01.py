import numpy as np
import cv2

# Определяем размеры будущего изображения
height, width = 300, 300

# # Создаем пустое изображение (черное). Массив нужного размера, заполненный нулями
# image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Заполнение всего изображения синим цветом
# image[:] = [255, 0, 0]  # BGR формат
#
# # Создаем новое пустое изображение
# image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Заполнение верхней половины изображения зеленым цветом
# image[:height // 2, :] = [0, 255, 0]  # Зеленый цвет
#
# # Создаем пустое изображение
# image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Создаем горизонтальный градиент от черного к белому
# for x in range(width):
#     color = int((x / width) * 255)
#     image[:, x] = [color, color, color]
#
# # Создаем пустое изображение
# image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Создаем вертикальный градиент от черного к белому
# for y in range(height):
#     color = int((y / height) * 255)
#     image[y, :] = [color, color, color]
#
# # Создаем пустое изображение
# image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Создаем диагональный градиент от черного к белому
# for y in range(height):
#     for x in range(width):
#         color = int(((x + y) / (width + height)) * 255)
#         image[y, x] = [color, color, color]


# Словарь цветов
# colors = [
#     [255, 0, 0],   # Синий
#     [0, 255, 0],   # Зеленый
#     [0, 0, 255],   # Красный
#     [0, 255, 255], # Желтый
#     [255, 255, 0], # Голубой
#     [255, 0, 255] # Пурпурный
# ]
#
# image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Количество сегментов для каждого цвета
# num_colors = len(colors)
# segment_height = height // num_colors
#
# for i in range(num_colors):
#     start_color = np.array(colors[i])
#     end_color = np.array(colors[(i + 1) % num_colors])
#     for y in range(segment_height):
#         color = (start_color * (segment_height - y) + end_color * y) / segment_height
#         image[i * segment_height + y, :] = color
# cv2.imshow('Original image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Определение начального и конечного цветов для градиента:
#
# start_color = np.array(colors[i])
# ​​​​​​​end_color = np.array(colors[(i + 1) % num_colors])
# start_color: текущий цвет из массива colors на шагеi.
# end_color: следующий цвет из массива colors. Индекс (i + 1) % num_colors использует операцию взятия
# остатка от деления на количество элементов, чтобы вернуться к первому цвету, когда доходит до конца
# массива (это создает замкнутый градиент).
# Внутренний цикл for y in range(segment_height)::
#
# Проходит по каждой строке текущего сегмента изображения.
# Внутри этого цикла вычисляется промежуточный цвет для каждой строки. Это создаёт плавный переход
# от start_color к end_color.
# Вычисление промежуточного цвета для текущей строки:
#
# color = (start_color * (segment_height - y) + end_color * y) / segment_height
#
# Для каждой строки y вычисляется значение цвета, которое является взвешенной
# суммой двух цветов — начального и конечного:
# Чем меньше y, тем ближе результат к start_color.
# Чем больше y, тем ближе результат к end_color.
# Это создаёт плавный линейный градиент по высоте сегмента.


# Сохранение созданного изображения
# Когда изображение готово, его можно сохранить на диск для дальнейшего использования.
# В OpenCV для этого используется функция cv2.imwrite. Она позволяет записывать изображение
# в файл с различными форматами, такими как PNG, JPEG и другие.
import numpy as np
import cv2

# Определяем размеры изображения
height, width = 300, 300

# Создаем градиентное изображение (например, вертикальный градиент)
image = np.zeros((height, width, 3), dtype=np.uint8)
for y in range(height):
    color = int((y / height) * 255)
    image[y, :] = [color, color, color]  # Черный к белому

# Сохраняем изображение
output_file = "image.png"
cv2.imwrite(output_file, image)

print(f"Изображение успешно сохранено в файл: {output_file}")

# Как это работает:
# cv2.imwrite(filename, img):
#
# filename — имя выходного файла. Оно должно содержать расширение (например, .png, .jpg).
# img — изображение, которое нужно сохранить. Оно должно быть в формате NumPy-массива.
# Поддерживаемые форматы:
#
# PNG (.png) — используется для сохранения изображений без потерь качества.
# JPEG (.jpg) — используется для сохранения сжатых изображений.
# Проверка сохранения:
#
# Если изображение успешно сохранено, cv2.imwrite возвращает True.

import numpy as np
import cv2
height=int(input())
width=int(input())
# Ваш код
image = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imwrite("output.png", image)
#print(image.shape)
print(f"Создано изображение с размерами: Высота: {image.shape[0]}px, Ширина: {image.shape[1]}px, Количество цветовых каналов = {image.shape[2]}")