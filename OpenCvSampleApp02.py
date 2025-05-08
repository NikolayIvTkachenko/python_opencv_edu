

import numpy as np
import cv2

# Определяем размеры будущего изображения
height, width = 300, 400

# Создаем пустое изображение (черное). Массив нужного размера, заполненный нулями
image = np.zeros((height, width, 3), dtype=np.uint8)


# Рисование линии
start_point = (50, 50)
end_point = (200, 200)
color = (255, 0, 0)  # Синий
thickness = 2  # Толщина линии

cv2.line(image, start_point, end_point, color, thickness)

start_point = (380, 20)
end_point = (250, 270)
color = (0, 255, 0)  # Зеленый
thickness = 20  # Толщина линии
cv2.line(image, start_point, end_point, color, thickness)


# Рисование прямоугольника
start_point = (50, 50)
end_point = (200, 200)
color = (0, 255, 255)  # Желтый
thickness = 2  # Толщина линии

cv2.rectangle(image, start_point, end_point, color, thickness)

# Рисование прямоугольника
start_point = (50, 210)
end_point = (370, 280)
color = (255, 255, 255)  # Белый
thickness = -1  # Толщина линии

cv2.rectangle(image, start_point, end_point, color, thickness)

# Рисование круга
center = (150, 150)
radius = 50
color = (255, 0, 0)  # Синий
thickness = 2  # Толщина линии

cv2.circle(image, center, radius, color, thickness)

# Рисование полигона
# Координаты вершин звезды
# Создаем пустое изображение
image = np.zeros((400, 400, 3), dtype=np.uint8)

# Координаты вершин звезды
points = np.array([
    [200, 50],   # Верхняя вершина
    [230, 150],  # Правая верхняя
    [330, 150],  # Правая верхняя вторая
    [250, 210],  # Правая средняя
    [270, 310],  # Правая нижняя
    [200, 250],  # Нижняя вершина
    [130, 310],  # Левая нижняя
    [150, 210],  # Левая средняя
    [70, 150],   # Левая верхняя вторая
    [170, 150]   # Левая верхняя
], np.int32)

# Преобразуем массив координат
points = points.reshape((-1, 1, 2))
cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=2)
cv2.fillPoly(image, [points], color=(0, 255, 255))


image = np.zeros((400, 400, 3), dtype=np.uint8)
# Рисование эллипса
center = (150, 150)
axes = (100, 50)  # Длины осей
angle = 30  # Угол поворота
start_angle = 0
end_angle = 360
color = (0, 0, 255)  # Красный
thickness = 2

cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness)
text = "Hello"
org = (50, 50)
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1
color = (255, 255, 255)  # Белый
thickness = 2
cv2.putText(image, text, org, font, font_scale, color, thickness=None, lineType=None)

cv2.imshow('Original image', image)
# # Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()


# Параметры
# img (обязательный):
#
# Тип: numpy.ndarray
# Описание: Изображение, на которое будет наложен текст. Должно быть изменяемым (например, создано с помощью np.zeros или загружено через cv2.imread).
# text (обязательный):
#
# Тип: str
# Описание: Строка текста, которую необходимо отобразить на изображении.
# org (обязательный):
#
# Тип: tuple или list из двух целых чисел (x, y)
# Описание: Координаты нижнего левого угла текста в пикселях.
# fontFace (обязательный):
#
# Тип: int
# Описание: Тип шрифта. OpenCV предоставляет несколько встроенных шрифтов, например:
# cv2.FONT_HERSHEY_SIMPLEX
# cv2.FONT_HERSHEY_COMPLEX — поддерживает кириллицу
# cv2.FONT_HERSHEY_DUPLEX
# fontScale (обязательный):
#
# Тип: float
# Описание: Масштаб шрифта. Определяет размер текста. Значение 1 соответствует базовому размеру шрифта. Большие значения увеличивают текст, меньшие уменьшают.
# color (обязательный):
#
# Тип: tuple из трёх целых чисел (B, G, R)
# Описание: Цвет текста в формате BGR (Blue, Green, Red). Например, (255, 0, 0) — синий, (0, 255, 0) — зелёный, (0, 0, 255) — красный.
# thickness (необязательный):
#
# Тип: int
# Описание: Толщина линий текста в пикселях. По умолчанию 1.
# Важно: Увеличение толщины делает текст более жирным и хорошо заметным. Например, 2 или 3 пикселя часто используются для улучшенной читаемости.
# lineType (необязательный):
#
# Тип: int
# Описание: Тип линии, используемый для рисования текста. Основные значения:
# cv2.LINE_8 (по умолчанию) — 8-связная линия.
# cv2.LINE_AA — антиалиасинг (сглаживание краёв), что делает текст более гладким и приятным для глаз.

# # Вывод текста
# text = "Привет, друг!"
# org = (50, 50)
# font = cv2.FONT_HERSHEY_COMPLEX
# font_scale = 1
# color = (255, 255, 255)  # Белый
# thickness = 2
#
# cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)



import cv2
import numpy as np

# Создаем пустое изображение
image = np.zeros((600, 600, 3), dtype=np.uint8)

# Функция для рисования тени
def draw_shadow(img, points, offset=(5, 5), color=(50, 50, 50)):
    shadow_points = [(x + offset[0], y + offset[1]) for x, y in points]
    shadow_points = np.array(shadow_points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [shadow_points], color)

# Рисуем треугольники
triangles = [
    [(100, 100), (200, 50), (250, 150)],
    [(300, 200), (400, 150), (450, 250)],
    [(150, 300), (250, 250), (200, 350)]
]

for points in triangles:
    draw_shadow(image, points)
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [points], (0, 0, 255))  # Красный треугольник

# Рисуем круги с градиентом
for i in range(3):
    center = (100 + i * 150, 400)
    for r in range(50, 0, -10):
        color = (0, 255 - r * 5, 0)
        cv2.circle(image, center, r, color, -1)

# Рисуем эллипсы с градиентом
for i in range(3):
    center = (400 - i * 150, 200)
    for r in range(50, 0, -10):
        color = (255 - r * 5, 0, 0)
        axes = (r, int(r / 2))
        cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)

# Рисуем линии
cv2.line(image, (50, 50), (550, 50), (255, 255, 0), 3)
cv2.line(image, (550, 50), (550, 550), (255, 0, 255), 3)
cv2.line(image, (550, 550), (50, 550), (0, 255, 255), 3)
cv2.line(image, (50, 550), (50, 50), (255, 0, 0), 3)
cv2.line(image, (50, 50), (550, 550), (0, 255, 0), 2)
cv2.line(image, (50, 550), (550, 50), (0, 0, 255), 2)

# Рисуем дополнительные линии
for i in range(10):
    start_point = (np.random.randint(0, 600), np.random.randint(0, 600))
    end_point = (np.random.randint(0, 600), np.random.randint(0, 600))
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    cv2.line(image, start_point, end_point, color, np.random.randint(1, 4))

# Рисуем пиксельные элементы (глитчи)
for i in range(20):
    top_left = (np.random.randint(0, 600), np.random.randint(0, 600))
    bottom_right = (top_left[0] + np.random.randint(5, 20), top_left[1] + np.random.randint(5, 20))
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    cv2.rectangle(image, top_left, bottom_right, color, -1)

# Добавляем текст
cv2.putText(image, "Иллюзия реальностей", (150, 580), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# Сохраняем изображение в файл
cv2.imwrite("art.png", image)