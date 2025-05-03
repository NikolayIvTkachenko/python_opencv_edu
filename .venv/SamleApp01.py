









# import cv2
# import numpy as np
#
# # Загрузка изображения из файла с помощью OpenCV
# image = cv2.imread('path_to_image.jpg')
#
# # Проверка типа данных и формы массива
# print(type(image))  # <class 'numpy.ndarray'>
# print(image.dtype)  # uint8
# print(image.shape)  # (height/высота, width/ширина, channels/количество каналов)
#
# # Отображение изображения
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imread('path_to_image.jpg') загружает изображение и возвращает его в виде массива NumPy

# import cv2
# import numpy as np
#
# image = cv2.imread('simple3x3.png')
# print(image)
#
# [[[  0   0 255]
#   [  0 255   0]
#   [255   0   0]]
#
#  [[255 255 255]
#   [127 127 127]
#   [  0   0   0]]
#
#  [[  0 255 255]
#   [255   0 255]
#   [255 255   0]]]


# import cv2
# import numpy as np
#
#
# image = read_image(input())
# print(image)
#
# import cv2
# import numpy as np
# image = read_image(input())  # читаем изображение
# print(image.shape)
#
# import cv2
# import numpy as np
# image = read_image(input())  # читаем изображение
# print(image.shape[0])
# print(image.shape[1])
# print(image.shape[2])
# print(image.shape)
# print(f"Ширина изображения: {image.shape[0]} пикс.")
# print(f"Высота изображения: {image.shape[1]} пикс.")

# Основные типы данных в NumPy
# Целочисленные типы (Integer types)
#
# int8: 8-битное целое число со знаком, диапазон значений от -128 до 127.
# int16: 16-битное целое число со знаком, диапазон значений от -32768 до 32767.
# int32: 32-битное целое число со знаком, диапазон значений от -2147483648 до 2147483647.
# int64: 64-битное целое число со знаком, диапазон значений от -9223372036854775808 до 9223372036854775807.
# uint8: 8-битное целое число без знака, диапазон значений от 0 до 255.
# uint16: 16-битное целое число без знака, диапазон значений от 0 до 65535.
# uint32: 32-битное целое число без знака, диапазон значений от 0 до 4294967295.
# uint64: 64-битное целое число без знака, диапазон значений от 0 до 18446744073709551615.
# Вещественные типы (Floating-point types)
#
# float16: 16-битное число с плавающей запятой, также известное как число с половинной точностью.
# float32: 32-битное число с плавающей запятой, также известное как число с обычной точностью.
# float64: 64-битное число с плавающей запятой, также известное как двойная точность.
# Комплексные типы (Complex types)
#
# complex64: Комплексное число, состоящее из двух 32-битных вещественных чисел.
# complex128: Комплексное число, состоящее из двух 64-битных вещественных чисел.
# Булевы типы (Boolean type)
#
# bool: Булевый тип, который может принимать значения True или False.
# Типы данных с фиксированной длиной (Fixed-length string types)
#
# S: Строка фиксированной длины. Например, S10 обозначает строку длиной 10 символов.
# Типы данных с переменной длиной (Variable-length string types)
#
# U: Юникод-строка фиксированной длины. Например, U10 обозначает юникод-строку длиной 10 символов.
# import numpy as np
# # Создание целочисленного массива
# int_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
# print(int_array.dtype)  # выведет int32
#
# # Создание массива с плавающей запятой
# float_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
# print(float_array.dtype)  # выведет float64
#
# # Создание комплексного массива
# complex_array = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
# print(complex_array.dtype)  # выведет complex128
#
# # Создание булевого массива
# bool_array = np.array([True, False, True], dtype=np.bool_)
# print(bool_array.dtype)  # выведет bool
#
# # Создание строкового массива
# str_array = np.array(['apple', 'banana', 'cherry'], dtype='S10')
# print(str_array.dtype)  # выведет |S10
#
# # Создание юникод-строкового массива
# unicode_array = np.array(['apple', 'banana', 'cherry'], dtype='U10')
# print(unicode_array.dtype)  # выведет <U10
#
#
# # Исходный массив
# array = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
#
# # Изменение типа данных на int32
# int_array = array.astype(np.int32)
# # или
# int_array = array.astype("int32")
#
#
# print(int_array.dtype)  # выведет int32
# print(int_array)  # выведет [1 2 3 4 5]

# array = np.array([1, 2, 3, 4, 5])
# print(array.dtype) #выведет int64

# zeros = np.zeros((2, 3)) # массив из нулей
# ones = np.ones((2, 3)) # массив из единиц
# arange = np.arange(10) # массив от 0 до 9
# linspace = np.linspace(0, 1, 5) # массив из 5 чисел от 0 до 1
# print(zeros)
# print(ones)
# print(arange)
# print(linspace)
#
# array = np.array([1, 2, 3, 4, 5])
# array = array + 5
# print(array)
#
# # Создание одномерного массива
# array_1d = np.array([1, 2, 3, 4, 5])
# size_1d = array_1d.shape
# print(size_1d) # выведет (5,)
#
#
# # Создание двумерного массива
# array_2d = np.array([[1, 2, 3], [4, 5, 6]])
# size_2d = array_2d.shape
# print(size_2d) # выведет (2, 3)
# print(f"Размер двумерного массива: {size_2d[0]} строки, {size_2d[1]} столбцы")

#
# daily_discount_items = {
#     "понедельник": {"хлеб", "молоко", "сыр"},
#     "вторник": {"яблоки", "бананы", "груши"},
#     "среда": {"курица", "рис", "овощи"},
#     "четверг": {"сок", "печенье", "мороженое"},
#     "пятница": {"кофе", "чай", "шоколад"},
#     "суббота": {"рыба", "лимон", "зелень"},
#     "воскресенье": {"мясо", "картофель", "салат"}
# }
#
#
#
#
# # 192.168.1.1, 192.168.1.2, 192.168.1.1, 192.168.1.3
# str=input()
# str=str.replace(" ", "")
# ll=str.split(",")
# print(ll)
# a=set(ll)
# print(a)
# print(len(a))


# l0=input()
# l=l0.lstrip("[").rstrip("]").split(", ")
# ll=list(l)
# s1=set(ll)
# res=list(s1)
# res.sort()
# res.reverse()
# list_a = list(map(int, res))
# print(list_a)

# l0=input()
# l=l0.lstrip("[").rstrip("]").split(", ")
# ll=list(l)
# s1=set(ll)
# res=list(s1)
# res.sort()
# res.reverse()
# print(res)
#
#
# l0=input()
# l=l0.lstrip("[").rstrip("]").split(", ")
# ll=list(l)
# s1=set(ll)
# print(s1)


#
# import json
# d = eval(input())
# key1=input()
# value1=input()
# d[key1]=value1
# #dd=json.dumps(d)
# dd=str(d)
# dd=dd.replace("'","\"")
# print(dd)
#
#
#
# import json
# d = eval(input())
# key1=input()
# value1=input()
# d[key1]=value1
# print(json.dumps(d))
#
#
# d = eval(input())
# key1=input()
# value1=input()
# d[key1]=value1
# print(d)



# d = eval(input()) # после выполнения этой строки кода в переменной d будет лежать словарь, который Вам необходимо обработать
#
# #your code:
# key=input()
# print(d[key])


# # Пример создания словаря
# person = {
#     "name": "Олег",
#     "age": 30,
#     "city": "Удомля"
# }
# # Пример доступа к значению по ключу
# name = person["name"]
# print(name)  # выведет 'Олег'
#
# # Пример добавления и изменения элементов
# person["email"] = "oleg@robotx.su"
# person["age"] = 31
# #
# # # Пример удаления элементов
# # del person["city"]
# # email = person.pop("email")
#
# # Пример использования методов
# keys = person.keys()  # возвращает список ключей
# values = person.values()  # возвращает список значений
# items = person.items()  # возвращает список пар (ключ, значение)
#
#
# # Пример вложенного словаря
# employee = {
#     "name": "Степан",
#     "job": {
#         "os": "Linux",
#         "department": "Development"
#     }
# }
#
# # Доступ к элементу вложенного словаря
# job_os = employee["job"]["os"]
# print(job_os)  # выведет 'Linux'
#
# # Пример проверки наличия ключа
# if "job" in person:
#     print("Работа найдена")
#
# # Пример отсутствия ключа
# if "зарплата" not in person:
#     print("Зарплаты не найдено")
#
# empty_dict = {}
# another_empty_dict = dict()
#
# pairs = [("name", "John"), ("age", 30)]
# person = dict(pairs)
#
#
# # Пример объединения словарей
# dict1 = {"a": 1, "b": 2}
# dict2 = {"b": 3, "c": 4}
# dict1.update(dict2)  # {"a": 1, "b": 3, "c": 4}
# print("===========")
# print(dict1)
#
# dict3 = {**dict1, **dict2}  # {"a": 1, "b": 3, "c": 4}


# # ((x1, y1), (x2, y2))
# l=input()
# print(l)
# tup = tuple(l)
# print(tup)
# print[tup[0]]
# print(tup[1])

# # put your python code here
# l=input()
# ll=l.lstrip("(").rstrip(")")
# list=ll.split(", ")
# listint=map(int, list)
# t=tuple(listint)
# if t[0]==255 and t[1]==255 and t[2]==255:
#     print(f"Белый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==255 and t[1]==0 and t[2]==0:
#     print(f"Красный цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==255 and t[2]==0:
#     print(f"Зеленый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==0 and t[2]==255:
#     print(f"Синий цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==0 and t[2]==0:
#     print(f"Черный цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==128 and t[1]==128 and t[2]==128:
#     print(f"Серый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==255 and t[1]==0 and t[2]==255:
#     print(f"Фиолетовый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# else:
#     print(t)


# # put your python code here
# l=input()
# ll=l.lstrip("(").rstrip(")")
# list=ll.split(", ")
# listint=list(map(int, xs))
# t=tuple(listint)
# if t[0]==255 and t[1]==255 and t[2]==255:
#     print(f"Белый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==255 and t[1]==0 and t[2]==0:
#     print(f"Красный цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==255 and t[2]==0:
#     print(f"Зеленый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==0 and t[2]==255:
#     print(f"Синий цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==0 and t[2]==0:
#     print(f"Черный цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==128 and t[1]==128 and t[2]==128:
#     print(f"Серый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==255 and t[1]==0 and t[2]==255:
#     print(f"Фиолетовый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# else:
#     print(t)
#
# l=input()
# print(l)
# ll=l.lstrip("(").rstrip(")")
# print(ll)
# list=ll.split(", ")
# print(list)
# t=tuple(list)
# print(t[0])
# print(t[1])
# print(t[2])
# print(t)
# if t[0]==255 and t[1]==255 and t[2]==255:
#     print(f"Белый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==255 and t[1]==0 and t[2]==0:
#     print(f"Красный цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==255 and t[2]==0:
#     print(f"Зеленый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==0 and t[2]==255:
#     print(f"Синий цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==0 and t[1]==0 and t[2]==0:
#     print(f"Черный цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==128 and t[1]==128 and t[2]==128:
#     print(f"Серый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")
# elif t[0]==255 and t[1]==0 and t[2]==255:
#     print(f"Фиолетовый цвет, это когда значение красного: {t[0]}, зеленого: {t[1]} и синего: {t[2]}")




# l=input()
# list=l.split(", ")
# print(f"Точка 1: ({list[0]}, {list[1]}), Точка 2: ({list[2]}, {list[3]})")


# l=input()
# list=l.split(",")
# w=input()
# if w=="list":
#     print(list)
# elif w=="tuple":
#     print(tuple(list))
# elif w=="space":
#     z=" ".join(list)
#     print(z)

# # Пример создания кортежа
# numbers = (1, 2, 3, 4, 5)
# mixed = (1, "Hello", 3.14, True)
#
# tuple = (1, 77, (8,9), 't', 21, 22)
# tuple_number_2 = tuple[2]
# item = tuple_number_2[0]
#
# print(item) # выведет 8
#
# tuple = (1, 77, (8,9), 't', 21, 22)
# item = tuple[2][0]
# print(item) # выведет 8
#
# fruits = ("apple", "banana", "cherry")
# first_fruit = fruits[0] # 'apple'
# last_fruit = fruits[-1] # 'cherry'
#
# numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# first_three = numbers[:3] # (1, 2, 3)
# last_three = numbers[-3:] # (7, 8, 9)
# every_second = numbers[::2] # (1, 3, 5, 7, 9)
# reversed_tuple = numbers[::-1] # (9, 8, 7, 6, 5, 4, 3, 2, 1)
#
# empty_tuple = ()
# another_empty_tuple = tuple()
#
# text = "hello"
# char_tuple = tuple(text) # ('h', 'e', 'l', 'l', 'o')
#
# text = ['h', 'e', 'l', 'l', 'o']
# char_tuple = tuple(text) # ('h', 'e', 'l', 'l', 'o')
#
# tuple1 = (1, 2, 3)
# tuple2 = (4, 5, 6)
# combined_tuple = tuple1 + tuple2 # (1, 2, 3, 4, 5, 6)
#
# repeated_tuple = (1, 2, 3) * 3 # (1, 2, 3, 1, 2, 3, 1, 2, 3)

# l=input()
# w=input()
# list=l.split(" ")
# try:
#     count = 0
#     for item in list:
#         if item==w:
#             count=count+1
#     if count>0:
#         print(count)
#     else:
#         print("Не так все просто")
# except IndexError as e:
#     print(f"Ошибка: {e}")




# fruits = ["apple", "banana"]
# fruits.append("cherry")
#
# fruits = ["apple", "banana"]
# try:
#     fruits.insert(10, "cherry") # Хотя это не вызовет ошибку, элемент будет добавлен в конец
# except IndexError as e:
#     print(f"Ошибка: {e}")
#
# fruits = ["apple", "banana"]
# try:
#     fruits.remove("cherry")
# except ValueError as e:
#     print(f"Ошибка: {e}")
#
# fruits = ["apple", "banana"]
# try:
#     idx = fruits.index("cherry")
# except ValueError as e:
#     print(f"Ошибка: {e}")
#
# fruits = ["apple", "banana"]
# try:
#     fruits.pop(2)
# except IndexError as e:
#     print(f"Ошибка: {e}")

# # put your python code here
# l=input()
# i=int(input())
# list=l.split(" ")
# try:
#     print(list[i])
# except IndexError as e:
#     print(f"Ошибка: {e}")
#
#
# fruits = ["apple", "banana"]
#
# try:
#     print(fruits[10])
# except IndexError as e:
#     print(f"Ошибка: {e}")

# В ресторане обновляется меню. Вам нужно написать программу, которая будет управлять этим меню, добавляя, удаляя и изменяя блюда.
#
# Условия задачи:
#
# Первой строкой вводится начальный список блюд, разделенных пробелами.
# Второй строкой вводится команда:
# add <блюдо>: добавить блюдо в конец списка.
# insert <позиция> <блюдо>: вставить блюдо на указанную позицию (индексация с 0).
# remove <блюдо>: удалить первое вхождение блюда из списка.
# pop <позиция>: удалить блюдо по указанной позиции.
# clear: удалить все блюда из меню.
# reverse: развернуть список блюд.
# sort: отсортировать список блюд.
# Выходные данные:
#
# После выполнения команды, выведите обновленный список блюд.



#
# # Примеры добавления элементов  в конец списка
# fruits = ["apple", "banana"]
# fruits.append("cherry") # ['apple', 'banana', 'cherry']
# # Примеры вставки в указанную позицию
# fruits.insert(1, "orange") # ['apple', 'orange', 'banana', 'cherry']
#
# # Примеры удаления элементов
# fruits = ["apple", "banana", "cherry", "banana"]
# fruits.remove("banana") # ['apple', 'cherry', 'banana']
# popped_fruit = fruits.pop(1) # 'cherry', fruits: ['apple', 'banana']
# fruits.clear() # []
#
# # Примеры поиска элементов
# fruits = ["apple", "banana", "cherry", "banana"]
# index_banana = fruits.index("banana") # 1
# count_banana = fruits.count("banana") # 2
#
# # Примеры изменения порядка элементов
# numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
# numbers.reverse() # [5, 3, 5, 6, 2, 9, 5, 1, 4, 1, 3]
# numbers.sort() # [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
#
# a1=input()
# a2=input()
# z=a1.split(", ")+a2.split(", ")
# print(z)


# l = input()
# z=input()
# a1=int(input())
# a2=int(input())
# a3=int(input())
# list = l.split(" ")
# aaa=list[a1]
# bbb=list[a2]
# ccc=list[a3]
# print(f"Добавьте следующие продукты: {aaa}, {bbb}, {ccc}! Хорошенько перемешайте и {z} готов!")

# Бабушка прячет свои секретные рецепты при помощи набора цифр в списке, и вам нужно сформировать рецепт:
#
# Входные данные:
#
# Первой строкой в программу передается список ингредиентов, которые есть на кухне, набор слов разделенных пробелов.
# Второй строкой блюдо, которое мы будем готовить
# Третьей, четвертой и пятой строкой - индексы ингредиентов из первого списка, которые нужно положить в определенном порядке.
# Выходные данные:
#
# Добавьте следующие продукты: AAA, BBB, CCC! Хорошенько перемешайте и БЛЮДО готов!



# a1 = input()
# index=int(input())
# list=a1.split(" ")
# print(list[index])


# sentence = "Это пример строки"
# words = sentence.split()
# print(words) # Вывод: ['Это', 'пример', 'строки']
#
# sentence = "Это пример строки"
# words = sentence.split()
# print(words) # Вывод: ['Это', 'пример', 'строки']

# Пример создания списка
# numbers = [1, 2, 3, 4, 5]
# mixed = [1, "Hello", 3.14, True]



# Считаем пробелы
# На вход в программу подается строка, которая вероятно окружена пробелами с двух сторон (но их может и не быть)!
# Вам нужно написать программу, которая определит количество пробелов с двух сторон строки и выведет это целое число в качестве ответа!
#
# Входные данные:
# Строка, возможно, окруженная пробелами
#
# Выходные данные:
# Целое число
# a1=input()
# lena1=len(a1)
# a2=a1.strip()
# lena2=len(a2)
# print(lena1-lena2)

# # Поиск подстроки в строке
# text = "Hello, World!"
# index = text.find("World")
# print(index)  # Вывод: 7
#
# # Поиск подстроки с указанием начала и конца поиска
# text = "Hello, World!"
# index = text.find("o", 5, 10)
# print(index)  # Вывод: 8
#
# # Если подстрока не найдена
# index = text.find("Python")
# print(index)  # Вывод: -1


# # Замена подстроки во всей строке
# text = "Hello, World!"
# new_text = text.replace("World", "Python")
# print(new_text)  # Вывод: "Hello, Python!"
#
# # Замена подстроки с указанием количества замен
# text = "apple, apple, apple"
# new_text = text.replace("apple", "orange", 2)
# print(new_text)  # Вывод: "orange, orange, apple"
#
#
# # Удаление пробелов из начала строки
# text = "    Не трогай лишнее!   "
# clean_text = text.lstrip()
# print(clean_text)  # Вывод: "Не трогай лишнее!   "
#
# # Удаление пробелов с конца строки
# text = "    Не трогай лишнее!   "
# clean_text = text.rstrip()
# print(clean_text)  # Вывод: "    Не трогай лишнее!"
#
# text = "xxxyyyЗачем так сложно?yyxxx"
# clean_text = text.strip('xy')
# print(clean_text)  # Вывод: "Зачем так сложно?"
#
# text = "   Модный код   "
# clean_text = text.strip()
# print(clean_text)  # Вывод: "Модный код"
#
# rate = 50
# rate_month = rate / 12 / 100
# final_summ = start_summ * (1 + rate_month) ** months
# m = max (3, 5, 6.5, 1)
# print(m) #6.5




# a=float(input())
# b=float(input())
# c=float(input())
# summ=a+b
# if abs(summ-c) < 0.0000001:
#     print("Результат верный")
# else:
#     print("Это не подходит")
#
# a = 20
# b = 5
# result = a / b
#
# # Проверка с учетом погрешности
# if abs(result - 4) < 0.0000001:
#     print("Результат верный")
# else:
#     print("Результат с погрешностью")
#
#
# a1=float(input())
# print(round(a1))

# a1=int(input())
# a2=float(input())
# a3=int(input())
# ss=3.14*((a1/2)**2)
# piec=ss/(a3+1)
#
# if piec>=a2:
#     print("Всем точно хватит")
# else:
#     print("Придется уменьшить порции")
#
# a1=int(input())
# a2=float(input())
# a3=int(input())
# ss=3.14*((a1/2)**2)
# piec=ss/a3
#
# print(a1)
# print(a2)
# print(a3)
# print(piec)
#
# if piec>=a2:
#     print("Всем точно хватит")
# else:
#     print("Придется уменьшить порции")

# a1=int(input())
# a2=float(input())
# a3=int(input())
# ss=3.14*((a1/2)**2)
# piec=ss/a3
# if piec>=a2:
#     print("Всем точно хватит")
# else:
#     print("Придется уменьшить порции")
#
# a1=float(input())
# a2=float(input())
# sum=round(a1,1)+round(a2,1)
# print(round(sum,1))


# a1=float(input())
# a2=float(input())
# sum=round(a1,1)+round(a2,1)
# print(sum)
#
#
# # Целые числа
# a = 10**16 + 1
# b = 3
#
# # Деление с образованием погрешности
# result = a / b
# print(f"Результат деления: {result:.15f}")
#
# # Проверка с использованием оператора if
# expected_result = 10**16 / 3
# if result == expected_result:
#     print("Результат точный")
# else:
#     print(f"Результат с погрешностью: {result - expected_result:.15f}")
#
# # Изначально переменная целое число
# num = 10
# print(f"Тип переменной num: {type(num)}")  # Вывод: <class 'int'>
#
# # Деление переменной, результат - вещественное число
# num = num / 2
# print(f"Значение переменной num: {num}")
# print(f"Тип переменной num: {type(num)}")  # Вывод: <class 'float'>

#
# animal_type = input("Введите тип животного (млекопитающее, птица): ")
# size = input("Размер животного (крупное, маленькое): ")
# can_fly = input("Умеет ли животное летать (да/нет): ")
#
# # создание переменной-флага
# flag = False
#
# # Проверяем условия и выводим соответствующие сообщения
# if animal_type == "млекопитающее":
#     if size == "крупное":
#         print("Это слон")
#         flag = True
#     elif size == "маленькое":
#         print("Это мышь")
#         flag = True
# elif animal_type == "птица":
#     if can_fly == "да":
#         print("Это воробей")
#         flag = True
#
# # Если ни одно из условий не подошло
# if not flag:
#     print("Неизвестный вид животного")



# legs=int(input())
# par=legs/2
# if legs==8:
#     #print(f"{legs} ног — паук")
#     print(f"паук")
# elif par%2!=0 and 15<=par<=80:
#     print(f"многоножка")
# else:
#     print(f"мы не знаем, что это такое")
#
# x = 25
# if 15 <= x <= 80:
#     print("x находится в пределах от 15 до 80")
# else:
#     print("x не входит в указанный диапазон")
# x = 25
# if 15 <= x and x <= 80:
#     print("x находится в пределах от 15 до 80")
# else:
#     print("x не входит в указанный диапазон")

# a1=input()
# p=a.lower().replace(" ", "")
# res=p[::-1]
# if res==p:
#     print("Эта фраза - палиндром")
# else:
#     print("Ничего подобного")
#
# a1=input()
# p=a1.lower().replace(" ", "")
# res=p[::-1]
# print(res)

# res=int(input())
# a1=int(input())
# a2=int(input())
# a3=int(input())
# a4=int(input())
#
# if res>=a1:
#     print("великолепно")
# elif res<a1 and res >=a2:
#     print("достойно")
# elif res<a2 and res >=a3:
#     print("моглобыбытьилучше")
# elif res<a3 and res>=a4:
#     print("нутакое")
# else:
#     print("нда")

# a1=int(input())
# a2=input()
#
# if a2=="подкрадули":
#     if a1==52 or a1==53:
#         print(f"Подкрадули - лучшее, что может быть, размерчик, кстати мой")
#     else:
#         print(f"Подкрадули - лучшее, что может быть, маловаты, как обычно")
# else:
#     if a1==52 or a1==53:
#         print(f'{a2} - совсем не подходит мне по стилю, размерчик, кстати мой')
#     else:
#         print(f'{a2} - совсем не подходит мне по стилю, маловаты, как обычно')

# a1=input()
# a2=input()
# if (a1=="не_продаю" and a2=="показываю") or (a2=="не_продаю" and a1=="показываю"):
#     print("красивое")
# else:
#     print("грустное")


# a1=input()
# a2=int(input())
# if a1=="сентябрь" and a2 == 3:
#     print("Горят костры рябин")
# else:
#     print("Ждем 3 сентября")
#
# password = "securepassword123"
# user_active = True
# is_admin = False
#
# if (password == "securepassword123" or password == "admin123") and user_active and not is_admin:
#     print("Пароль принят, пользователь активен и не является администратором")
# else:
#     print("Либо пароль неверен, либо пользователь не активен, либо это администратор")
#
# is_admin = False
#
# if not is_admin:
#     print("Доступ запрещен")
# else:
#     print("Добро пожаловать, администратор")
#
# password = "admin123"
# backup_password = "adminbackup"
#
# if password == "admin123" or password == "adminbackup":
#     print("Пароль принят")
# else:
#     print("Пароль неверен")

# a1 =input()
# a2 =input()
# if a1= ="файлообменник":
#     if a2= ="скайп":
#         print("С-К-А-А-А-Й-П")


# a1=int(input())
# if a1%2 == 0:
#     print("YES")
# else:
#     print("NO")

# a1=int(input())
# print(f"a1%10 = {a1%10}")
# print(f"a1//10 = {a1//10}")
#
# if a1%10 == a1//10:
#     print("Как две капли")
# else:
#     print("Не совпали")

# a1=input()
# a2=a1[::-1]
# if a1==a2:
#     print("YES")
# else:
#     print("NO")


# a1=input()
# if a1.isupper():
#     print("Не ори, пожалуйста!")
# else:
#     print("Хорошо сидим, болтаем")

# a1=int(input())
# if a1 < 18:
#     print("Школьник")
# else:
#     print("Студент")
#
# a1=input()
# l=len(a1)
# if l >= 10:
#     print("Она огромная")
# else:
#     print("Она что, маленькая?")

# a1=input()
# d1=int(input())
# d2=int(input())
# d3=int(input())
# print(a1[d1:d2:d3])
#
# a1=input()
# print(a1[18:-1])

# a1=input()
# a2=int(input())
# print(f'В строке "{a1}" символ с индексом {a2} вот такой: "{a1[a2]}"')
#
# a1=input()
# print(f'В строке "{a1}" первый символ: "{a1[0]}", а последний символ "{a1[-1]}"')

# a1=input()
# print(a1[0])
#
# a1=input()
# print(f"Проверка, что тут только цифры! Результат: {a1.isdigit()}")
#
# a1 = input()
# print(a1.startswith("Дорогой"))


# a1 = input()
# res = len(a1)
# print(f'В строке "Исходная строка" - {res} символов, включая пробелы')
#
# a1 = input()
# res = len(a1)
# print(f'В строке "{a1}" - {res} символов, включая пробелы')


# a1=input()
# print(f'"{a1.lower()}"')

# a1=input()
# print(a1.upper())
#
#
# a1 = input()
# res = len(a1)
# print(res)

# a1=input()
# result = eval(a1)
# print(f"{a1}={result}")

# a1 = int(input())
# a2 = int(input())
# print(f"Сумма чисел {a1} и {a2} равна: {a1+a2}, при этом их разность равна: {a1-a2}")
#
# a1 = int(input())
# a2 = int(input())
# print(f"Первое число: {a1}")
# print(f"Второе число: {a2}")
# print(f"Сумма чисел: {a1+a2}")
# print(f"Разность чисел: {a1-a2}")
# print(f"Остаток от деления: {a1%a2}")

# a = 3
# b = 4
# print(f"Сумма чисел {a} и {b} равна {a + b}")
#
# a = 5
# b = 4
# print(f"Сумма чисел {a} и {b} равна {a + b}\nПроизведение чисел {a} и {b} равно {a * b}")

# year = int(input())
# result = year + 1
# print(result)
#
# year = int(input())
# classChild = year-7+1
# print(classChild,"класс")

# print("apple", "banana", "cherry", sep=", ")
#
# my_symbol = "@"
# print(1, "2 слово", 3, sep=my_symbol)
#
# one, two, three, four  = input(), input(), input(), input()

# print(6+9)
# a = "pe"
# b = "rec"
# print(a + b)
#
# a = "h"
# b = 6
# print(a * b)
#
# name = input("Введите свое имя: ")
# age = int(input("Введите свой возраст: "))
#
# a = input("Введите что-нибудь: ")
# print("Вы ввели:", a)
#
# str1 = "Hello"
# str2 = "World"
# result = str1 + " " + str2
# print(result)  # Вывод: Hello World
#
# num1 = 10
# num2 = 20
# result = num1 + num2
# print(result)  # Вывод: 30
#
# num1 = input("Введите первое число: ")
# num2 = input("Введите второе число: ")
#
# num1 = int(num1)
# num2 = int(num2)
#
# result = num1 + num2
# print("Сумма чисел:", result)
#
# num1 = int(input("Введите первое число: "))
# num2 = int(input("Введите второе число: "))
# result = num1 + num2
# print("Сумма чисел:", result)
#
# text = "Привет"
# num = 3
# result = text * num
# print(result)  # Вывод: ПриветПриветПривет