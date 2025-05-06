

#1 2 3 4 5
#2
#2 8

# import numpy as np
# str=input()
# nl=np.array(list(map(int, str.split())))
# pl=int(input())
# rstr=list(map(int, input().split()))
#
# print(nl)
# print(pl)
# print(f"{rstr}")
#
# result=np.clip(nl*pl, rstr[0], rstr[1])
# print(result)
#
#
# import numpy as np
# str=input()
# nl=np.array(list(map(int, str.split())))
#
# import numpy as np
# str=input()
# nl=np.array(list(map(int, str.split())))
# indicies=np.where(nl % 2 == 0)
# nl[indicies]=nl[indicies]**2
# print(nl)


# Сложение элементов матрицы
# Условие задачи
# Напишите программу, которая принимает матрицу чисел, суммирует все элементы, а также вычисляет суммы по строкам и столбцам.
#
# Входные данные
# Первая строка: два целых числа, разделенные пробелами (размеры матрицы: количество строк и столбцов).
# Следующие строки: целые числа, разделенные пробелами (элементы матрицы).
# Выходные данные
# Сумма всех элементов матрицы.
# Суммы по строкам.
# Суммы по столбцам.

# 3 3
# 1 2 3
# 4 5 6
# 7 8 9
import numpy as np
str=input()
nl=np.array(list(map(int, str.split())))
shp=input()
nlshp=np.array(list(map(int, shp.split())))
matrix=nl.reshape(nlshp[0], nlshp[1])
print(f"Сумма всех элементов матрицы: {np.sum(matrix)}")
print(f"Суммы по строкам:  {np.sum(matrix, axis=1)}")
print(f"Суммы по столбцам: {np.sum(matrix, axis=0)}")