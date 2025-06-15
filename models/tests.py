import pandas as pd
import numpy as np
from collections import Counter

import os
from graphviz import Digraph

# Укажи путь до bin, если не в PATH
os.environ["PATH"] += os.pathsep + r"C:\Graphviz\bin"

# Создаём граф
dot = Digraph()
dot.node('A', 'Начало')
dot.node('B', 'Конец')
dot.edge('A', 'B', label='Переход')

# Сохраняем и открываем
dot.render('test_graph', format='png', view=True)

