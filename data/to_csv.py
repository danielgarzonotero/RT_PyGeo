import pandas as pd

# Lee el archivo de texto
data = pd.read_csv('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/RT_PyGeo/data/dia.txt', delimiter='\t')

# Guarda el dataframe en un archivo CSV
data.to_csv('data/dia.csv', index=False)
