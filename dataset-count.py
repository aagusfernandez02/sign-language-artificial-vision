import csv
from collections import Counter

csv_filename = 'dataset.csv'

# Leer todas las etiquetas (última columna)
labels = []

with open(csv_filename, mode='r', newline='') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Saltear encabezados

    for row in reader:
        if row:  # Ignorar filas vacías
            labels.append(row[-1])  # Última columna = etiqueta

# Contar ejemplos por letra
counter = Counter(labels)

# Mostrar resultados ordenados alfabéticamente
for letra in sorted(counter):
    print(f"Letra {letra}: {counter[letra]} ejemplos")

print(f"\nTotal de ejemplos: {sum(counter.values())}")
