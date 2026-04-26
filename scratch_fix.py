import re

texto_img = "Imagen 13.4"
etiquetas_en_img = re.findall(r'(?:imagen|figura)\s*(\d+(?:[\-\.·]\d+)?)', texto_img.lower())
print(etiquetas_en_img)
