from PIL import Image
import numpy as np

# Abertura de uma imagem
im = Image.open("uepg_sm.jpg")

# Transformacao de rotacao
im = im.rotate(45)

# Gravacao de uma imagem
im.save("pil_example_01.png")

# Um array em numpy a partir de uma imagem PIL
arr = np.array(im)

# Uma imagem a partir de um array
im2 = Image.fromarray(arr.astype("uint8"), "RGB")

im2.save("pil_example_02.png")

# Conversao de uma imagem RGB para escala de cinza
im3 = im.convert("L")

im3.save("pil_example_03.png")

# Conversao de uma imagem para preto e branco
im4 = im.convert("1")

im4.save("pil_example_04.png")

# Um array com valores aleatorios em numpy
arr = np.random.rand(200,300,3)*255

# Imagem aleatoria criada a partir de um numpy array
im5 = Image.fromarray(arr.astype("uint8"))

im5.save("pil_example_05.png")
