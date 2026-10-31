import json
import sys
from PIL import Image

img = Image.open(sys.argv[1]).convert("L")
# img = img.rotate(90, expand=True)  # descomente se a foto sair deitada
img = img.resize((28, 28), Image.LANCZOS)
pixels = [255 - p for p in img.tobytes()]  # fundo preto, traço branco
print(json.dumps({"pixels": pixels, "top_k": 3}))
