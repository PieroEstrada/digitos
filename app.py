import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import io
from PIL import Image, ImageFilter

app = FastAPI()

def build_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, activation='relu', input_dim=784),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.load_weights("midigito.weights.h5")
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    return model

def otsu(valores):
    h, b = np.histogram(valores, bins=256, range=(0, 256))
    p = h / h.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    s = (m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-9)
    return np.argmax(s)

def representativo(im):
    im = im.convert("L")
    im.thumbnail((300, 300))
    gris = np.array(im, np.float32)
    fondo = np.array(im.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(10)), np.float32)
    tinta = np.clip(fondo - gris, 0, 255)
    tinta = tinta * (255.0 / max(tinta.max(), 1))
    umbral = max(otsu(tinta), 60)
    mascara = Image.fromarray(((tinta > umbral) * 255).astype(np.uint8))
    mascara = mascara.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    m = np.array(mascara) > 0
    if m.sum() < 20:
        raise ValueError("No se encontro ningun numero en la foto")
    filas = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    y0, y1, x0, x1 = filas[0], filas[-1] + 1, cols[0], cols[-1] + 1
    m = m[y0:y1, x0:x1]
    recorte = np.where(m, tinta[y0:y1, x0:x1], 0)
    alto, ancho = recorte.shape
    interior = np.array(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))) > 0
    grosor = 2.0 * m.sum() / max((m & ~interior).sum(), 1)
    falta = int(round(0.12 * max(alto, ancho) - grosor))
    img = Image.fromarray(np.clip(recorte * 1.5, 0, 255).astype(np.uint8))
    if falta >= 2:
        img = img.filter(ImageFilter.MaxFilter(falta // 2 * 2 + 1))
    escala = 20.0 / max(alto, ancho)
    img = img.resize((max(1, round(ancho * escala)), max(1, round(alto * escala))), Image.LANCZOS)
    a = np.array(img, np.float32)
    yy, xx = np.indices(a.shape)
    cy = (yy * a).sum() / a.sum()
    cx = (xx * a).sum() / a.sum()
    oy = min(max(int(round(14 - cy)), 0), 28 - a.shape[0])
    ox = min(max(int(round(14 - cx)), 0), 28 - a.shape[1])
    lienzo = np.zeros((28, 28), np.float32)
    lienzo[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
    lienzo = lienzo / max(lienzo.max(), 1)
    return lienzo.reshape(-1)

model = build_model()

@app.get("/salud")
def salud():
    return {"mensaje": "API de digitos funcionando"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model
    try:
        contents = await file.read()
        im = Image.open(io.BytesIO(contents))
        rep = representativo(im)
        prediction = model.predict(np.array([rep[:]]), verbose=0)[0]
        digito = int(np.argmax(prediction))
        confianza = round(float(prediction[digito]) * 100, 2)
        print("Digito:", digito, "Confianza:", confianza)
        return {"prediction": str(digito), "confianza": confianza}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
