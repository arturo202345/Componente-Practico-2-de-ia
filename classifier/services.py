"""Carga de modelos, normalización, inferencia y renderizado de espectrogramas."""
import base64
import io
import json
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings

from . import features as F

CLASES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]
ETIQUETAS_ES = {
    "air_conditioner": "Aire acondicionado",
    "car_horn": "Bocina de auto",
    "children_playing": "Niños jugando",
    "dog_bark": "Ladrido de perro",
    "drilling": "Taladro",
    "engine_idling": "Motor en ralentí",
    "gun_shot": "Disparo",
    "jackhammer": "Martillo neumático",
    "siren": "Sirena",
    "street_music": "Música callejera",
}

MODELOS = {
    "cnn2d": {
        "nombre": "CNN-2D · log-mel",
        "archivo": "cnn2d_espectrograma_optimizado.keras",
        "descripcion": "Espectrograma log-mel 64×173 tratado como imagen",
        "metrica": "Accuracy en test: 0.751",
    },
    "dnn": {
        "nombre": "DNN · MFCC",
        "archivo": "dnn_mfcc_optimizado.keras",
        "descripcion": "160 descriptores: media y desv. de 40 MFCC + 40 delta",
        "metrica": "Accuracy en test: 0.648",
    },
}

_lock = threading.Lock()
_cache = {}
_stats = None


def _cargar_modelo(clave):
    with _lock:
        if clave not in _cache:
            import keras

            ruta = settings.MODELOS_DIR / MODELOS[clave]["archivo"]
            _cache[clave] = keras.saving.load_model(ruta, compile=False)
        return _cache[clave]


def _cargar_stats():
    """Estadísticas z-score de train (norm_stats.json). None si no existe."""
    global _stats
    if _stats is None:
        ruta = settings.NORM_STATS_PATH
        _stats = json.loads(ruta.read_text()) if ruta.exists() else {}
    return _stats or None


def hay_stats():
    return _cargar_stats() is not None


# ---------- normalización ----------
def _norm_logmel(S):
    st = _cargar_stats()
    if st and "cnn2d" in st:
        return (S - st["cnn2d"]["mean"]) / (st["cnn2d"]["std"] + 1e-8)
    # Aproximación sin estadísticos de train: z-score con el propio clip
    return (S - S.mean()) / (S.std() + 1e-8)


def _norm_mfcc(v):
    st = _cargar_stats()
    if st and "dnn" in st:
        mu = np.array(st["dnn"]["mean"], dtype=np.float32)
        sd = np.array(st["dnn"]["std"], dtype=np.float32)
        return (v - mu) / (sd + 1e-8)
    # Aproximación: z-score dentro de cada bloque de 40 coeficientes
    out = v.copy()
    for i in range(0, 160, 40):
        b = v[i:i + 40]
        out[i:i + 40] = (b - b.mean()) / (b.std() + 1e-8)
    return out


# ---------- renderizado ----------
def _png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _estilo_ejes(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#8a8578", labelsize=8, length=0)
    ax.xaxis.label.set_color("#8a8578")
    ax.yaxis.label.set_color("#8a8578")


def render_espectrograma(clave, y):
    """Imagen de la representación que ve el modelo seleccionado."""
    import librosa.display

    if clave == "cnn2d":
        S = F.logmel(y)
        fig, ax = plt.subplots(figsize=(7, 2.7))
        im = librosa.display.specshow(
            S, sr=F.SR, hop_length=F.HOP, x_axis="time", y_axis="mel",
            cmap="magma", ax=ax,
        )
        cb = fig.colorbar(im, ax=ax, format="%+2.0f dB", pad=0.015)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors="#8a8578", labelsize=8, length=0)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Hz (mel)")
    else:
        mfcc, delta = F.mfcc_delta(y)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 3.4), sharex=True)
        for ax, M, tit in ((a1, mfcc, "MFCC"), (a2, delta, "Delta-MFCC")):
            librosa.display.specshow(
                M, sr=F.SR, hop_length=512, x_axis="time", cmap="magma", ax=ax
            )
            ax.set_ylabel(tit)
            _estilo_ejes(ax)
        a1.set_xlabel("")
        a2.set_xlabel("Tiempo (s)")
        ax = None
    if ax is not None:
        _estilo_ejes(ax)
    fig.tight_layout()
    return _png_b64(fig)


def render_onda(y):
    fig, ax = plt.subplots(figsize=(7, 1.1))
    # Envolvente por bloques: evita el moiré de dibujar cientos de miles de puntos
    n = 700
    paso = max(len(y) // n, 1)
    bloques = y[: paso * (len(y) // paso)].reshape(-1, paso)
    env = np.abs(bloques).max(axis=1)
    t = np.linspace(0, len(y) / F.SR, len(env))
    ax.fill_between(t, env, -env, color="#c15f3c", linewidth=0)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0)
    return _png_b64(fig)


# ---------- inferencia ----------
def predecir(clave, y):
    modelo = _cargar_modelo(clave)
    if clave == "cnn2d":
        S = F.logmel(y)
        x = _norm_logmel(S)[None, :, :, None].astype(np.float32)
    else:
        x = _norm_mfcc(F.vector_mfcc(y))[None, :].astype(np.float32)

    probs = np.asarray(modelo.predict(x, verbose=0))[0]
    orden = np.argsort(probs)[::-1]
    ranking = [
        {
            "clase": CLASES[i],
            "etiqueta": ETIQUETAS_ES[CLASES[i]],
            "prob": float(probs[i]),
        }
        for i in orden
    ]
    return {
        "clase": ranking[0]["clase"],
        "etiqueta": ranking[0]["etiqueta"],
        "confianza": ranking[0]["prob"],
        "ranking": ranking,
        "espectrograma": render_espectrograma(clave, y),
        "onda": render_onda(y),
        "duracion": round(len(y) / F.SR, 2),
        "stats_train": hay_stats(),
    }
