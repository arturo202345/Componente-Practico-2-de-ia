"""Extracción de características — réplica del preprocesamiento del informe (Tabla 3).

Este módulo no depende de Django para poder reutilizarlo en compute_norm_stats.py.

DNN + MFCC : sr 22,050 Hz, 40 MFCC y 40 delta-MFCC (n_fft 2048) resumidos con
             media y desviación estándar -> vector de 160 (duración real del clip).
CNN-2D     : sr 22,050 Hz, 4 s (recorte/relleno con ceros), log-mel 64 x 173
             (n_fft 2048, hop 512, top_db 80), z-score global con estadísticos de train.
"""
import numpy as np
import librosa

SR = 22050
DURACION = 4.0
N_SAMPLES = int(SR * DURACION)
N_FFT = 2048
HOP = 512
N_MELS = 64
N_MFCC = 40
TOP_DB = 80


def cargar_audio(fuente, duracion_max=None):
    """Carga un audio (ruta o file-like) a mono, 22,050 Hz."""
    y, _ = librosa.load(fuente, sr=SR, mono=True, duration=duracion_max)
    return y.astype(np.float32)


def fijar_duracion(y):
    """Recorta o rellena con ceros hasta exactamente 4 s."""
    if len(y) >= N_SAMPLES:
        return y[:N_SAMPLES]
    return np.pad(y, (0, N_SAMPLES - len(y)))


def logmel(y):
    """Espectrograma log-mel en dB (64 x 173) de un clip de 4 s."""
    y = fijar_duracion(y)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB).astype(np.float32)


def mfcc_delta(y):
    """MFCC y delta-MFCC (40 x T) sobre la duración real del clip."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT)
    # width=9 por defecto; si el clip es muy corto se reduce para evitar error
    ancho = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    delta = librosa.feature.delta(mfcc, width=max(ancho, 3), mode="nearest")
    return mfcc, delta


def vector_mfcc(y):
    """Vector de 160 descriptores: [media MFCC, std MFCC, media delta, std delta]."""
    mfcc, delta = mfcc_delta(y)
    v = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1), delta.std(axis=1)]
    )
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
