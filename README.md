# Clasificador de sonidos urbanos (Django)

Interfaz web para usar los modelos **CNN-2D (log-mel)** y **DNN (MFCC)** entrenados sobre UrbanSound8K.
Flujo: **audio → espectrograma → clase predicha**, con selector de modelo.

## Ejecutar

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

Abre http://127.0.0.1:8000/ y arrastra un audio (WAV, MP3, FLAC, OGG). Al cambiar de modelo se vuelve a clasificar el mismo audio.

## Importante: estadísticas de normalización

Los `.keras` no guardan el z-score de entrada que se calculó con los datos de train. Sin ese archivo la app
normaliza con el propio clip (aproximado; la CNN-2D lo tolera mejor que la DNN). Para usar la normalización real:

```bash
python compute_norm_stats.py --particion particion.json --audio_dir RUTA/UrbanSound8K
```

Esto crea `modelos/norm_stats.json` y la app lo usa automáticamente (el aviso desaparece). Solo usa los clips de `train`, sin aumento.

## Supuestos del preprocesamiento (Tabla 3 del informe)

- CNN-2D: 22,050 Hz, 4 s, log-mel 64×173 (n_fft 2048, hop 512, `power_to_db(ref=max, top_db=80)`), z-score global.
- DNN: 40 MFCC + 40 delta (n_fft 2048), vector `[media MFCC, std MFCC, media Δ, std Δ]` = 160, z-score por característica.
- Orden de clases: el de UrbanSound8K (0 air_conditioner … 9 street_music).

Si en tu script de entrenamiento el orden del vector MFCC, la referencia de `power_to_db` u otro detalle difiere,
ajusta `classifier/features.py` (es el único lugar donde está).

## Estructura

```
config/            ajustes y urls de Django (sin base de datos)
classifier/
  features.py      extracción de características (compartida con compute_norm_stats.py)
  services.py      carga de modelos, normalización, predicción, imágenes
  views.py         index + endpoint /predecir/
  templates/ static/
modelos/           los dos .keras (+ norm_stats.json cuando lo generes)
```
