"""Calcula las estadísticas z-score de TRAIN y las guarda en modelos/norm_stats.json.

Los modelos .keras no incluyen la normalización de entrada, así que la app
necesita los mismos estadísticos que usó el entrenamiento (solo clips de train,
sin aumento de datos).

Uso:
    python compute_norm_stats.py --particion particion.json --audio_dir RUTA/UrbanSound8K

--audio_dir debe contener los .wav (se busca recursivamente por nombre de archivo).
particion.json debe tener la lista de train; se aceptan estos formatos:
    {"train": ["134717-0-0-0.wav", ...], ...}
    {"train": [{"archivo": "...wav"}, ...]}   (también claves "file", "filename", "path")
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from classifier import features as F  # noqa: E402


def nombre(item):
    if isinstance(item, str):
        return Path(item).name
    if isinstance(item, (list, tuple)):  # formato [ruta, etiqueta]
        return nombre(next(x for x in item if isinstance(x, str)))
    for k in ("archivo", "file", "filename", "path", "nombre", "name", "clip"):
        if k in item:
            return Path(item[k]).name
    raise ValueError(f"Formato de elemento no reconocido: {item}")


CLAVES_TRAIN = ("train", "training", "entrenamiento", "entrenar", "tr")


def es_train(valor):
    return str(valor).strip().lower() in CLAVES_TRAIN


def obtener_train(part):
    """Devuelve la lista de nombres de archivo de train, aceptando varios formatos."""
    # 1) {"train": [...]} (también anidado un nivel: {"particion": {"train": [...]}})
    if isinstance(part, dict):
        for nivel in (part, *[v for v in part.values() if isinstance(v, dict)]):
            for k, v in nivel.items():
                if es_train(k) and isinstance(v, list):
                    return [nombre(x) for x in v]
    # 2) {"archivo.wav": "train", ...}
    if isinstance(part, dict) and all(isinstance(v, str) for v in part.values()):
        return [Path(k).name for k, v in part.items() if es_train(v)]
    # 3) [{"archivo": "...", "split": "train"}, ...]
    if isinstance(part, list):
        out = []
        for it in part:
            split = next((it[k] for k in ("split", "particion", "set", "subset", "conjunto") if k in it), None)
            if split is not None and es_train(split):
                out.append(nombre(it))
        if out:
            return out
    # Diagnóstico
    if isinstance(part, dict):
        info = {k: (f"lista de {len(v)}" if isinstance(v, list) else type(v).__name__) for k, v in part.items()}
    else:
        info = f"lista de {len(part)} elementos; primero: {part[0] if part else None}"
    sys.exit(f"No pude encontrar la partición de train en particion.json. Estructura: {info}\n"
             "Envíame las primeras líneas del archivo para ajustar el script.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--salida", default=str(Path(__file__).parent / "modelos" / "norm_stats.json"))
    ap.add_argument("--max_clips", type=int, default=0,
                    help="Usar solo una muestra aleatoria de N clips de train (0 = todos). "
                         "Con ~2000 la estimación es casi idéntica y mucho más rápida.")
    a = ap.parse_args()

    part = json.loads(Path(a.particion).read_text())
    train = obtener_train(part)
    if a.max_clips and a.max_clips < len(train):
        rng = np.random.default_rng(42)
        train = list(rng.choice(train, size=a.max_clips, replace=False))
    indice = {p.name: p for p in Path(a.audio_dir).rglob("*.wav")}
    print(f"{len(train)} clips en train; {len(indice)} wav encontrados")

    s = n = s2 = 0.0           # log-mel: estadísticos globales
    vecs = []                  # DNN: un vector de 160 por clip

    for i, nom in enumerate(train, 1):
        if nom not in indice:
            continue
        try:
            y = F.cargar_audio(indice[nom])
        except Exception:
            continue
        if len(y) == 0:
            continue
        S = F.logmel(y)
        s += S.sum(); s2 += (S.astype(np.float64) ** 2).sum(); n += S.size
        vecs.append(F.vector_mfcc(y))
        if i % 500 == 0:
            print(f"  {i}/{len(train)}")

    mean = s / n
    std = float(np.sqrt(s2 / n - mean ** 2))
    V = np.stack(vecs)
    out = {
        "cnn2d": {"mean": float(mean), "std": std},
        "dnn": {"mean": V.mean(axis=0).tolist(), "std": V.std(axis=0).tolist()},
    }
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.salida).write_text(json.dumps(out))
    print(f"Guardado en {a.salida} (clips usados: {len(vecs)})")


if __name__ == "__main__":
    main()