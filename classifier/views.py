import io

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from . import features as F
from . import services


def index(request):
    return render(
        request,
        "classifier/index.html",
        {
            "modelos": services.MODELOS,
            "stats_train": services.hay_stats(),
            "max_mb": settings.MAX_UPLOAD_MB,
        },
    )


@require_POST
def predecir(request):
    clave = request.POST.get("modelo", "cnn2d")
    if clave not in services.MODELOS:
        return JsonResponse({"error": "Modelo no válido."}, status=400)

    archivo = request.FILES.get("audio")
    if archivo is None:
        return JsonResponse({"error": "No se recibió ningún audio."}, status=400)
    if archivo.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        return JsonResponse(
            {"error": f"El archivo supera {settings.MAX_UPLOAD_MB} MB."}, status=413
        )

    try:
        # Se limita a 60 s: los modelos usan 4 s (CNN-2D) o el clip completo (DNN)
        y = F.cargar_audio(io.BytesIO(archivo.read()), duracion_max=60)
    except Exception:
        return JsonResponse(
            {"error": "No se pudo leer el audio. Usa WAV, MP3, FLAC u OGG."},
            status=400,
        )
    if len(y) < F.SR * 0.2:
        return JsonResponse({"error": "El audio es demasiado corto o está vacío."}, status=400)

    # La CNN-2D ve solo los primeros 4 s; se muestra exactamente eso
    if clave == "cnn2d":
        y = F.fijar_duracion(y)

    try:
        return JsonResponse(services.predecir(clave, y))
    except Exception as e:  # noqa: BLE001
        return JsonResponse({"error": f"Error al predecir: {e}"}, status=500)
