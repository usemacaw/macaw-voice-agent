"""
Executor compartilhado para inferencia de modelos ML.

Limita o numero de threads concorrentes para evitar sobrecarga
de CPU/GPU quando multiplas sessoes rodam inferencia simultaneamente.

Uso:
    from providers.executor import run_inference

    result = await run_inference(model.transcribe, audio)
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Limite de threads para inferencia (default: 2)
# Em GPU, operacoes sao serializadas pelo hardware — mais threads nao ajudam.
# Em CPU, limitar evita contencao de recursos.
_MAX_INFERENCE_WORKERS = int(os.getenv("INFERENCE_MAX_WORKERS", "2"))

inference_executor = ThreadPoolExecutor(
    max_workers=_MAX_INFERENCE_WORKERS,
    thread_name_prefix="inference",
)


async def run_inference(fn, *args, **kwargs):
    """Executa funcao blocking de inferencia no executor limitado.

    Substitui asyncio.to_thread() e run_in_executor(None, ...) para
    garantir que no maximo N inferencias rodem em paralelo.

    Args:
        fn: Funcao blocking (ex: model.transcribe, model.generate).
        *args: Argumentos posicionais para fn.
        **kwargs: Argumentos nomeados para fn.

    Returns:
        Resultado de fn(*args, **kwargs).
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(
            inference_executor, lambda: fn(*args, **kwargs)
        )
    return await loop.run_in_executor(inference_executor, fn, *args)
