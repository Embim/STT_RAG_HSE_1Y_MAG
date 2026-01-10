"""
GPU utilities.

Функции для управления GPU памятью.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def clear_gpu_cache() -> None:
    """Очищает кеш GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    except ImportError:
        pass


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Возвращает информацию о GPU.

    Returns:
        Словарь с информацией о GPU или None
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device_count = torch.cuda.device_count()
        devices = []

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024 ** 3)  # GB
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            memory_cached = torch.cuda.memory_reserved(i) / (1024 ** 3)

            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(memory_total, 2),
                "allocated_memory_gb": round(memory_allocated, 2),
                "cached_memory_gb": round(memory_cached, 2),
                "free_memory_gb": round(memory_total - memory_allocated, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })

        return {
            "cuda_available": True,
            "device_count": device_count,
            "current_device": torch.cuda.current_device(),
            "devices": devices,
        }

    except ImportError:
        return None
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return None


def get_optimal_batch_size(
    model_memory_gb: float = 2.0,
    item_memory_mb: float = 50.0,
    safety_factor: float = 0.8,
) -> int:
    """
    Рассчитывает оптимальный batch size для GPU.

    Args:
        model_memory_gb: Память занимаемая моделью в ГБ
        item_memory_mb: Память на один элемент батча в МБ
        safety_factor: Коэффициент безопасности (0.8 = использовать 80% памяти)

    Returns:
        Рекомендуемый batch size
    """
    gpu_info = get_gpu_info()

    if not gpu_info or not gpu_info["devices"]:
        return 32  # Default для CPU

    # Берём текущее устройство
    current = gpu_info["current_device"]
    device = gpu_info["devices"][current]

    free_memory_gb = device["free_memory_gb"]
    available_memory_gb = (free_memory_gb - model_memory_gb) * safety_factor

    if available_memory_gb <= 0:
        return 1

    # Конвертируем в МБ
    available_memory_mb = available_memory_gb * 1024

    batch_size = int(available_memory_mb / item_memory_mb)
    return max(1, batch_size)


def is_cuda_available() -> bool:
    """Проверяет доступность CUDA."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(prefer_cuda: bool = True) -> str:
    """
    Возвращает устройство для вычислений.

    Args:
        prefer_cuda: Предпочитать CUDA если доступно

    Returns:
        "cuda" или "cpu"
    """
    if prefer_cuda and is_cuda_available():
        return "cuda"
    return "cpu"
