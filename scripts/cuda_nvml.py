"""Minimal Linux NVML memory fallback for benchmark environments without pynvml.

The stable nvmlMemory_t ABI is declared in the installed CUDA nvml.h. This
module reads telemetry only and creates no CUDA context or device allocation.
"""

import ctypes


class MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlMemory:
    backend = "ctypes_nvml_memory_v1"

    def __init__(self):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        for name, arguments in (
            ("nvmlInit_v2", []),
            ("nvmlShutdown", []),
            (
                "nvmlDeviceGetHandleByIndex_v2",
                [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            ),
            ("nvmlDeviceGetMemoryInfo", [ctypes.c_void_p, ctypes.POINTER(MemoryInfo)]),
        ):
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f"NVML memory query failed with status {result}")

    def nvmlInit(self):
        self.check(self.library.nvmlInit_v2())

    def nvmlShutdown(self):
        self.check(self.library.nvmlShutdown())

    def nvmlDeviceGetHandleByIndex(self, index):
        handle = ctypes.c_void_p()
        self.check(
            self.library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))
        )
        return handle

    def nvmlDeviceGetMemoryInfo(self, handle):
        memory = MemoryInfo()
        self.check(self.library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
        return memory
