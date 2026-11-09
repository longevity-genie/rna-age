"""2-minute GPU probe: device, memory, bf16 support, a matmul timing."""
import time

import torch

print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, f"{p.total_memory / 2**30:.1f} GiB", f"cc {p.major}.{p.minor}")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(4096, 4096, device="cuda", dtype=dt)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
        print(dt, f"{20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOP/s")
