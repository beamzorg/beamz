import json
import threading
import time

import torch

results = {}


def burn(i):
    with torch.cuda.device(i):
        a = torch.randn((8192, 8192), device=f"cuda:{i}", dtype=torch.float16)
        b = a.clone()
        out = torch.empty_like(a)
        torch.cuda.synchronize()
        start = time.monotonic()
        n = 0
        while time.monotonic() - start < 45:
            torch.mm(a, b, out=out)
            torch.cuda.synchronize()
            n += 1
        results[i] = dict(iterations=n, seconds=time.monotonic() - start)


threads = [threading.Thread(target=burn, args=(i,)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(json.dumps(results))
