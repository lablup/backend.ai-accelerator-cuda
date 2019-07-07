Backend.AI Accelerator Plugin for CUDA
======================================

Just install this along with Backend.AI agents, using the same virtual environment.
This will allow the agents to detect CUDA devices on their hosts and make them
available to Backend.AI kernel sessions.

```console
$ pip install backend.ai-accelerator-cuda
```

This open-source edition of CUDA plugins support allocation of one or more CUDA
devices to a container, slot-by-slot.

Compatibility Matrix
--------------------

| Backend.AI | CUDA Plugin |
|:----------:|:-----------:|
|  19.06.x   |   0.11.x    |
|  19.03.x   |   0.10.x    |
