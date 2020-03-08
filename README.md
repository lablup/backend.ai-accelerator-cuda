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

|       Backend.AI Agent       |    CUDA Plugin   |
|:----------------------------:|:----------------:|
|  19.12.x                     |  0.13.x          |
|  19.06.17 ~                  |  0.13.x          |
|  19.06.x, 19.09.0 ~ 19.09.16 |  0.11.x, 0.12.x  |
|  19.03.x                     |  0.10.x          |
