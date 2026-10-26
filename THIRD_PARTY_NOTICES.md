# Third-party notices

## Convolutional-KANs (MIT License)

Source: https://github.com/AntonioTepsich/Convolutional-KANs

Copyright (c) 2024 Alexander Bodner, Antonio Tepsich, Jack Spolski and Santiago Pourteau.
The full license text is in [`kan_convolutional/LICENSE`](kan_convolutional/LICENSE).

- `kan_convolutional/KANLinear.py` is the file of upstream commit `c84523969edd48afb0ee7e16cd9740b566a650b9`, without changes. The upstream README credits [efficient-kan](https://github.com/Blealtan/efficient-kan) for its KAN implementation.
- `kan_convolutional/kan_conv.py` is the file of upstream commit `19266d73cc232711615d5baf5093222106ef7d7e`, without changes.
- The modules in `architectures/` that define CNN (Small), CNN (Medium), CNN (Big) and Conv & KAN (`SimpleModels.py`, `ConvNet.py`, `conv_and_kan.py`) are adapted from the `architectures_28x28` modules of the same project.

## FastKAN (Apache License 2.0)

Source: https://github.com/ZiyaoLi/fast-kan

Copyright 2024 Li, Ziyao. FastKAN is not included in this repository. It is installed as a dependency (`requirements.txt`) and imported as `from fastkan import FastKAN`.
