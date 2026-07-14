# test-project

An attempt at creating a project (in this case a chess engine) where all the building is done by AI, I only do the designing, orchestration and review of what the coding agents build

To play against the engine, import the `chess.exe` file into a software such as Cute Chess or en-croissant, and play against it. The engine follows UCI protocol so any such software would work. This is for Windows based software, for Linux `chess` itself might work. Also, if you want the engine to use NNUE evaluation please ensure the `nnue.bin` file is also present in the same path as the binary

If you want to compile your own binary, compile all the `.cpp` files present in `src/`, something like `src/*.cpp` in the compilation command is fine. For training the NNUE weights use `nnue_train.py` or `nnue_train_gpu.py` with appropriate arguments
