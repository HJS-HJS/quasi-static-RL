# quasi-static-RL

## 1. Setup

1. **Install the code**
   ```bash
   git clone https://github.com/HJS-HJS/quasi-static-RL.git
   ```

2. **Install the submodule**
   ```bash
   git submodule init
   git submodule update
   ```

3. **Install the requirements**
    ```bash
    git submodule init
    git submodule update
    ```

    - When installing in the server, you need additional work for torch
        ```bash
        pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
        ```
    
    - Check torch is installed correctly
        ```bash
        python3 src/utils/check_cuda.py
        ```