# Scan op_type for ONNX Model

1.  Clone the source code.
    ```bash
    cd ~
    git clone https://github.com/nexgus/opset.git
    ```
1.  Create the virtual environment and activate it.
    ```bash
    cd opset
    python3.10 -m venv .
    source bin/activate
    ```
1.  Update `pip`.
    ```bash
    pip install -U pip
    ```
1.  Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
1.  Validate the model.
    ```bash
    python main.py <path_to_model> validate
    ```
