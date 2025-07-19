# retico-multilingual-tts

A Retico module for producing TTS output in various languages.

## Contents

- `multilingual_tts.py`: Contains the `MultilingualTTSModule` class, which detects the language of a textual or audio IU and forwards it to the next module as an ISO 639-1 language code (under the `language` attribute).
- `test_network.py`: Contains a test script that demonstrates how to use the `MultilingualTTSModule` class in a Retico network. All you have to do is type text in the language of your choice in the terminal, and the network will produce the corresponding TTS output.

## Installation

- To automatically install this repository:

  ```bash
  pip install --upgrade pip setuptools wheel
  pip install git+https://github.com/retico-team/retico-multilingual-tts.git
  ```
- To install from source, you can clone the repository, install the dependencies, and test out the module by running the `test_network.py` script:

  ```bash
   git clone https://github.com/retico-team/retico-multilingual-tts
  ```

### Troubleshooting

The following errors can arise when using the recommended versions for Python and libraries:

- ```plain

  File ".venv/lib/python3.9/site-packages/bangla/__init__.py", line 139, in <module>
      def get_date(passed_date=None, passed_month=None, passed_year=None, ordinal: bool | None = False):
  TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
  ```

  - **Solution:** This error occurs because the `|` operator is not supported in Python 3.9. You can fix this by removing `bool | None` in the source code. Alternatively, you can upgrade to Python 3.10 or later, although this is not recommended at the moment due to potential compatibility issues with other Retico modules.
- ```plain
  File ".venv/lib/python3.9/site-packages/TTS/utils/io.py", line 54, in load_fsspec
      return torch.load(f, map_location=map_location, **kwargs)
  File ".venv/lib/python3.9/site-packages/torch/serialization.py", line 1470, in load
      raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
  _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.

  (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
  ```

  - **Solution:** You can fix this by adding `weights_only=False` in the source code of the module, or by downgrading to PyTorch<=2.5, although the latter solution is also not recommended at the moment due to potential compatibility issues with other Retico modules.

## Setup

> [!WARNING]
> It is recommended to use **Python 3.9.x** at the moment in order to avoid any compatibility issues with the rest of the Retico modules.

- If you haven't yet, create a virtual environment and activate it:

  - **Linux/MacOS**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
  - **Windows**:

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    If you encounter an error when activating the virtual environment, retry the above command after running the following line:

    ```powershell
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
- Install the dependencies:

  ```bash
  python3 -m pip install -r requirements.txt
  ```

## How to use

Simply subscribe this module to one that emits `TextIU`s. If those IUs have a `language` attribute, the module will automatically select the right voice when producing audio. You can try it out by running `test_network.py`.
