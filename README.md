# NativaGPT

---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />

Welcome to NativaGPT, a Python backend toll allowing task execution through integration of Large Language Models (LLMs) and Text-to-Speech (TTS) and Speech-to-Text (STT) models.

Author: Pedro Afonso Dias

### <a name="Description"></a>1. Index

---

* [Description](#Description)
* [Prerequisites](#Prerequisites)
* [Installation](#Installation)
* [Usage](#Usage)

### <a name="Description"></a>2. Description

---

Nativa is an AI-Framework designed to improve the interaction between the user and the cell it is implemented. It was firstly designed
to be used in robotic cells, in order to give them some sort of reasoning, enhancing its
capabilities and collaborative functions.

### <a name="Prerequisites"></a>3. Prerequisites

---

This package has been tested on:

- Operating System: Ubuntu 24.04
- Python Version: 3.10+

**Dependencies**:

It is possible to install all required dependencies through the requirements.txt file located in the root repository:

```bash
pip install -r requirements.txt
```

### <a name="Installation"></a>4. Installation

---

1. Clone the [WebGPTHandler](/) repository to a specified directory:

```bash
git clone https://github.com/pascd/NativaGPT.git
```

2. Create a python environment

```bash
sudo apt install -y python3-dev python3-venv portaudio19-dev
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### <a name="Usage"></a>5. Usage

---

To start the conversation with Nativa, just need to execute the command:

```bash
cd NativaGPT/scripts
python3 start_nativa.py
```



-----------------------------------------------------------------------------------------------------------------
<br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.