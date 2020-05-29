# chatbot-prz

chatbot-prz is simple chatbot that can answer questions related to formatting of engineering thesis in accordance with the WEiI PRz [template](https://weii.prz.edu.pl/studenci/praca-dyplomowa/jak-napisac-prace-dyplomowa). You can access chatbot at the following URL:
* [https://chatbot-prz.herokuapp.com](https://chatbot-prz.herokuapp.com)

## Installation

Clone this repo to your local machine using [https://github.com/kryha5555/chatbot-prz](https://github.com/kryha5555/chatbot-prz.git) and install requirements with:

```shell
$ pip install -r requirements.txt
```

## Usage

To run the application you can either use the `flask` command or python’s `-m` switch with Flask after exporting the `FLASK_APP` environment variable:

```shell
$ export FLASK_APP=chatbot.py
$ flask run
$ # or
$ python -m flask run
```

On Windows you need to use `set` instead of `export`.

## Training

To use your own intents for bot answers, you can edit them in `training/intents.json` file. You can retrain bot by accessing `training` directory and using:

```shell
$ python train_chatbot.py
```
