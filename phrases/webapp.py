
from flask import Flask
app = Flask(__name__)

from phrases.model.paraphraser import Paraphrases
net = Paraphrases()

from phrases.ui import *
