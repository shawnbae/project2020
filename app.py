from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime
import pandas as pd

app = Flask(__name__)

if __name__ == "__main__":
    app.run(host='0.0.0.0')