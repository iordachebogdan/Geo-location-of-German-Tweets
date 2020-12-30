import re
from emoji import demojize


def basic_clean(text):
    text = re.sub(r"@[a-zA-Z0-9äöüÄÖÜß]+", " ", text)
    text = re.sub(r"https?://[^ ]+", " ", text)
    text = re.sub(r"www.[^ ]+", " ", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß]", " ", text)
    text = re.sub(r" +", " ", text)
    return text


def clean(text):
    text = text.lower()
    text = demojize(text, delimiters=(" :", ": "))
    text = re.sub(r"@[^\s]+", " :user_handle: ", text)
    text = re.sub(r"https?://[^ ]+", " :url: ", text)
    text = re.sub(r"www.[^ ]+", " :url: ", text)
    text = re.sub(r"[^a-zA-Zäöüß:_#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
