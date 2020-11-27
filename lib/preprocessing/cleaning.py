import re


def basic_clean(text):
    text = re.sub(r"@[a-zA-Z0-9äöüÄÖÜß]+", " ", text)
    text = re.sub(r"https?://[^ ]+", " ", text)
    text = re.sub(r"www.[^ ]+", " ", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß]", " ", text)
    text = re.sub(" +", " ", text)
    return text
