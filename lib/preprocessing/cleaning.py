import re
from emoji import demojize
import emot


def basic_clean(text):
    text = re.sub(r"@[a-zA-Z0-9äöüÄÖÜß]+", " ", text)
    text = re.sub(r"https?://[^ ]+", " ", text)
    text = re.sub(r"www.[^ ]+", " ", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß]", " ", text)
    text = re.sub(r" +", " ", text)
    return text


def clean(text):
    text = text.lower()
    text = re.sub(r"@[^\s]+", " #user_handle# ", text)
    text = re.sub(r"https?://[^ ]+", " #url# ", text)
    text = re.sub(r"www.[^ ]+", " #url# ", text)
    text = demojize(text, delimiters=(" #", "# "))
    while True:
        emoticons = emot.emoticons(text)
        if not isinstance(emoticons, dict) or not emoticons["flag"]:
            break
        text = (
            text[: emoticons["location"][0][0]]
            + " #"
            + "_".join(emoticons["mean"][0].lower().split(" "))
            + "# "
            + text[emoticons["location"][0][1] :]
        )
    text = re.sub(r"[^a-zA-Zäöüß_#-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
