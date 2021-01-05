import re
import emot


def basic_clean(text):
    # remove usernames
    text = re.sub(r"@[a-zA-Z0-9äöüÄÖÜß]+", " ", text)
    # remove urls
    text = re.sub(r"https?://[^ ]+", " ", text)
    text = re.sub(r"www.[^ ]+", " ", text)
    # remove non-letter symbols
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß]", " ", text)
    # contract spaces
    text = re.sub(r" +", " ", text)
    return text


def clean(text):
    # lowercase
    text = text.lower()
    # replace usernames
    text = re.sub(r"@[^\s]+", " #user_handle# ", text)
    # replace urls
    text = re.sub(r"https?://[^ ]+", " #url# ", text)
    text = re.sub(r"www.[^ ]+", " #url# ", text)
    # replace emoji
    while True:
        emoji = emot.emoji(text)
        if not isinstance(emoji, dict) or not emoji["flag"]:
            break
        mean = emoji["mean"][0][1:-1].lower()
        mean = re.sub(r"[^a-z]", " ", mean)
        mean = re.sub(r"\s+", " ", mean)
        mean = mean.strip()
        mean = "_".join(mean.split(" "))
        text = (
            text[: emoji["location"][0][0]]
            + " #"
            + mean
            + "# "
            + text[emoji["location"][0][1] + 1 :]
        )
    # replace emoticons
    while True:
        emoticons = emot.emoticons(text)
        if not isinstance(emoticons, dict) or not emoticons["flag"]:
            break
        mean = emoticons["mean"][0].lower()
        mean = re.sub(r"[^a-z]", " ", mean)
        mean = re.sub(r"\s+", " ", mean)
        mean = mean.strip()
        mean = "_".join(mean.split(" "))
        text = (
            text[: emoticons["location"][0][0]]
            + " #"
            + mean
            + "# "
            + text[emoticons["location"][0][1] :]
        )
    # remove other symbols
    text = re.sub(r"[^a-zA-Zäöüß_#]", " ", text)
    # contract spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
