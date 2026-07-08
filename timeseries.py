import re

regex = re.compile(r"[^a-zA-Z0-9]", re.MULTILINE)


def sanitize_string(string):
    return regex.sub("", string)
