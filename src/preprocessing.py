import os
import re

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

# Download NLTK dependencies silently if not found
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

###################################################################################################################
# Regular Expressions
###################################################################################################################
re_html = "<.*?>"
re_dates = "[A-Z,a-z]{3}\s[0-9]{2}/[0-9]{4}"
re_months = "january|february|march|april|may|june|july|august|september|october|november|december"
re_short_months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
re_days = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
re_romans_low = "\b(m{0,3})(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})\b"
re_noncharsandnums = "[^A-Za-z0-9ñÑáéíóúÁÉÍÓÚ\s\,;:\/]+"
re_onlychars = "[^A-Za-záéíóúÁÉÍÓÚ\.\s]"
re_newline = "[\\n\\r]+"
re_p = '<p [a-z,0-9\n"\,\&-;:\s\=\%#]*>'
re_span = '<span [a-z,0-9"\,\&-;:\s\=\%#]*>'
re_td = '<td [a-z,0-9"\,\&-;:\s\=\%#]*>'
re_td2 = '<td [a-z,0-9"\,\&-;:\s\=\%#]*;'
re_tr = '<tr [a-z,0-9"\,\&-;:\s\=]*>'
re_url = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
re_file_extension = "\.[a-zA-Z0-9]{3}"
re_empty_sentences = "\.[\s]+\."
re_spaces_before_dot = "[\s]+\."
re_short_sentences = "\.[\s]+[A-Z,a-z]+[\s]*\."
re_useless_words = "https|flwsk"
re_table_of_contents = r"[Tt]able\s*of\s*[Cc]ontents"
re_10k = r"10[\-]?[Kk]"
re_extra_spaces = r"\s{2,}"


###################################################################################################################
# Base functions
###################################################################################################################
def remove_matches(text, rx, replace=""):
    clean = re.compile(rx, re.IGNORECASE)
    return re.sub(clean, replace, str(text))


def clean_text(text):
    text = remove_matches(text, re_html)
    text = remove_matches(text, re_dates)
    text = remove_matches(text, re_months)
    text = remove_matches(text, re_short_months)
    text = remove_matches(text, re_days)
    text = remove_matches(text, re_romans_low)
    text = remove_matches(text, re_onlychars)
    text = remove_matches(text, re_newline)
    text = remove_matches(text, re_p)
    text = remove_matches(text, re_span)
    text = remove_matches(text, re_td)
    text = remove_matches(text, re_td2)
    text = remove_matches(text, re_tr)
    text = remove_matches(text, re_url)
    text = remove_matches(text, re_file_extension)
    text = remove_matches(text, re_useless_words)
    text = remove_matches(text, re_empty_sentences, replace=".")
    text = remove_matches(text, re_spaces_before_dot, replace=".")
    text = remove_matches(text, re_short_sentences, replace=".")
    text = remove_matches(text, re_table_of_contents, replace="")
    text = remove_matches(text, re_10k, replace="")
    text = remove_matches(text, re_extra_spaces, replace=" ")
    return text.strip()


def split_into_sentences(text):
    return sent_tokenize(text)
