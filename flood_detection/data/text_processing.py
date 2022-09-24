import re
import string
from typing import Union

import spacy


class TextProcessing:
    def __init__(self):
        self.__sp = spacy.load("en_core_web_sm")

    def clean_text(self, text_list: Union[list, str]) -> list[str]:

        if isinstance(text_list, str):
            text_list = [text_list]

        for id, text in enumerate(text_list):
            stopwords = self.__sp.Defaults.stop_words

            # Remove URLs/Mentions/Hashtags/new lines/numbers
            text = re.sub(
                "((www.[^s]+)"
                "|(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\."
                "[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*))"
                "|(@[a-zA-Z0-9]*)"
                "|([0-9]+)"
                "|\n)",
                " ",
                text,
            )
            # Remove stopwords
            text = " ".join(
                [word for word in str(text).split() if word not in stopwords]
            )
            # Remove punctuation
            text = "".join([char for char in text if char not in string.punctuation])
            # Remove Emojis
            emoji_pattern: re.Pattern[str] = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                # flags (iOS)
                "\U0001F1E0-\U0001F1FF" "]+",
                flags=re.UNICODE,
            )
            text = emoji_pattern.sub(r"", text)
            # Lower case
            text = text.lower()
            text_list[id] = text

        return text_list
