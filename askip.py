# -*- coding: UTF-8 -*-

import re
import wikipedia
from urllib.parse import urlparse

from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class AskipModel:
    def __init__(self, wikipedia_url):
        self._vectorizer = None

        loc = urlparse(wikipedia_url)
        lang = loc.netloc.split(".", 1)[0]
        name = loc.path.split("/")[-1]
        self.set_model(name, lang=lang)

    def set_model(self, name, lang="en"):
        wikipedia.set_lang(lang)

        # https://github.com/goldsmith/Wikipedia/issues/124
        page = wikipedia.page(name, auto_suggest=False)

        texts = []
        titles = 1
        for p in re.split(r"\n+", page.content):
            if p[0] == "=" and p[-1] == "=":
                titles += 1
                continue  # title

            texts.append(p)

        stop_words = "english" if lang == "en" else get_stop_words(lang)

        vectorizer = TfidfVectorizer(
                strip_accents="unicode",
                stop_words=stop_words)
        X = vectorizer.fit_transform(texts)

        n_clusters = max(titles, 4)  # arbitrary
        km = KMeans(n_clusters=n_clusters).fit(X)

        self._vectorizer = vectorizer
        self._km = km
        self._texts = texts

    def ask(self, q):
        q = re.sub(r"[?!]+$", "", q)

        cluster = self._km.predict(self._vectorizer.transform([q]))[0]

        for i, cl in enumerate(self._km.labels_):
            if cl == cluster:
                print(self._texts[i])
                break
