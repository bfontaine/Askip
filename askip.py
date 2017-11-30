# -*- coding: UTF-8 -*-

import re
import argparse
import wikipedia
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer

LANGUAGES = {
    "en": "english",
    "fr": "french",
    "it": "italian",
    "es": "spanish",
    "pt": "portuguese",
}

def get_stop_words(lang):
    return stopwords.words(LANGUAGES.get(lang, lang))

def mk_tokenizer(lang):
    stemmer = SnowballStemmer(LANGUAGES.get(lang, lang))
    def tokenize(text):
        tokens = word_tokenize(text)
        return [stemmer.stem(tok) for tok in tokens]
    return tokenize

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
        for sent in sent_tokenize(page.content):
            for p in re.split(r"\n+", sent):
                if p[0] == "=" and p[-1] == "=":
                    titles += 1
                    continue  # title

                if len(p) < 30:
                    continue

                if "»" in p and not "«" in p:
                    continue

                texts.append(p)

        stop_words = "english" if lang == "en" else get_stop_words(lang)

        vectorizer = TfidfVectorizer(
                tokenizer=mk_tokenizer(lang),
                max_df=0.97,
                min_df=0.01,
                strip_accents="unicode",
                stop_words=stop_words)
        X = vectorizer.fit_transform(texts)

        n_clusters = max(titles, 16, len(texts)//10)  # arbitrary

        km = KMeans(n_clusters=n_clusters).fit(X.todense())

        self._vectorizer = vectorizer
        self._km = km
        self._texts = texts

    def ask(self, q):
        q = re.sub(r"[?!]+$", "", q)

        q = re.sub(r"^(?:what is|what's|quel est|quelle est|que) +", "",
                q, re.IGNORECASE)

        cluster = self._km.predict(self._vectorizer.transform([q]))[0]

        n = 0

        for i, cl in enumerate(self._km.labels_):
            if cl == cluster:
                n += 1
                print(self._texts[i], end=" ")

                if n > 4:
                    break

        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wikipedia_url")
    args = parser.parse_args()

    m = AskipModel(args.wikipedia_url)
    while True:
        try:
            q = input("--> ")
        except EOFError:
            break

        if not q or q in {"bye", "exit", "quit"}:
            break

        m.ask(q)

if __name__ == "__main__":
    main()
