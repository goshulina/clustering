from sklearn.base import TransformerMixin
from pymystem3 import Mystem
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_multiple_whitespaces, strip_numeric, \
    strip_punctuation, strip_short
from copy import copy
import nltk


# from typing import Awaitable, Callable, TypeVar

# R = TypeVar("str")

class Preprocessor(TransformerMixin):
    def __init__(self, infinitive_mode='no'):
        self.stopwords = nltk.corpus.stopwords.words('russian')
        self.stem_lem_dict = {"lemm": Mystem(),
                              "stemm": SnowballStemmer("russian"),
                              "no": lambda x: x}
        self.infinitive_mode = infinitive_mode

    @staticmethod
    def is_iterable(X):
        """
        Cherck if X is iterable
        """
        try:
            _ = iter(X)
        except TypeError:
            raise Exception('X is not iterable!')

    @staticmethod
    def is_strings(X):
        """
        Cherck if all elements are strings
        """
        for n, x in enumerate(X):
            if type(x) != str:
                raise Exception(f'{n} element is not STRING type!')

    #######################
    ### Preproc methods ###
    #######################
    @staticmethod
    def del_meta(docs):
        """
        Delete tags, multiple whitespaces, numbers
        """
        custom_filters = [lambda i: i, strip_tags, strip_multiple_whitespaces, strip_numeric]
        data = [' '.join(preprocess_string(a, custom_filters)) for a in docs]
        return data

    def del_punctuation(self, docs):
        custom_filters = [lambda x: x, strip_punctuation]
        data = [' '.join(preprocess_string(i, custom_filters)) for i in docs]
        return data

    def lower_all(self, docs):
        """
        Lower all list elements
        """
        return [i.lower() for i in docs]

    def stop_words(self, docs):
        """
        Delete stop words
        """
        data = [' '.join([i for i in ii.split() if i not in self.stopwords]) for ii in docs]
        return data

    @staticmethod
    def cut_short_words(self, x, n=3):
        """
        delete all words that a strictly lower than n chars.
        """
        data = [strip_short(i, minsize=n) for i in x]
        return data

    def stem_lem(self, docs, mode='no'):
        """
        mode: ['lemm', 'stemm', 'no'], default is 'no'
        """
        stemmer = self.stem_lem_dict[mode]
        # TODO раскидать по разным класса
        # базовый класс класс: AbstractTextTransformer, наследники - MystemTransformer, SnowballTransformer
        if mode == 'lemm':
            data = ''.join(stemmer.lemmatize(
                '@@@'.join(docs)
            )).split('@@@')
        elif mode == 'stemm':
            data = [' '.join([stemmer.stem(ii) for ii in i.split()]) for i in docs]
        elif mode == 'no':
            data = stemmer(docs)
        return data

    ########################
    ### TRANSFORMER PIPE ###
    ########################
    def transform(self, X, y=None):
        """
        Iterate over preprocess algorithms over all data
        """
        self.is_iterable(X)
        self.is_strings(X)

        methods = (
            ('lower_all', self.lower_all),
            ('delete_tags_numbers', self.del_meta),
            ('delete_stop_words', self.stop_words),
            ('delete_punctuation', self.del_punctuation),
            ('infinitive', self.stem_lem),
        )
        args = {
            # убирать пунктуацию перед стеммером потому что там сплит по пробелам
            # Если None - то закомментить
            # 'lower_all': None,
            # 'delete_tags_numbers': None,
            # 'delete_stop_words': None,
            # 'delete_punctuation': None,
            'infinitive': {'mode': self.infinitive_mode}
        }

        x = copy(X)
        for method_name, method in methods:
            if method_name in args:  # TODO сделать по красивше
                method_args = args[method_name]
                x = method(x, **method_args)
            else:
                x = method(x)
        return x

    def fit(self, X, y=None):
        """
        redundant method
        """
        return self
