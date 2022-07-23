import os
from glob import glob

import ginza
import numpy as np
import spacy
import spacy.symbols as symbol
from Character import Character
from tqdm import tqdm
from utils import count_dict, extend_readlines

END = -1  # 文末（平叙）表す
END_QUESTION = -2  # 文末（疑問）を表す


class Converter:
    # キャラクタ性変換を行うクラス
    def __init__(self, config):
        # コンストラクタ

        # ファイル関連のディレクトリ
        self.serifs_directory_path = config.serifs_directory_path
        self.output_directory_path = os.path.join(self.serifs_directory_path, "result/")
        self.symbols_directory_path = config.symbols_directory_path
        self.base_sentences_filepath = config.base_sentences_filepath

        # ファイル読み込み先
        self.characters = None
        self.base_sentences = None

        # 構文解析器（ ginza ）
        self.syntax_analyst = spacy.load("ja_ginza")

        # 読み書きに使うエンコーディング
        self.ENCODING = config.encoding

        # 各品詞を表す集合
        self.noun_set = config.noun_set
        self.pos_set = config.pos_set
        self.dep_set = config.dep_set
        self.function_pos_set = config.function_pos_set

        # 記号集合
        self.symbols = None
        self.question_symbols = None

        self.read_files()

    def read_files(self):

        # ファイル読み込みを行う

        # セリフディレクトリ中の全てのファイル名を取得。
        serif_filepaths = glob(f"{self.serifs_directory_path}/*.csv")

        # 全ての全キャラクタを作成
        target_character_names = (
            os.path.splitext(os.path.basename(serif_filepath))[0]
            for serif_filepath in serif_filepaths
        )
        serif_pairs = [
            extend_readlines(filepath, self.ENCODING) for filepath in serif_filepaths
        ]
        self.characters = [
            Character(character_name, serifs)
            for character_name, serifs in zip(target_character_names, serif_pairs)
        ]

        # 基となる無キャラクタ文の読み込み。
        self.base_sentences = extend_readlines(
            self.base_sentences_filepath, self.ENCODING
        )

        # 記号の読み込み
        self.symbols = extend_readlines(
            os.path.join(self.symbols_directory_path, "symbols.txt"),
            encoding=self.ENCODING,
        )
        self.question_symbols = extend_readlines(
            os.path.join(self.symbols_directory_path, "question_symbols.txt"),
            encoding=self.ENCODING,
        )

    def get_pos_dep(self, before_span, after_span=None):
        # 係り元（ before_span ）と係り先（ after_span ）の文節から、主辞品詞（ pos ）と係り種別（ dep ）を取得する。
        # after_span is None のとき、文末とする。

        # 主辞品詞を取得する。
        pos = before_span.root.pos

        # もし主辞品詞が名詞の1つなら、「名詞」として統一する。
        if pos in self.noun_set:
            pos = symbol.NOUN

        # もし、文末なら
        if after_span is None:
            # もし何らかの疑問符が係り元に存在するなら、係種別は文末（疑問）
            if any([qs in before_span.text for qs in self.question_symbols]):
                dep = END_QUESTION
            else:
                dep = END
        else:  # 文末でないなら、係り種別は係り先の主辞品詞
            dep = after_span.root.pos

            # 係り種別が、名詞の1つなら、「名詞」で統一
            if dep in self.noun_set:
                dep = symbol.NOUN

        # 主辞品詞と係り種別が、想定される品詞になっていることを確認する。
        # 想定外の場合は、None を返し、失敗とする。
        if all([pos in self.pos_set, dep in self.dep_set]):
            return pos, dep
        else:
            return None

    def get_spans(self, text):
        sents = self.syntax_analyst(text)

        return ginza.bunsetu_spans(sents)

    def get_chara_nochara_spans(self, chara_nochara_pair):
        # chara_nochara_pair は（キャラクタ文1文、無キャラクタ1文）の tuple
        # キャラクタ文と無キャラクタ文の spans を返す。

        nochara_serif, chara_serif = chara_nochara_pair
        chara_serif_spans = self.get_spans(chara_serif)
        nochara_serif_spans = self.get_spans(nochara_serif)

        if len(chara_serif_spans) != len(nochara_serif_spans):
            return None
        else:
            return chara_serif_spans, nochara_serif_spans

    def get_symbols(self, text):
        # 文字列（ text ）に含まれている記号文字列を返す。

        removed_symbols = list()
        for symbol in self.symbols:
            if symbol in text:
                text = text.replace(symbol, "")
                removed_symbols.append(symbol)

        return "".join(removed_symbols)

    def split_span(self, span):
        # 文節（ span ）から、機能表現・内容語・記号の返す。

        function_token_list = list()
        root_token = span.root  # 主辞

        span_symbol = self.get_symbols(span.text)

        frag = False

        for token in span:
            remove_symbol_token = token.text.replace(span_symbol, "")
            if token == root_token:
                frag = True
            elif any([sym in token.text for sym in self.symbols]):
                continue
            elif token.pos in self.function_pos_set and frag:
                function_token_list.append(remove_symbol_token)
            elif token.pos not in self.function_pos_set:
                function_token_list = list()

        function_string = "".join(function_token_list)

        assert function_string is not None

        content_string = span.text.replace(function_string, "").replace(span_symbol, "")

        return function_string, content_string, span_symbol

    def add_chara_rule(self, chara_rule, key_tupl, function_string):
        # 規則を追加する。

        assert chara_rule is not None
        assert key_tupl is not None
        assert function_string is not None

        if key_tupl in chara_rule.keys():
            rule_set = chara_rule.get(key_tupl)
            rule_set.add(function_string)
            chara_rule[key_tupl] = rule_set

        else:
            chara_rule[key_tupl] = set([function_string])

        return chara_rule

    def make_rule_and_count_funtion_word(
        self,
        chara_rule_dict,
        chara_freq_dict,
        chara_span,
        nochara_span,
        chara_token=None,
        nochara_token=None,
    ):

        assert not (
            (chara_token is None and nochara_token is not None)
            or (chara_token is not None and nochara_token is None)
        )

        # pos_depの取得
        if chara_token is None and nochara_token is None:
            # 文末
            chara_pos_dep = self.get_pos_dep(chara_span)
            nochara_pos_dep = self.get_pos_dep(nochara_span)
        else:
            # それ以外
            chara_pos_dep = self.get_pos_dep(
                ginza.bunsetu_span(chara_token), chara_span
            )
            nochara_pos_dep = self.get_pos_dep(
                ginza.bunsetu_span(nochara_token), nochara_span
            )

        if chara_pos_dep != nochara_pos_dep:
            # pos_depが一致しないなら
            return chara_rule_dict, chara_freq_dict
        
        if any([chara_pos_dep is None,nochara_pos_dep is None]):
            return chara_rule_dict,chara_freq_dict

        if chara_token is None and nochara_token is None:
            # 文末
            chara_function_string, _, _ = self.split_span(chara_span)
            nochara_function_string, _, _ = self.split_span(nochara_span)
        else:
            # それ以外
            assert chara_token is not None
            assert nochara_token is not None

            chara_function_string, _, _ = self.split_span(
                ginza.bunsetu_span(chara_token)
            )
            nochara_function_string, _, _ = self.split_span(
                ginza.bunsetu_span(nochara_token)
            )

        chara_freq_dict = count_dict(chara_freq_dict, chara_function_string)

        rule_key_tupl = (nochara_function_string, chara_pos_dep)
        chara_rule_dict = self.add_chara_rule(
            chara_rule_dict, rule_key_tupl, chara_function_string
        )

        return chara_rule_dict, chara_freq_dict

    def make_rule_and_count_function_word_single_chara(self, character):
        # 一人のキャラクタ（ character ）に対して、規則の作成と頻度の数え上げを行う。

        chara_rule_dict = dict()
        chara_freq_dict = dict()

        chara_nochara_tupls = character.get_serif_pair_tuples()

        for chara_nochara_tupl in chara_nochara_tupls:
            if self.get_chara_nochara_spans(chara_nochara_tupl) is None:
                continue
            else:
                chara_spans, nochara_spans = self.get_chara_nochara_spans(
                    chara_nochara_tupl
                )
            
            for chara_span, nochara_span in zip(chara_spans, nochara_spans):
                if chara_span == chara_spans[-1]:
                    # 文末なら
                    (
                        chara_rule_dict,
                        chara_freq_dict,
                    ) = self.make_rule_and_count_funtion_word(
                        chara_rule_dict, chara_freq_dict, chara_span, nochara_span
                    )
                for chara_token, nochara_token in zip(
                    chara_span.lefts, nochara_span.lefts
                ):
                    (
                        chara_rule_dict,
                        chara_freq_dict,
                    ) = self.make_rule_and_count_funtion_word(
                        chara_rule_dict,
                        chara_freq_dict,
                        chara_span,
                        nochara_span,
                        chara_token,
                        nochara_token,
                    )

        return chara_rule_dict, chara_freq_dict

    def make_rule_and_count_function_word_all_chara(self):
        # 全てのキャラクタに対して、規則の作成と頻度の数え上げを行う。

        print("make rule and count freq ・・・")

        for chara in tqdm(self.characters):
            (
                chara_rule,
                chara_function_freq,
            ) = self.make_rule_and_count_function_word_single_chara(chara)
            chara.rule_dict = chara_rule
            chara.freq_dict = chara_function_freq

    def change_word(self, character, nochara_span, nochara_token=None):
        # 一人のキャラクタ（ character ）の特性を表すように、1つの文節を変換する。

        # pos_depの取得
        if nochara_token is None:
            # 文末
            nochara_pos_dep = self.get_pos_dep(nochara_span)
        else:
            # それ以外
            nochara_pos_dep = self.get_pos_dep(
                ginza.bunsetu_span(nochara_token), nochara_span
            )

        function_string, content_string, span_symbol = self.split_span(nochara_span)

        if (function_string,nochara_pos_dep) not in character.rule_dict.keys():
            return nochara_span.text

        new_function_string_candidate = character.rule_dict.get(
            (function_string, nochara_pos_dep)
        )
        new_function_string_probabilities = [
            character.freq_dict[function_word]
            for function_word in new_function_string_candidate
        ]

        new_function_string_probabilities = [
            freq / sum(new_function_string_probabilities)
            for freq in new_function_string_probabilities
        ]

        print(new_function_string_candidate)
        print(new_function_string_probabilities)
        new_function_string = np.random.choice(
            new_function_string_candidate,size = 1,p = new_function_string_probabilities
        )

        return content_string + new_function_string + span_symbol

    def characterize(self, character):

        # 一人のキャラクタ（ character ）に対して、キャラクタ性変換する。

        characterized_sentences = list()

        with open(
            os.path.join(self.output_directory_path, character.chara_name + ".csv"),
            "w",
            encoding=self.ENCODING,
        ) as output_fp:

            for base_sentence in self.base_sentences:
                indexes = list()
                words = list()
                sorted_words = list()
                sorted_indexes = list

                output_fp.write(f"{base_sentence},")

                spans = ginza.bunsetu_spans(self.syntax_analyst(base_sentence))
                for span in spans:
                    if span == spans[-1]:
                        words.append(self.change_word(character, span))
                        indexes.append(span.root.i)
                    for token in span.lefts:
                        words.append(self.change_word(character, span, token))
                        indexes.append(token.i)

                sorted_indexes = sorted(indexes)

                for index in sorted_indexes:
                    i = indexes.index(index)
                    sorted_words.append(words[i])

                characterized_sentences = "".join(sorted_words)
                output_fp.write(f"{characterized_sentences}\n")

    def characterize_all(self):
        # 全てのキャラクタに対して、キャラクタ性変換する。

        print("characterizing・・・")

        for character in tqdm(self.characters):
            self.characterize(character)

    def __call__(self):
        # 一連の処理

        self.make_rule_and_count_function_word_all_chara()
        self.characterize_all()
