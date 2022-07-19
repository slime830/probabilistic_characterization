import spacy.symbols as symbol
from Converter import END, END_QUESTION, Converter


class Config:
    def __init__(self):
        self.serifs_directory_path = "./serifs"
        self.symbols_directory_path = "./symbols"
        self.base_sentences_filepath = "./base_sentences.txt"

        self.encoding = "utf-8"

        self.noun_set = {symbol.NOUN, symbol.PRON, symbol.PROPN}
        self.pos_set = {symbol.NOUN, symbol.VERB, symbol.ADJ}
        self.dep_set = {symbol.NOUN, symbol.VERB, END, END_QUESTION}
        self.function_pos_set = {
            symbol.AUX,
            symbol.CONJ,
            symbol.SCONJ,
            symbol.ADP,
            symbol.PART,
        }


def main():
    converter = Converter(Config())
    converter()


if __name__ == "__main__":
    main()
