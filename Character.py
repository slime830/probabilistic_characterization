# 各キャラクタを表すクラス
class Character:
    def __init__(self, chara_name, serif_pairs):
        self.chara_name = chara_name
        self.rule_dict = dict()
        self.freq_dict = dict()
        self.chara_serifs = list()
        self.nochara_serifs = list()

        for serif_pair in serif_pairs:
            self.nochara_serifs.append(serif_pair.split(",")[0])
            self.chara_serifs.append(serif_pair.split(",")[1])

    def get_serif_pair_tuples(self):
        return [
            (nochara_serif, chara_serif)
            for nochara_serif, chara_serif in zip(
                self.nochara_serifs, self.chara_serifs
            )
        ]
