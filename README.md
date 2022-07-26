# 「文節機能部の確率的書き換えによる言語表現のキャラクタ性変換」

## 概要
[宮崎千明，平野徹，東中竜一郎，牧野俊朗，松尾義博，佐藤理史： 文節機能部の確率的書
き換えによる言語表現のキャラクタ性変換，人工知能学会論文誌，Vol. 31, No. 1, pp. 1–9
(2016).](https://www.jstage.jst.go.jp/article/tjsai/advpub/0/advpub_DSF-515/_article/-char/ja/)の再現。

勝手に再現し、勝手に公開しています。ないと思いますが、しかるべき所から怒られたら、非公開にします。

## 開発環境

- OS: Ubuntu / Windows10 (Windows Subsystem for Linux)
- 言語: Python 3.8.4

## 必要ライブラリ

- numpy == 1.22.2  
- pandas == 1.1.3  
- spacy == 3.2.1  
- ginza == 5.1.0  
- ja-ginza == 5.1.0  

以下コマンドでインストール

```sh
pip install -r requirements.txt
```

## 実行方法

0. Python のセットアップ、必要ライブラリ等のインストール

1. 各キャラクタごとに、以下のような（無キャラクタ文 , キャラクタ文）形式の csv ファイルを作成し、`/serifs`に配置する。

```cs
あなたはバカですね。,おまえアホやな。
このラーメンしょっぱいですね。,このラーメン辛いなぁ。
```

2. 変換元の無キャラクタ文を、`/base_sentences.txt`として保存する。

```
これはとても美味しそうです。
あしたの天気は晴れだと思います。
```

3. `main.py`を実行する。実行後、`/serifs/result`に変換結果が保存されている

```
これはえらい美味そうやな。
あしたの天気は晴れやと思うで。
```
