#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared preprocessing for unsupervised relation-event analysis.

The retrieval lexicon defines *where to observe*, not a typology. We extract local
windows in which a nonhuman term and a relationship term occur near each other,
then anonymize proper nouns before embedding. The target lexicon itself is preserved.
"""

import re
from janome.tokenizer import Tokenizer

NONHUMAN_TERMS = [
    "人外","異類","異種族","妖怪","あやかし","もののけ","鬼","妖狐","狐","狸","蛇","龍神","竜神","龍","竜",
    "神","神様","女神","精霊","妖精","天狗","雪女","吸血鬼","ヴァンパイア","悪魔","魔族","獣人","亜人",
    "エルフ","人魚","ラミア","ハーピー","フェンリル","アンデッド","ゾンビ","幽霊","死神","宇宙人","異星人",
    "アンドロイド","ロボット","人工生命","怪物","化け物","モンスター"
]
RELATION_TERMS = [
    "恋","恋愛","好き","愛する","愛され","惹かれ","想い","結婚","婚姻","嫁","嫁入り","花嫁","妻","夫","夫婦",
    "婚約","求婚","伴侶","つがい","恋人","恋仲","同居","暮らす","一緒に暮ら","新婚","溺愛","契約婚","生贄","生け贄"
]

# Longest first prevents short substrings such as 神 from masking 神様.
_NONHUMAN_RE = re.compile("|".join(map(re.escape, sorted(NONHUMAN_TERMS, key=len, reverse=True))))
_RELATION_RE = re.compile("|".join(map(re.escape, sorted(RELATION_TERMS, key=len, reverse=True))))
_TOKENIZER = None


def lexical_hits(text):
    return set(_NONHUMAN_RE.findall(text)), set(_RELATION_RE.findall(text))


def _spans(pattern, text):
    return [(m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)]


def extract_relation_event_windows(text, proximity=500, window_chars=1200, max_windows=25):
    """Return (start, end, raw_window) around close nonhuman/relation co-occurrences.

    Overlapping windows are merged so repeated hits in one scene do not dominate.
    If there are more windows than max_windows, they are sampled approximately
    uniformly over narrative position.
    """
    nh = _spans(_NONHUMAN_RE, text)
    rel = _spans(_RELATION_RE, text)
    if not nh or not rel:
        return []

    half = max(int(window_chars) // 2, 100)
    intervals = []
    for a0, a1, _ in nh:
        ac = (a0 + a1) // 2
        for b0, b1, _ in rel:
            bc = (b0 + b1) // 2
            if abs(ac - bc) <= proximity:
                center = (ac + bc) // 2
                st = max(0, center - half)
                en = min(len(text), center + half)
                intervals.append((st, en))

    if not intervals:
        return []
    intervals.sort()
    merged = []
    for st, en in intervals:
        if not merged or st > merged[-1][1]:
            merged.append([st, en])
        else:
            merged[-1][1] = max(merged[-1][1], en)

    windows = [(st, en, text[st:en]) for st, en in merged if en - st >= 200]
    if max_windows and len(windows) > max_windows:
        # Uniform narrative coverage without importing numpy into this small module.
        idx = [round(i * (len(windows) - 1) / (max_windows - 1)) for i in range(max_windows)] if max_windows > 1 else [0]
        seen = set()
        windows = [windows[i] for i in idx if not (i in seen or seen.add(i))]
    return windows


def anonymize_proper_nouns(text):
    """Replace Janome proper nouns while preserving target nonhuman/relation terms."""
    global _TOKENIZER
    protected = {}
    counter = 0

    # Protect research-target terms from being anonymized even if Janome tags them as proper nouns.
    def protect_match(m):
        nonlocal counter
        key = f"ZXQPROTECT{counter}QXZ"
        protected[key] = m.group(0)
        counter += 1
        return key

    tmp = _NONHUMAN_RE.sub(protect_match, text)
    tmp = _RELATION_RE.sub(protect_match, tmp)
    if _TOKENIZER is None:
        _TOKENIZER = Tokenizer()

    parts = []
    last_was_proper = False
    for tok in _TOKENIZER.tokenize(tmp):
        pos = tok.part_of_speech.split(",")
        is_proper = len(pos) > 1 and pos[0] == "名詞" and pos[1] == "固有名詞"
        if is_proper:
            if not last_was_proper:
                parts.append("<PROPER>")
            last_was_proper = True
        else:
            parts.append(tok.surface)
            last_was_proper = False
    out = "".join(parts)
    for key, value in protected.items():
        out = out.replace(key, value)
    return out
