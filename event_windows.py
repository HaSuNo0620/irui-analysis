#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared preprocessing for cross-species relationship-event analysis.

The retrieval rules define *where to observe*, not a typology. A qualifying local event
must explicitly contain (1) a nonhuman marker, (2) a human-side marker, and (3) a
romantic/marital relationship marker within the same or adjacent sentence block.
Downstream clustering remains unsupervised.
"""

import re
from janome.tokenizer import Tokenizer

NONHUMAN_TERMS = [
    "人外","異類","異種族","妖怪","あやかし","もののけ","鬼","妖狐","狐","狸","蛇","龍神","竜神","龍","竜",
    "神","神様","女神","精霊","妖精","天狗","雪女","吸血鬼","ヴァンパイア","悪魔","魔族","獣人","亜人",
    "エルフ","人魚","ラミア","ハーピー","フェンリル","アンデッド","ゾンビ","幽霊","死神","宇宙人","異星人",
    "アンドロイド","ロボット","人工生命","怪物","化け物","モンスター"
]

# Human-side evidence is deliberately explicit. Single-character 男/女 are excluded
# because they create many accidental matches inside words such as 女神.
HUMAN_TERMS = [
    "人間","人族","人間族","普通の人間","人間の少年","人間の少女","人間の男","人間の女",
    "少年","少女","青年","若者","男性","女性","男子","女子","主人公"
]

RELATION_TERMS = [
    "恋","恋愛","好き","大好き","惚れ","愛する","愛して","愛され","惹かれ","想い",
    "恋に落ち","恋人","恋仲","結婚","婚姻","婚約","求婚","伴侶","つがい","夫婦",
    "嫁","嫁入り","嫁ぐ","嫁にする","花嫁","妻","夫","娶る","新婚","契約婚",
    "結婚させら","婚約させら","嫁がされ","妻にされ","夫にされ","花嫁にされ",
    "生贄","生け贄"
]

_NONHUMAN_RE = re.compile("|".join(map(re.escape, sorted(NONHUMAN_TERMS, key=len, reverse=True))))
_HUMAN_RE = re.compile("|".join(map(re.escape, sorted(HUMAN_TERMS, key=len, reverse=True))))
_RELATION_RE = re.compile("|".join(map(re.escape, sorted(RELATION_TERMS, key=len, reverse=True))))
_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?|\n")
_TOKENIZER = None


def lexical_hits(text):
    return set(_NONHUMAN_RE.findall(text)), set(_RELATION_RE.findall(text))


def _spans(pattern, text):
    return [(m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)]


def _non_overlapping_human_hits(text):
    nh_spans = [(a, b) for a, b, _ in _spans(_NONHUMAN_RE, text)]
    hits = []
    for a, b, term in _spans(_HUMAN_RE, text):
        if any(not (b <= x0 or a >= x1) for x0, x1 in nh_spans):
            continue
        hits.append((a, b, term))
    return hits


def _sentence_spans(text):
    spans = []
    for m in _SENTENCE_RE.finditer(text):
        s = m.group(0)
        if not s or s == "\n":
            continue
        spans.append((m.start(), m.end(), s))
    return spans


def relation_evidence_blocks(text, max_sentences=2):
    """Find blocks that explicitly mention human, nonhuman, and relationship evidence.

    Blocks comprise one sentence or two adjacent sentences. Human hits overlapping a
    nonhuman expression (e.g. 女 inside 女神) are ignored.
    Returns (start, end, block, nonhuman_hits, human_hits, relation_hits).
    """
    sentences = _sentence_spans(text)
    out = []
    seen = set()
    for i in range(len(sentences)):
        for n in range(1, max_sentences + 1):
            j = i + n
            if j > len(sentences):
                break
            st = sentences[i][0]
            en = sentences[j - 1][1]
            block = text[st:en]
            nh = _spans(_NONHUMAN_RE, block)
            human = _non_overlapping_human_hits(block)
            rel = _spans(_RELATION_RE, block)
            if not (nh and human and rel):
                continue
            key = (st, en)
            if key in seen:
                continue
            seen.add(key)
            out.append((st, en, block,
                        sorted({x[2] for x in nh}),
                        sorted({x[2] for x in human}),
                        sorted({x[2] for x in rel})))
    return out


def has_cross_species_relation_evidence(text, max_sentences=2):
    return bool(relation_evidence_blocks(text, max_sentences=max_sentences))


def extract_relation_event_windows(text, proximity=500, window_chars=600, max_windows=25, max_sentences=2):
    """Return windows centered on explicit human-nonhuman relationship evidence.

    ``proximity`` is retained for API compatibility but no longer drives qualification;
    sentence-level relational evidence is stricter. Overlapping windows are merged.
    """
    events = relation_evidence_blocks(text, max_sentences=max_sentences)
    if not events:
        return []

    half = max(int(window_chars) // 2, 120)
    intervals = []
    for st0, en0, *_ in events:
        center = (st0 + en0) // 2
        st = max(0, center - half)
        en = min(len(text), center + half)
        intervals.append((st, en))

    intervals.sort()
    merged = []
    for st, en in intervals:
        if not merged or st > merged[-1][1]:
            merged.append([st, en])
        else:
            merged[-1][1] = max(merged[-1][1], en)

    windows = [(st, en, text[st:en]) for st, en in merged if en - st >= 180]
    if max_windows and len(windows) > max_windows:
        idx = [round(i * (len(windows) - 1) / (max_windows - 1)) for i in range(max_windows)] if max_windows > 1 else [0]
        seen = set()
        windows = [windows[i] for i in idx if not (i in seen or seen.add(i))]
    return windows


def anonymize_proper_nouns(text):
    """Replace Janome proper nouns while preserving research-target terms."""
    global _TOKENIZER
    protected = {}
    counter = 0

    def protect_match(m):
        nonlocal counter
        key = f"ZXQPROTECT{counter}QXZ"
        protected[key] = m.group(0)
        counter += 1
        return key

    tmp = _NONHUMAN_RE.sub(protect_match, text)
    tmp = _HUMAN_RE.sub(protect_match, tmp)
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
