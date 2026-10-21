"""Byte-level BPE training and encoding, with GPT-2 pre-tokenization."""

import heapq
import json
from collections import Counter, defaultdict
from functools import lru_cache

import regex

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN = regex.compile(PATTERN)


def _special_pattern(tokens):
    return (
        regex.compile("|".join(regex.escape(t) for t in sorted(tokens, key=lambda t: (-len(t), t)))) if tokens else None
    )


def _pieces(text, special_pattern):
    if special_pattern is None:
        yield False, text
        return
    start = 0
    for match in special_pattern.finditer(text):
        yield False, text[start : match.start()]
        yield True, match.group()
        start = match.end()
    yield False, text[start:]


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if any(not t for t in self.special_tokens):
            raise ValueError("Special tokens must be nonempty")
        self.byte_ids = {value: key for key, value in self.vocab.items()}
        for token in self.special_tokens:
            value = token.encode("utf-8")
            if value not in self.byte_ids:
                index = max(self.vocab, default=-1) + 1
                self.vocab[index] = value
                self.byte_ids[value] = index
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_pattern = _special_pattern(self.special_tokens)
        # Bounded and instance-local so streaming does not retain an entire corpus.
        self._encode_piece = lru_cache(maxsize=8192)(self._encode_piece)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as f:
            raw = json.load(f)
        # GPT-2's reversible printable representation of individual bytes.
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars = values[:]
        extra = 0
        for byte in range(256):
            if byte not in values:
                values.append(byte)
                chars.append(256 + extra)
                extra += 1
        decoder = {chr(char): byte for byte, char in zip(values, chars)}

        def decode(token):
            return bytes(decoder[c] for c in token)

        vocab = {index: decode(token) for token, index in raw.items()}
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(decode(t) for t in line.split()) for line in f if line.strip() and not line.startswith("#version:")]
        return cls(vocab, merges, special_tokens)

    def save(self, path):
        """Save a lossless JSON representation; byte tokens need no text codec."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                dict(
                    vocab={str(i): list(b) for i, b in self.vocab.items()},
                    merges=[[list(a), list(b)] for a, b in self.merges],
                    special_tokens=self.special_tokens,
                ),
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            {int(i): bytes(b) for i, b in data["vocab"].items()},
            [(bytes(a), bytes(b)) for a, b in data["merges"]],
            data["special_tokens"],
        )

    def _encode_piece(self, piece):
        tokens = [bytes([byte]) for byte in piece.encode("utf-8")]
        while len(tokens) > 1:
            pairs = zip(tokens, tokens[1:])
            best = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if best not in self.ranks:
                break
            merged = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tuple(self.byte_ids[t] for t in tokens)

    def encode(self, text):
        result = []
        for special, piece in _pieces(text, self.special_pattern):
            if special:
                result.append(self.byte_ids[piece.encode("utf-8")])
            else:
                for match in PRETOKEN.finditer(piece):
                    result.extend(self._encode_piece(match.group()))
        return result

    def encode_iterable(self, iterable):
        # Retain the last two regex tokens: chunk edges can extend words,
        # contractions, or the whitespace preceding a word. Also retain enough
        # characters to recognize any special token split across chunks.
        pending = ""
        reserve = max(map(len, self.special_tokens), default=0)
        for chunk in iterable:
            pending += chunk
            spans = []
            offset = 0
            for special, piece in _pieces(pending, self.special_pattern):
                if special:
                    spans.append((offset, offset + len(piece), True, piece))
                else:
                    spans.extend(
                        (offset + m.start(), offset + m.end(), False, m.group()) for m in PRETOKEN.finditer(piece)
                    )
                offset += len(piece)
            end = 0
            for start, stop, special, piece in spans[:-2]:
                if stop > len(pending) - reserve:
                    break
                if special:
                    yield self.byte_ids[piece.encode("utf-8")]
                else:
                    yield from self._encode_piece(piece)
                end = stop
            pending = pending[end:]
        yield from self.encode(pending)

    def decode(self, ids):
        return b"".join(self.vocab[index] for index in ids).decode("utf-8", errors="replace")


def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    special_tokens = list(dict.fromkeys(special_tokens))
    if any(not t for t in special_tokens) or vocab_size < 256 + len(special_tokens):
        raise ValueError("Invalid vocabulary size or special tokens")
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    pattern = _special_pattern(special_tokens)
    counts = Counter()
    pending = ""
    reserve = max(map(len, special_tokens), default=0)
    with open(input_path, encoding="utf-8") as f:
        while chunk := f.read(1 << 20):
            pending += chunk
            spans = []
            offset = 0
            for special, piece in _pieces(pending, pattern):
                if special:
                    spans.append((offset + len(piece), True, piece))
                else:
                    spans.extend((offset + m.end(), False, m.group()) for m in PRETOKEN.finditer(piece))
                offset += len(piece)
            end = 0
            for stop, special, piece in spans[:-2]:
                if stop > len(pending) - reserve:
                    break
                if not special:
                    counts[piece.encode("utf-8")] += 1
                end = stop
            pending = pending[end:]
    for special, piece in _pieces(pending, pattern):
        if not special:
            counts.update(m.group().encode("utf-8") for m in PRETOKEN.finditer(piece))
    words = [tuple(bytes([b]) for b in word) for word in counts]
    frequencies = list(counts.values())
    pair_counts = Counter()
    locations = defaultdict(set)
    for index, word in enumerate(words):
        for pair, count in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += count * frequencies[index]
            locations[pair].add(index)

    class Entry:
        __slots__ = ("count", "pair")

        def __init__(self, count, pair):
            self.count, self.pair = count, pair

        def __lt__(self, other):
            return (self.count, self.pair) > (other.count, other.pair)

    heap = [Entry(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(vocab) < vocab_size and heap:
        entry = heapq.heappop(heap)
        pair = entry.pair
        if entry.count != pair_counts[pair] or entry.count <= 0:
            continue
        merges.append(pair)
        vocab[len(vocab)] = b"".join(pair)
        changed = set()
        for index in list(locations[pair]):
            old = words[index]
            new = []
            i = 0
            while i < len(old):
                if i + 1 < len(old) and (old[i], old[i + 1]) == pair:
                    new.append(old[i] + old[i + 1])
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
            new = tuple(new)
            old_pairs = Counter(zip(old, old[1:]))
            new_pairs = Counter(zip(new, new[1:]))
            for key in old_pairs.keys() | new_pairs.keys():
                delta = new_pairs[key] - old_pairs[key]
                if delta:
                    pair_counts[key] += delta * frequencies[index]
                    changed.add(key)
                if key not in new_pairs:
                    locations[key].discard(index)
                else:
                    locations[key].add(index)
            words[index] = new
        for key in changed:
            if pair_counts[key] > 0:
                heapq.heappush(heap, Entry(pair_counts[key], key))
    return vocab, merges
