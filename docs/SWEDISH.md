# Swedish frontend (`language: sv`)

Swedish is the first phoneme-level frontend of the repo. Turkish and English feed characters to the text encoder;
Swedish normalises the text and then converts it to IPA with [espeak-ng](https://github.com/espeak-ng/espeak-ng)
through [phonemizer](https://github.com/bootphon/phonemizer), so the model's symbols are phonemes rather than letters.

- [Why phonemes](#why-phonemes)
- [Install](#install)
- [Normalisation](#normalisation)
- [Phonemes and symbols](#phonemes-and-symbols)
- [Pitch accents](#pitch-accents)
- [Training and synthesis](#training-and-synthesis)
- [Limits](#limits)
- [Tests](#tests)

## Why phonemes

Swedish spelling is not close enough to its pronunciation for a character vocabulary to be the obvious choice: `k` is
[k] or [ɕ] (`kall` / `kär`), `g` is [ɡ], [j] or part of the sj sound (`gå` / `ge` / `gärna`), the sj and tj sounds have
several spellings (`sjuk`, `skjorta`, `stjärna`, `tjock`), and `o` is [oː] or [ɔ]. A G2P built on espeak-ng's Swedish
dictionary removes that ambiguity and keeps the input short. The frontend is `drifting_tts/text_sv.py`.

## Install

`phonemizer` is an optional dependency, and espeak-ng is a system package:

```bash
pip install -e '.[sv]'                 # phonemizer
brew install espeak-ng                 # macOS
sudo apt install espeak-ng             # Debian / Ubuntu
phonemize --version                    # or: python -c 'from phonemizer import phonemize; print("ok")'
```

The import is lazy: `normalize()` and `prepare` work without phonemizer or espeak-ng (preparation stores normalised
orthography). Only `phonemize()`, `text_to_ids()` (unless `phonemized=True`) and therefore training data loading,
synthesis and evaluation need both.

## Normalisation

`normalize()` turns raw text into lower-case orthographic Swedish; it is what `prepare` stores as `norm_text` and what
`split_sentences()` splits. It covers:

| input | output |
|---|---|
| `12:30`, `07:05` | `tolv trettio`, `sju noll fem` |
| `500 kr`, `1 kr`, `1 250 kr` | `femhundra kronor`, `en krona`, `ettusen tvåhundrafemtio kronor` |
| `$5`, `€3`, `£2` | `fem dollar`, `tre euro`, `två pund` |
| `20%`, `20°C` | `tjugo procent`, `tjugo grader celsius` |
| `5 kg`, `10 km`, `100 km/h` | `fem kilogram`, `tio kilometer`, `hundra kilometer i timmen` |
| `5 januari 2024`, `2024-01-05`, `5/1/2024` | `femte januari tvåtusentjugofyra` (day-first) |
| `1/2`, `3/4`, `1 1/2` | `en halv`, `tre fjärdedelar`, `en och en halv` |
| `21:a`, `3:e`, `nr 7` | `tjugoförsta`, `tredje`, `nummer sju` |
| `070-123 45 67`, `+46 70 123 45 67` | digit by digit, `plus fyra sex` for the country code |
| `t.ex.`, `d.v.s.`, `kl.`, `ca.` | `till exempel`, `det vill säga`, `klockan`, `cirka` |
| `Dr. Anders`, `St. Erik` | `doktor Anders`, `sankt Erik` |
| `AB`, `USA`, `EU`, `PDF` | `aktiebolag`, `u ess a`, `e u`, `pe de eff` |
| `www.example.se/test`, `a.b@example.com` | `ve ve ve punkt example punkt se snedstreck test`, … |

Only a sentence-final `.`, `!` or `?` survives (so `split_sentences()` finds sentence ends); the dots of
abbreviations, titles and addresses are resolved. `1 250` and `1.250` are read as 1250 (Swedish thousands); `3,14`
is read as *tre komma ett fyra*.

## Phonemes and symbols

`phonemize()` calls

```python
phonemize(text, language="sv", backend="espeak", with_stress=True,
          preserve_punctuation=True, strip=True)
```

and keeps `ˈ` (primary stress), `ˌ` (secondary stress), `ː` (length) and the IPA inventory espeak-ng's Swedish voice
emits (`ɑ ɔ ɕ ə ɛ ɡ ɪ ɵ ʉ ʊ ʂ ʃ ø œ ŋ …`), plus the punctuation ` .,!?`. espeak-ng writes the Swedish sj phoneme as
the two-character name `sx` (sometimes with a zero-width joiner); `phonemize()` rewrites it to the single symbol `ɧ`.
Anything outside the table is replaced by a space, so `text_to_ids()` can never produce an unknown id. `SYMBOLS`
keeps `<pad>` and `<blank>` at ids 0 and 1, as the other frontends do; a Swedish model therefore has a larger
vocabulary (`drifting_tts.models.tts` reads its size from `frontend(language).SYMBOLS`).

`text_to_ids(text, normalized=True)` expects text that has already passed through `normalize()` and still phonemises
it; `text_to_ids(ipa, phonemized=True)` maps an IPA string directly (useful for tests and for cached phonemes).

## Pitch accents

Swedish has two lexical pitch accents (*accent 1* / *accent 2*, historically acute / grave): `anden` "the duck" and
`anden` "the spirit" have the same phonemes and differ only in the pitch contour of the word. espeak-ng's IPA output
does **not** mark them, so the current frontend cannot distinguish the two from text alone.

The acoustic model can learn the distribution from a sufficiently large corpus, and espeak-ng already writes primary
and secondary stress (`ˈ`, `ˌ`), which is the part the IPA can express. If exact accents matter, add a lexicon-driven
accent tagger before training: append one symbol per word (for example `˥` for accent 1 and `˩` for accent 2) in
`phonemize()`, add it to `PHONEMES`, and train the model on the accented output. This must be done before the model's
`SYMBOLS` are fixed; changing the inventory afterwards changes the embedding size.

## Training and synthesis

Prepare with the Swedish frontend and train the ARDrift recipe:

```bash
drifting-tts prepare --parquet-glob "data/sv/*.parquet" --out data/sv --backend bigvgan \
    --language sv --speaker-column speaker_id --workers 8
drifting-tts train-ar --config configs/ar_tts_sv.yaml --workdir runs/ar_sv
```

`configs/ar_tts_sv.yaml` is the English recipe with `model.text.language: sv`. The dataset's `stats.json` records
`"language": "sv"`, so `data.MelDataset`, `evaluate`, `benchmark`, `prosody`, `score` and the `Synthesizer` pick the
Swedish normaliser and symbol table automatically. A model checkpoint with `model.text.language: sv` synthesises
Swedish text directly:

```python
from drifting_tts import Synthesizer
tts = Synthesizer("runs/ar_sv/model_ema.pt")
wav, info = tts("Hej! Det kostar 500 kr.")
```

The context mode (`context=True`) conditions each sentence on the previous one; its boundary is computed in tokens,
not characters, so it works for a phoneme frontend too.

## Limits

- **Phonemisation is not cached.** Each `text_to_ids()` call invokes espeak-ng. Data loading workers do it in
  parallel; for repeated inference of the same text, cache the IPA or pass `phonemized=True`.
- **Orthography is not in `SYMBOLS`.** `normalize()` returns Swedish letters, so do not validate `norm_text` against
  `SYMBOL_TO_ID`; phonemes are the model input. `ids_to_text()` returns IPA, not orthography.
- **espeak-ng's dictionary is not a pronunciation model.** Names, loanwords and new compounds can be wrong; add them
  to a user dictionary (`espeak-ng --compile=sv` / a phonemizer lexicon) if a corpus needs them.
- **Prosody features** (`drifting_tts.sentence_features`) are Turkish-specific. They still run on the phoneme string
  (spaces and sentence punctuation remain), but only the statement/position features are meaningful for Swedish.
- **MLX conversion** (`drifting_tts.mlx`) is Turkish-only until its converter and synthesiser take the checkpoint's
  language; use the PyTorch `Synthesizer` for Swedish.

## Tests

`tests/test_text_sv.py` covers normalisation, idempotence, sentence splitting, the symbol table, phoneme-id mapping,
the phonemizer (skipped when phonemizer or espeak-ng is missing), syllables and the model vocabulary:

```bash
pytest tests/test_text_sv.py -q
```
