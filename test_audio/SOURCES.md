# Test Audio Sources

All test audio files are derived from **LibriSpeech** (OpenSLR #12), a corpus of
public-domain audiobook recordings.

- **Dataset:** LibriSpeech `dev-clean`
- **Speaker:** 3081 (female, US English)
- **Chapter:** 166546 — *The Circular Staircase* by Mary Roberts Rinehart
- **License:** Public domain / CC0 (LibriSpeech is derived from Librivox recordings
  which are public domain in the USA)

## Upstream URLs

| Resource | URL |
|----------|-----|
| OpenSLR LibriSpeech home | https://www.openslr.org/12/ |
| dev-clean tarball | https://www.openslr.org/resources/12/dev-clean.tar.gz |
| Speaker 3081 page (Librivox) | https://librivox.org/reader/3081 |
| Chapter text (Gutenberg) | https://www.gutenberg.org/ebooks/1699 |

## Fixture files

| File | Approx duration | Description |
|------|----------------|-------------|
| `1min.flac` | ~48 s | First ~1 minute of chapter 166546 |
| `5min.flac` | ~5 min | First ~5 minutes of chapter 166546 |
| `full.flac` | ~8 min | Complete chapter 166546 |

## Rebuilding

Run `scripts/build_test_audio.sh` from the project root to rebuild these
fixtures from upstream LibriSpeech assets. Requires Docker (for curl, ffmpeg).

## License note

These derived fixture files (short excerpts of a public-domain recording) are
distributed under the same public-domain terms as the original LibriSpeech
corpus.
