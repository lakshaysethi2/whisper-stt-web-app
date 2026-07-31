"""Build ground-truth transcripts from LibriSpeech chapter data.

Given a directory of .flac files and a .trans.txt file, produce truncated
transcripts matching 1min and 5min audio durations.

Usage: python3 _build_transcripts.py <input_dir> <output_dir>
"""
import os
import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Find .trans.txt
    trans_file = None
    for f in os.listdir(input_dir):
        if f.endswith(".trans.txt"):
            trans_file = os.path.join(input_dir, f)
            break
    if not trans_file:
        print("ERROR: no .trans.txt found in", input_dir, file=sys.stderr)
        sys.exit(1)

    # Read utterances: (utt_id, duration_sec, text)
    utterances = []
    with open(trans_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = parts[1]
            flac_path = None
            for candidate in os.listdir(input_dir):
                if candidate.startswith(utt_id) and candidate.endswith(".flac"):
                    flac_path = os.path.join(input_dir, candidate)
                    break
            if flac_path:
                result = subprocess.run(
                    ["soxi", "-D", flac_path],
                    capture_output=True, text=True,
                )
                try:
                    dur = float(result.stdout.strip())
                except ValueError:
                    dur = 0.0
                utterances.append((utt_id, dur, text))

    utterances.sort(key=lambda x: x[0])  # sort by utt_id

    print(f"      Loaded {len(utterances)} utterances")

    def write_transcript(limit_sec, outpath):
        accum = 0.0
        lines = []
        for utt_id, dur, text in utterances:
            if accum >= limit_sec and lines:
                break
            lines.append(text)
            accum += dur
        # Always include at least 1 utterance
        if not lines and utterances:
            lines.append(utterances[0][2])
        with open(outpath, "w") as f:
            f.write(" ".join(lines) + "\n")
        num_words = len(" ".join(lines).split())
        return num_words

    for limit, suffix in [(60, "1min"), (300, "5min")]:
        outpath = os.path.join(output_dir, f"{suffix}_transcript.txt")
        num_words = write_transcript(limit, outpath)
        print(f"      {suffix}_transcript.txt: {num_words} words")


if __name__ == "__main__":
    main()
