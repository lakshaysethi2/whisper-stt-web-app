#!/usr/bin/env python3
"""Guard AWQ __global__ kernel bodies so CTranslate2 compiles for sm_50 (Maxwell)."""
import sys

GUARD_OPEN = "\n#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530\n"
GUARD_CLOSE = "\n#endif\n"


def code_mask(s):
    n = len(s)
    mask = [True] * n
    st = None
    i = 0
    while i < n:
        c = s[i]
        nxt = s[i + 1] if i + 1 < n else ""
        if st is None:
            if c == "/" and nxt == "/":
                mask[i] = mask[i + 1] = False; i += 2; st = "line"; continue
            if c == "/" and nxt == "*":
                mask[i] = mask[i + 1] = False; i += 2; st = "block"; continue
            if c == '"':
                mask[i] = False; i += 1; st = "str"; continue
            if c == "'":
                mask[i] = False; i += 1; st = "char"; continue
            i += 1; continue
        mask[i] = False
        if st == "line":
            if c == "\n": st = None
            i += 1
        elif st == "block":
            if c == "*" and nxt == "/":
                mask[i + 1] = False; i += 2; st = None
            else:
                i += 1
        elif st in ("str", "char"):
            if c == "\\" and i + 1 < n:
                mask[i + 1] = False; i += 2
            elif (st == "str" and c == '"') or (st == "char" and c == "'"):
                st = None; i += 1
            else:
                i += 1
    return mask


def patch(path):
    s = open(path).read()
    mask = code_mask(s)
    n = len(s)
    out, i, cnt = [], 0, 0
    while True:
        g = s.find("__global__", i)
        while g != -1 and not mask[g]:
            g = s.find("__global__", g + 1)
        if g == -1:
            out.append(s[i:]); break
        b = g
        while b < n and not (s[b] == "{" and mask[b]):
            b += 1
        depth, j = 0, b
        while j < n:
            if mask[j]:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        out.append(s[i:b + 1]); out.append(GUARD_OPEN)
        out.append(s[b + 1:j]); out.append(GUARD_CLOSE); out.append(s[j])
        i, cnt = j + 1, cnt + 1
    open(path, "w").write("".join(out))
    print(f"patched {cnt} kernel(s) in {path}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        patch(p)
