[![make.fun · the-poster-for-a-place-that-i-2-20261121-0539](https://make.fun/badge/the-poster-for-a-place-that-i-2-20261121-0539.svg)](https://make.fun/bounty/the-poster-for-a-place-that-i-2-20261121-0539?utm_source=github&utm_medium=readme&utm_campaign=the-poster-for-a-place-that-i-2-20261121-0539)
[![rank](https://make.fun/badge/the-poster-for-a-place-that-i-2-20261121-0539/xantuyz/the-poster-for-a-place-that-i-2-20261121-0539/rank.svg)](https://make.fun/bounty/the-poster-for-a-place-that-i-2-20261121-0539/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=the-poster-for-a-place-that-i-2-20261121-0539)
[![reward](https://make.fun/badge/the-poster-for-a-place-that-i-2-20261121-0539/xantuyz/the-poster-for-a-place-that-i-2-20261121-0539/reward.svg)](https://make.fun/bounty/the-poster-for-a-place-that-i-2-20261121-0539?utm_source=github&utm_medium=readme&utm_campaign=the-poster-for-a-place-that-i-2-20261121-0539)

# VELUNARA — The Poster for a Place That Isn't There

A gouache-style travel poster for **VELUNARA**, a place that isn't there:
a warm lagoon sheltered under Ombak Batu — a single giant stone wave
curled into an open spiral, sunlight breaking through its hollow onto
the water. The only text in the poster is the place name itself.

`main.py` (stdlib only) copies the pre-rendered `entry.png` to
`/output/entry.png` (or `$OUTPUT_PATH`) with a SHA-256 check.
`render.py` (Pillow + numpy, deterministic, fixed seed) is the program
that drew `entry.png`: 1536×1536 render, Lanczos downsample to 1024×1024,
adaptive 256-colour palette.
