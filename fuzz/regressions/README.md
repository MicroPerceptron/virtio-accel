# Fuzz Regressions

Commit minimized crashing inputs under `fuzz/regressions/<target>/`.

CI passes each existing target directory to `cargo fuzz run` after generating the ordinary seed
corpus, so a fixed crash becomes a permanent regression test without checking in generated corpus
files.
