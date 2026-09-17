# Working notes for this repo

Conventions for this project. Read before editing.

## Voice and comments

This is a capstone project that gets read by a human marker and defended out loud, so the
code has to be explainable, not just correct.

- Comments explain **why**, never what. `# increment counter` above `i += 1` is noise.
- Put the reasoning where the decision was made. If a line looks wrong until you know some
  context, that context belongs next to the line.
- Vary the register. Not every function needs a docstring; the ones carrying a real design
  decision need a proper one, and a two-line helper needs nothing.
- Where something was tried and rejected, say so and say why. The Socrata `$order` timing,
  the cost-optimal threshold, the random split: those notes are the most useful thing in the
  file, because they are the questions a marker will ask.
- Avoid the `--` dash tic, stacked bold, and the same sentence shape repeated down a file.
- British or American spelling, just be consistent within a file.

## Correctness rules that are not negotiable

- **Never write a number into the README, a docstring or a notebook that was not produced by
  running the code.** If a metric changes, re-run `train.py` and copy the new value. Inventing
  or adjusting results fails the whole capstone outright (evaluation criteria §6).
- After any change to `src/sla.py` or `train.py`, run `python test_sla.py` (11 assertions,
  ~3 s). After a change touching features, the target or the split, re-run `train.py` and
  confirm the reported metrics still match the README.
- The test set is scored once, in `train.py`, after the model and threshold are fixed on
  validation. No other code path should touch it.
- Preprocessing stays inside the sklearn Pipeline. Nothing gets fitted on the dataframe
  outside it.
- `data/*.csv` and `mlruns/` stay out of git. The dataset is ~73 MB and re-downloadable.

## Git conventions

- Commit messages say what changed and why, not just what.
- Keep `Co-Authored-By:` trailers on commits that had AI assistance. The course policy (§6)
  lists manipulating authorship records as misconduct, and the acknowledgement in the README
  depends on the history matching it.
- Tests pass before anything is pushed.

## Course requirements this repo has to keep satisfying

- README must keep: title, problem statement, track, dataset source, ML task type, pipeline,
  models tested, final model and justification, metrics, install steps, training steps, demo
  instructions, example input/output, limitations, Responsible AI, student's full name.
- `demo.ipynb` must run top to bottom on a clean Colab runtime, cloning this repo.
- The repo stays public so mentors can reach it, and `REPO_URL` in the notebook's first cell
  has to keep pointing here.
