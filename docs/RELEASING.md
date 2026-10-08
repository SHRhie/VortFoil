# Releasing a new version

Checklist for a maintainer cutting a new `vortfoil` release.

## Before the first release: the publication gate

The first release is tied to the manuscript, so four things have to be settled before any of the
steps below are worth starting. None of them is a packaging question.

1. **License.** Settled on 2026-10-01: GPL-3.0-or-later (`LICENSE`, `pyproject.toml`,
   `CITATION.cff`), because `xfoil_closures.py` is a translation of XFOIL 6.99, which is GPL v2 or
   later; `NOTICE` records the origin. Data notices are in `DATA_LICENSES.md`.
2. **Funding acknowledgment.** Absent from the manuscript.
3. **Authors, order and CRediT statement.** Unchanged pending coauthor review.
4. **AI-use declaration.** Deliberately absent; the corresponding author drafts it.

Two couplings decide the *order* of everything else.

**The repository must go public before the manuscript is submitted.** The manuscript's
code-and-data-availability section states that the source is publicly available. Until the
repository is public that sentence is false, so the visibility flip, the release and the
submission belong in one sitting. Publishing to PyPI while the repository is private also leaves
the project page pointing at a URL that returns 404 for everyone.

**Enable the Zenodo GitHub integration before creating the release, not after.** Zenodo archives
releases created *after* the webhook is connected. A release made first is not picked up, and has
to be uploaded by hand.

**Sync with the manuscript before going public.** `paper/abstract.md`, the
README field table and `figures/vf_field_f0cm_tp_p10.*` must match the manuscript as it will be
submitted. Re-check them after the last manuscript build, and before building anything for PyPI:
the README is the PyPI project description.

So the publication-day order is: settle the four decisions above, sync with the manuscript, flip the
repository to public, connect Zenodo, then run the checklist below, and only then submit the
manuscript with the DOI in place.

## Cutting the release

1. Bump `__version__` in `src/vortfoil/__init__.py` and `version` in
   `pyproject.toml` together. The two must match.
2. Regenerate `src/vortfoil/weights/MANIFEST.json` if any weight file or any
   recorded score changed.
3. Run the tests with the evaluation grid:

   ```bash
   VORTFOIL_EVAL_DATA=data/eval/holdout_targets.npz pytest -q tests/
   ```

4. Build the distribution:

   ```bash
   python -m build
   ```

5. Check the built artifacts:

   ```bash
   twine check dist/*
   ```

6. Upload to TestPyPI, then install from it into a clean virtual environment
   with `--no-deps` plus a TensorFlow of your choice, and run the self-check:

   ```bash
   pip install --no-deps -i https://test.pypi.org/simple/ vortfoil
   pip install "tensorflow>=2.10"   # or whatever TensorFlow you already have pinned
   python -m vortfoil.cli selfcheck --data data/eval/holdout_targets.npz
   ```

   > **Warning.** Inside a pinned TensorFlow environment, always install
   > `vortfoil` with `--no-deps`. The package depends on `tensorflow>=2.10`;
   > without `--no-deps`, pip will pull a newer TensorFlow and Keras over the
   > one you have pinned.

7. Do not upload to PyPI by hand. Publishing the GitHub release in step 9 runs the `publish` job of
   `.github/workflows/ci.yml`, which uploads with PyPI's trusted publishing once the tests and the clean-wheel check
   pass. On PyPI, the pending publisher is: project `VortFoil` (normalized `vortfoil`), repository
   `SHRhie/VortFoil`, workflow `ci.yml`, environment `pypi`. It was registered on 2026-10-08 and becomes active with
   that first upload. The job also checks that the release tag is `v` + the wheel's version.
8. Confirm the Zenodo GitHub integration is already connected for this repository. If it is not,
   connect it now; a release created before it is connected will not be archived.
9. Tag the commit `v2.0.0` and create a GitHub release, attaching the wheel and the source
   distribution. Release notes are drafted in `RELEASE_NOTES_v2.0.0.md` next to this file:

   ```bash
   gh release create v2.0.0 dist/* --notes-file docs/RELEASE_NOTES_v2.0.0.md
   ```

   This one step triggers the PyPI upload (step 7) and the Zenodo archive (step 8). GitHub creates the `pypi`
   environment when the job first runs. A private repository may need a paid plan for environments, which is one
   more reason the release comes after the repository is public. A version uploaded to PyPI can never be
   re-uploaded, so a fix after this needs 2.0.1.

10. Zenodo mints a DOI for the archived release. Put it in `CITATION.cff` (which carries an
    author placeholder for exactly this) and in the manuscript's code-and-data-availability
    section, then rebuild the manuscript.

## State as of 2026-10-08

Version `2.0.0` agrees across `src/vortfoil/__init__.py`, `pyproject.toml` and
`src/vortfoil/weights/MANIFEST.json`. The repository starts from a single commit of the release tree that matches
the 51-page manuscript build of 2026-10-06 (the manuscript PDF itself is added after submission). It is private
until the publication gate above is closed. There are no tags and no releases, no DOI has been minted, and the
name `vortfoil` was unclaimed on PyPI and TestPyPI on 2026-10-01. A PyPI pending trusted publisher for it was
registered on 2026-10-08 (step 7).
