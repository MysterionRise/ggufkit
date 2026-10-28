# Releasing ggufkit

Releases are built and published by `.github/workflows/release.yml`. A `vX.Y.Z` tag on `main`
publishes; running the workflow by hand is a dry run that does everything except publish. No
PyPI token is stored anywhere: PyPI trusts this repository's workflow directly (trusted
publishing).

## One-time setup

1. **PyPI trusted publisher.** Sign in at pypi.org (with 2FA), open
   [Publishing](https://pypi.org/manage/account/publishing/) and add a *pending publisher* for
   GitHub:

   | field | value |
   |---|---|
   | PyPI project name | `ggufkit` |
   | Owner | `MysterionRise` |
   | Repository name | `ggufkit` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The first successful publish turns it into a normal trusted publisher for the project.

2. **Protect the `pypi` environment (recommended).** In GitHub, Settings → Environments →
   `pypi` (it appears after the workflow first runs; you can also create it now): add yourself
   as a required reviewer and limit deployment to tags matching `v*`. Every publish then waits
   for your approval.

3. **Private vulnerability reporting.** Settings → Code security → Private vulnerability
   reporting → Enable. `SECURITY.md` sends reporters there.

## Each release

1. **Dry run.** Actions → Release → *Run workflow* on `main`. It runs the full CI, builds the
   wheel and sdist, checks them with `twine check --strict`, installs each in a clean
   environment and runs `scripts/smoke_test.py`, and records build provenance. Nothing is
   published. Fix anything red before going on.

2. **Version and date.** On a branch: set `version` in `pyproject.toml` (it must match the
   tag), replace `Unreleased` with today's date in the version's `CHANGELOG.md` heading, set
   `version` and `date-released` in `CITATION.cff`, run `uv lock`. Merge it.

3. **Tag.** On the merged `main`:

   ```sh
   git switch main && git pull
   git tag -a v0.1.0 -m "ggufkit 0.1.0"
   git push origin v0.1.0
   ```

   Without a terminal, use GitHub's *Releases → Draft a new release* instead:
   - choose a tag, type `v0.1.0`, and pick *Create new tag on publish*, with target `main`;
   - leave the title and description empty;
   - click *Publish release*. A draft creates no tag, so it starts nothing.

   Publishing creates the tag, which starts the workflow. The workflow then attaches the
   wheel and sdist to that release and fills in its title and the CHANGELOG notes.

4. **Watch the workflow.** It checks that the tag matches `pyproject.toml` and is on `main`,
   runs CI, builds, smoke-tests and attests, then waits for your approval on the `pypi`
   environment if you set that up. After the upload it creates the GitHub release with the
   CHANGELOG section as notes and the built files attached.

5. **Check the result.**

   ```sh
   python -m venv /tmp/check && /tmp/check/bin/pip install "ggufkit[viz]==0.1.0"
   /tmp/check/bin/ggufkit --version

   # Optional: check the build provenance of the published wheel.
   pip download ggufkit==0.1.0 --no-deps -d /tmp/dl
   gh attestation verify /tmp/dl/ggufkit-0.1.0-py3-none-any.whl --repo MysterionRise/ggufkit \
     --signer-workflow MysterionRise/ggufkit/.github/workflows/release.yml \
     --source-ref refs/tags/v0.1.0
   ```

   The wheel builds reproducibly, so dry runs attest the same bytes as the release and the
   lookup finds their attestations too. `--source-ref` accepts only the one built from the tag.

   On https://pypi.org/project/ggufkit/ the files show a *Provenance* section (the PyPI
   attestations). The GitHub release page lists the wheel and sdist.

If something goes wrong after the upload, do not delete and re-upload the same version: PyPI
refuses reused file names. Fix it, bump the patch version and release again; yank the broken
version on PyPI if it should not be installed.
