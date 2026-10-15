# Agent notes

## Releasing a new version

Release steps are in "Releasing" in CONTRIBUTING.md. Only maintainers can push
to `main`; everyone else bumps both pyproject versions in a pull request and
creates the tag from the GitHub release page.

Downstream projects pin this image by digest instead of following `latest`,
so a new version reaches them only when their pins are raised. After a
change merges to `main` or a release tag is pushed, ask the maintainer
whether to roll the new version out to those projects.

Each `main` build is tagged `sha-<commit>` (Python 3.11) and
`sha-<commit>-3.11` / `sha-<commit>-3.12`. Pin a version as
`<tag>@sha256:<digest>`.
