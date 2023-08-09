Directory for notebook files for demonstrating features prior to a merge.

Before merging to the main branch, this should be cleared of any .ipynb files
to keep the main branch clean. This is checked by a GitHub action
(.github/workflows/premerge-checks.yml) which should block merging if tests
remain
