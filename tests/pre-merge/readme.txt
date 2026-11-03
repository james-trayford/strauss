Directory for notebook files for demonstrating features prior to a merge.

Before merging to the main branch, this should be cleared of any .ipynb files
to keep the main branch clean. This is checked by a GitHub action
(.github/workflows/premerge-checks.yml) which should block merging if tests
remain

Notebooks here should install the branch under test only when run on Colab,
so a local run uses the local install:

    import sys
    if 'google.colab' in sys.modules:
        %pip --quiet install git+https://github.com/james-trayford/strauss.git@<branch>
