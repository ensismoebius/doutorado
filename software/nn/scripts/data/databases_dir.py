"""databases_dir.py -- the one directory that holds every data base of the project.

    ~/Documentos/academico/UNESP/doutorado/databases/
        fsdDataset/  audioMNIST_8k/  eegmmidb/  siena/  mitbih/   (meeting01; ensure_datasets.sh)
        database.sqlite   (+ its .e05dscache caches and .mirror-verified stamp)   Coretto 2017
        biomex.sqlite     (+ .verified stamp)  biomex/{archives,extracted}        BIOMEX-DB
        coretto_zenodo/                                          the Coretto EEG mirror

The thesis profiles name their base as `~/Documentos/academico/UNESP/doutorado/databases/<file>`
(dataset.root), so a script that downloads, converts or verifies a base uses the same place by
default; a path given on the command line still wins. It is a fixed relative path under the home
directory, not a setting: the cluster has to hold the same layout under its own home
(gridunesp_deploy.sh creates it), because the profile is the same on both machines.

The same string is in scripts/pipeline/thesis/00_thesis_generate_profiles.py (DATABASES) and
scripts/pipeline/meeting01/{ensure_datasets,gridunesp_deploy}.sh; a test pins them together
(test_databases_dir.py).
"""
import os

DATABASES_REL = "Documentos/academico/UNESP/doutorado/databases"


def databases_dir() -> str:
    return os.path.join(os.path.expanduser("~"), DATABASES_REL)


def in_databases(name: str) -> str:
    return os.path.join(databases_dir(), name)
