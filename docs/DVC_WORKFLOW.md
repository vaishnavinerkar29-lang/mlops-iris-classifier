# DVC Workflow

## Remote Configuration

A local folder was configured as the DVC remote:

dvc remote add -d myremote ~/dvc-remote-storage

## Dataset Versioning Workflow

For every dataset change:

1. dvc add
2. git add *.dvc
3. git commit
4. dvc push

## Dataset Versions

Version 1:
150 rows

Version 2:
170 rows

## Version Comparison

dvc diff was used to compare dataset versions.

## Version Restoration

git checkout was used to restore the required .dvc pointer.

dvc checkout was then used to restore the actual dataset.

This allows reproducible data versions along with Git code versions.