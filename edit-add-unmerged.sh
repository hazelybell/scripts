#!/bin/bash

kate --new `git diff --name-only --diff-filter=U`
git add `git diff --name-only --diff-filter=U`
