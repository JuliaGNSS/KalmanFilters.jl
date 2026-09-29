#!/usr/bin/env bash
# Checks that breaking changes in the given commit range are written so that
# the release notes pick them up.
#
# The shared semantic-release configuration bumps the version with the
# `conventionalcommits` preset but writes the release notes with the default
# `angular` preset. The two disagree:
#   - `feat!: ...` bumps the major version, but the angular parser cannot parse
#     the `!`, so it adds nothing to the "BREAKING CHANGES" section.
#   - That section is built only from `BREAKING CHANGE:` footers; `BREAKING:`
#     and friends are ignored.
# So a `!` commit is fine as long as it also carries a `BREAKING CHANGE:` footer.
# Release notes without a breaking-change section block the General registry's
# AutoMerge for a breaking release (see JuliaRegistries/General#169671).
#
# Usage: check-commit-messages.sh <revision range>, e.g. origin/master..HEAD
set -euo pipefail

range=${1:?usage: $0 <revision range>}
status=0

commits=$(git rev-list --no-merges "$range")
for sha in $commits; do
    subject=$(git log -1 --format=%s "$sha")
    body=$(git log -1 --format=%b "$sha")
    short=$(git rev-parse --short "$sha")

    if [[ $subject =~ ^[a-zA-Z]+(\([^\)]*\))?\!: ]] && ! grep -q "^BREAKING CHANGE: " <<<"$body"; then
        echo "::error::$short \"$subject\": the '!' alone does not reach the release notes' breaking-change section." \
            "Add a 'BREAKING CHANGE: <description>' footer."
        status=1
    fi

    # Any footer that looks like a breaking-change note but is not spelled
    # exactly `BREAKING CHANGE:` is silently ignored by one of the parsers.
    while IFS= read -r line; do
        if [[ ($line =~ ^[[:space:]]*BREAKING || ${line,,} =~ ^[[:space:]]*breaking[\ -]changes?[[:space:]]*:) && ! $line =~ ^BREAKING\ CHANGE:\  ]]; then
            echo "::error::$short \"$subject\": footer '${line:0:40}' is not recognised." \
                "Write it as 'BREAKING CHANGE: <description>'."
            status=1
        fi
    done <<<"$body"
done

if [[ $status -eq 0 ]]; then
    echo "All commit messages in $range are fine."
fi
exit $status
