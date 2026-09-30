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
# Separately, the conventionalcommits parser takes any line that starts with the
# keyword, in any case and followed by ':' or a space, as a breaking-change note,
# so wrapped prose can bump the major version by accident.
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

    while IFS= read -r line; do
        [[ $line =~ ^BREAKING\ CHANGE:\  ]] && continue
        lower=${line,,}
        # The conventionalcommits parser takes any line that starts with the
        # keyword, in any case and followed by ':' or whitespace, as a
        # breaking-change note, and bumps the major version. Wrapped prose
        # such as "breaking-change section and ..." is enough.
        if [[ $lower =~ ^[[:space:]|*]*breaking[\ -]change[:[:space:]] ]]; then
            echo "::error::$short \"$subject\": line '${line:0:40}' is read as a breaking-change note and would bump the major version." \
                "Reword it so no line starts with 'breaking change'; for a breaking change, write 'BREAKING CHANGE: <description>'."
            status=1
        # Any other footer that looks like a breaking-change note but is not
        # spelled exactly `BREAKING CHANGE:` is silently ignored by one of the
        # parsers.
        elif [[ $line =~ ^[[:space:]]*BREAKING || $lower =~ ^[[:space:]]*breaking[\ -]changes?[[:space:]]*: ]]; then
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
