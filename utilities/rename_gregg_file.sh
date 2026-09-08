#!/usr/bin/env bash
set -uo pipefail

# Rename files according to a tab-separated spelling-correction key.
#
# Required form:
#   ./rename_gregg_file.sh --modify "DIRECTORY_PATH" "SPELLING_KEY"
#
# Example:
#   ./rename_gregg_file.sh --modify \
#       "/home/nathantonning/path/to/media/gregg" \
#       "/home/nathantonning/Downloads/spelling_corrections_round2.txt"
#
# Spelling-key format:
#   OLD_FILENAME<TAB>NEW_FILENAME
#
# Safety:
#   - Existing destination files are NEVER overwritten.
#   - If the corrected destination already exists, the script asks whether
#     to delete the old/misspelled source file.
#   - Only an explicit "y" or "Y" deletes a conflicting source file.
#   - Missing source files are reported and skipped.
#   - Blank lines and lines beginning with # in the key are ignored.

usage() {
    cat <<'EOF_USAGE'
Usage:
  ./rename_gregg_file.sh --modify "DIRECTORY_PATH" "SPELLING_KEY"

Arguments:
  --modify         Apply the filename corrections.
  DIRECTORY_PATH   Directory containing the files to rename.
  SPELLING_KEY     Tab-separated correction file in the form:
                       OLD_FILENAME<TAB>NEW_FILENAME

Example:
  ./rename_gregg_file.sh --modify \
      "/home/nathantonning/Music/Sounds/PD_Tracks/gregg" \
      "/home/nathantonning/Downloads/spelling_corrections_round2.txt"

Conflict behavior:
  If both OLD_FILENAME and NEW_FILENAME already exist, NEW_FILENAME is kept
  and you are asked whether to delete OLD_FILENAME:

      CONFLICT: misspelled.svg -> corrected.svg
      Corrected destination already exists.
      Delete misspelled source 'misspelled.svg'? [y/N]

  Press y (or Y) to delete that one misspelled source file.
  Press Enter or anything else to keep it.
EOF_USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 3 || "$1" != "--modify" ]]; then
    echo "ERROR: Expected --modify, a target directory, and a spelling key." >&2
    echo >&2
    usage >&2
    exit 2
fi

TARGET_DIR="$2"
KEY_FILE="$3"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "ERROR: Directory does not exist: $TARGET_DIR" >&2
    exit 1
fi

if [[ ! -f "$KEY_FILE" ]]; then
    echo "ERROR: Spelling key does not exist: $KEY_FILE" >&2
    exit 1
fi

# Convert both inputs to absolute paths for clearer output and to make the
# behavior independent of the caller's current working directory.
TARGET_DIR="$(cd -- "$TARGET_DIR" && pwd)"
KEY_DIR="$(cd -- "$(dirname -- "$KEY_FILE")" && pwd)"
KEY_FILE="$KEY_DIR/$(basename -- "$KEY_FILE")"

echo "Mode:         MODIFY"
echo "Directory:    $TARGET_DIR"
echo "Spelling key: $KEY_FILE"
echo

renamed=0
missing=0
conflicts=0
deleted_conflicts=0
kept_conflicts=0
invalid=0

while IFS=$'\t' read -r old new extra || [[ -n "${old:-}" ]]; do
    # Ignore blank lines and comments.
    [[ -z "${old:-}" ]] && continue
    [[ "$old" == \#* ]] && continue

    # Remove a trailing carriage return in case the key uses CRLF line endings.
    new="${new%$'\r'}"
    extra="${extra%$'\r'}"

    if [[ -z "${new:-}" || -n "${extra:-}" ]]; then
        printf 'INVALID: expected exactly two tab-separated fields: %q\n' "$old" >&2
        ((invalid++))
        continue
    fi

    # The correction key is intentionally filename-only. Prevent a malformed
    # entry from escaping the selected target directory.
    if [[ "$old" == */* || "$new" == */* || "$old" == "." || "$old" == ".." || "$new" == "." || "$new" == ".." ]]; then
        printf 'INVALID: filenames may not contain path components: %s -> %s\n' "$old" "$new" >&2
        ((invalid++))
        continue
    fi

    src="$TARGET_DIR/$old"
    dst="$TARGET_DIR/$new"

    if [[ ! -e "$src" ]]; then
        printf 'MISSING:  %s\n' "$old"
        ((missing++))
        continue
    fi

    # If the old and new names are identical, there is nothing to do.
    if [[ "$old" == "$new" ]]; then
        printf 'SKIP:     %s already has the requested name\n' "$old"
        continue
    fi

    # Conflict: corrected destination already exists. Never overwrite it.
    if [[ -e "$dst" ]]; then
        ((conflicts++))
        printf '\nCONFLICT: %s -> %s\n' "$old" "$new"
        echo "Corrected destination already exists."

        # Read directly from the terminal so the spelling-key input stream
        # remains untouched.
        if [[ -r /dev/tty ]]; then
            printf "Delete misspelled source '%s'? [y/N] " "$old" > /dev/tty
            IFS= read -r answer < /dev/tty || answer=""
        else
            answer=""
            echo "No interactive terminal available; keeping source file."
        fi

        case "$answer" in
            y|Y)
                if rm -- "$src"; then
                    printf 'DELETED:  %s\n' "$old"
                    ((deleted_conflicts++))
                else
                    printf 'ERROR:    failed to delete %s\n' "$old" >&2
                    ((kept_conflicts++))
                fi
                ;;
            *)
                printf 'KEPT:     %s\n' "$old"
                ((kept_conflicts++))
                ;;
        esac

        continue
    fi

    # Normal correction: destination does not already exist.
    if mv -- "$src" "$dst"; then
        printf 'RENAMED:  %s -> %s\n' "$old" "$new"
        ((renamed++))
    else
        printf 'ERROR:    failed to rename %s -> %s\n' "$old" "$new" >&2
    fi

done < "$KEY_FILE"

echo
echo "Summary"
echo "-------"
printf 'Renamed:             %d\n' "$renamed"
printf 'Conflicts found:     %d\n' "$conflicts"
printf 'Conflicts deleted:   %d\n' "$deleted_conflicts"
printf 'Conflicts kept:      %d\n' "$kept_conflicts"
printf 'Missing:             %d\n' "$missing"
printf 'Invalid key entries: %d\n' "$invalid"
