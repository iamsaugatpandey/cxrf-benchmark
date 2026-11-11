#!/bin/zsh
# Download one NLM TB set (Montgomery or Shenzhen) with a manifest of sha256 checksums.
# Usage: fetch_nlm_tb.sh <Montgomery|Shenzhen> <dest_dir>
set -e
B=https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets
case $1 in
  Montgomery) ROOT="$B/Montgomery-County-CXR-Set/MontgomerySet"; SUBS=(CXR_png ClinicalReadings ManualMask/leftMask ManualMask/rightMask); EXTRA=(NLM-MontgomeryCXRSet-ReadMe.pdf montgomery_consensus_roi.csv);;
  Shenzhen)   ROOT="$B/Shenzhen-Hospital-CXR-Set"; SUBS=(CXR_png ClinicalReadings); EXTRA=(NLM-ChinaCXRSet-ReadMe.docx shenzhen_consensus_roi.csv);;
  *) echo "unknown set"; exit 1;;
esac
DEST=$2; mkdir -p "$DEST"
for f in $EXTRA; do curl -sSfL -o "$DEST/$f" "$ROOT/$f"; done
for s in $SUBS; do
  mkdir -p "$DEST/$s"
  for f in $(curl -sSfL "$ROOT/$s/index.html" | grep -oE "href='[^'/]+\.(png|txt)'" | sed "s/href='//;s/'//"); do
    [[ -s "$DEST/$s/$f" ]] || curl -sSfL --retry 5 --retry-all-errors --retry-delay 3 -o "$DEST/$s/$f" "$ROOT/$s/$f"
  done
done
(cd "$DEST" && find . -type f ! -name MANIFEST.sha256 | sort | xargs shasum -a 256 > MANIFEST.sha256)
echo "$1 done: $(find "$DEST" -name '*.png' | wc -l) png, $(du -sh "$DEST" | cut -f1), $(date -u +%FT%TZ)"
