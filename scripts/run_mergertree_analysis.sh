#!/bin/bash
# run_mergertree_analysis.sh
# Runs AHF MergerTree cross-correlation between CDM and WDM simulations.
#
# NOTE: MergerTree should be compiled WITHOUT the `MTREE_BOTH_WAYS` flag
#
# Usage:
#   ./run_mergertree_analysis.sh cdm_wdm   # CDM as file 1, WDM as file 2 (primaries = CDM)
#   ./run_mergertree_analysis.sh wdm_cdm   # WDM as file 1, CDM as file 2 (primaries = WDM)
#
# Output is written to:
#   data/raw/CDM_WDM_crossmatch/   for cdm_wdm
#   data/raw/WDM_CDM_crossmatch/   for wdm_cdm
#
# A logfile is written alongside the output for each setup directory.

# ---------------------------------------------------------------------------
# Parse direction argument
# ---------------------------------------------------------------------------
DIRECTION="${1:-}"

if [ "$DIRECTION" = "cdm_wdm" ]; then
    FILE1_LABEL="CDM"
    FILE2_LABEL="WDM"
elif [ "$DIRECTION" = "wdm_cdm" ]; then
    FILE1_LABEL="WDM"
    FILE2_LABEL="CDM"
else
    echo "ERROR: direction argument required."
    echo "Usage: $0 cdm_wdm | wdm_cdm"
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
rundir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$rundir" || exit 1

BASEPATH="$rundir/data/raw"
CDMSIMDIR="B25_N512_CDM_3LPT"
WDMSIMDIR="B25_N512_WDM_3LPT"
SETUPDIRS=( "z39_adapt" "z39_fixed" "z99_adapt" "z99_fixed" )
OUTBASEPATH="$BASEPATH/${FILE1_LABEL}_${FILE2_LABEL}_crossmatch"

: "${MTREEBINPATH:?MTREEBINPATH must be set — path to the AHF MergerTree binary}"

# ---------------------------------------------------------------------------
# Header — printed to stdout and captured in each logfile via tee
# ---------------------------------------------------------------------------
echo "========================================"
echo "MergerTree cross-correlation"
echo "Direction : $FILE1_LABEL (file 1, primaries) -> $FILE2_LABEL (file 2, children)"
echo "Meaning   : each $FILE1_LABEL halo lists its $FILE2_LABEL counterparts"
echo "Date      : $(date)"
echo "Hostname  : $(hostname -s)"
echo "Run dir   : $(pwd)"
echo "Output    : $OUTBASEPATH"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for setupdir in "${SETUPDIRS[@]}"
do
    CDM_DIR="$BASEPATH/$CDMSIMDIR/$setupdir"
    WDM_DIR="$BASEPATH/$WDMSIMDIR/$setupdir"

    MTREEOUTPATH="$OUTBASEPATH/$setupdir"
    mkdir -p "$MTREEOUTPATH"

    LOGFILE="$MTREEOUTPATH/run_mergertree_${DIRECTION}.log"

    # Tee all output for this setup dir to its logfile
    {
        echo "==============================="
        echo "Setup : $setupdir"
        echo "Direction : $FILE1_LABEL -> $FILE2_LABEL"
        echo "Started   : $(date)"

        # Assign file 1 and file 2 based on direction
        if [ "$DIRECTION" = "cdm_wdm" ]; then
            FILE1_CANDIDATES=( "$CDM_DIR"/AHF."${CDMSIMDIR}".snap_*.0000.z0.000.AHF_particles )
            FILE2_CANDIDATES=( "$WDM_DIR"/AHF."${WDMSIMDIR}".snap_*.0000.z0.000.AHF_particles )
            PREFIX_TAG="${CDMSIMDIR}_vs_${WDMSIMDIR}"
        else
            FILE1_CANDIDATES=( "$WDM_DIR"/AHF."${WDMSIMDIR}".snap_*.0000.z0.000.AHF_particles )
            FILE2_CANDIDATES=( "$CDM_DIR"/AHF."${CDMSIMDIR}".snap_*.0000.z0.000.AHF_particles )
            PREFIX_TAG="${WDMSIMDIR}_vs_${CDMSIMDIR}"
        fi

        FILE1="${FILE1_CANDIDATES[0]}"
        FILE2="${FILE2_CANDIDATES[0]}"

        if [ ! -f "$FILE1" ]; then
            echo "ERROR: $FILE1_LABEL AHF_particles file not found in — skipping."
            echo "       Expected: $FILE1"
            echo ""
            continue
        fi
        if [ ! -f "$FILE2" ]; then
            echo "ERROR: $FILE2_LABEL AHF_particles file not found — skipping."
            echo "       Expected: $FILE2"
            echo ""
            continue
        fi

        echo "File 1 ($FILE1_LABEL) : $FILE1"
        echo "File 2 ($FILE2_LABEL) : $FILE2"

        # Extract snap identifier from file 1 filename (e.g. snap_056)
        SNAP=$(basename "$FILE1" | cut -d'.' -f3)

        MTREEPREFIX="$MTREEOUTPATH/MTREE.${PREFIX_TAG}.${setupdir}.${SNAP}.z0.000"
        echo "Output prefix : $MTREEPREFIX"

        # Build MergerTree input file
        rm -f "$MTREEOUTPATH/MTREE.temp_nfiles" \
              "$MTREEOUTPATH/MTREE.temp_particles" \
              "$MTREEOUTPATH/MTREE.temp_prefix"

        printf '2\n'                    > "$MTREEOUTPATH/MTREE.temp_nfiles"
        printf '%s\n' "$FILE1" "$FILE2" > "$MTREEOUTPATH/MTREE.temp_particles"
        printf '%s\n' "$MTREEPREFIX"    > "$MTREEOUTPATH/MTREE.temp_prefix"

        cat "$MTREEOUTPATH/MTREE.temp_nfiles" \
            "$MTREEOUTPATH/MTREE.temp_particles" \
            "$MTREEOUTPATH/MTREE.temp_prefix" \
            > "$MTREEOUTPATH/MTREE.input"

        rm -f "$MTREEOUTPATH/MTREE.temp_nfiles" \
              "$MTREEOUTPATH/MTREE.temp_particles" \
              "$MTREEOUTPATH/MTREE.temp_prefix"

        echo "Running MergerTree ..."
        "$MTREEBINPATH" < "$MTREEOUTPATH/MTREE.input"

        echo "Finished  : $(date)"
        echo ""

    } 2>&1 | tee -a "$LOGFILE"

done

echo "All setups complete."
echo "Logfiles written under $OUTBASEPATH/<setupdir>/run_mergertree_${DIRECTION}.log"
