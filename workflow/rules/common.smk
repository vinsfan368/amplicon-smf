import sys
import pandas as pd


##### load config and sample sheets #####

samplesheet = pd.read_csv(config["samples"], sep="\t").set_index("sample_name", drop=False)

# Optional samplesheet columns with defaults
if 'include_cpg' not in samplesheet.columns:
    samplesheet['include_cpg'] = False
if 'deaminase' not in samplesheet.columns:
    samplesheet['deaminase'] = False
if 'dedup_on' not in samplesheet.columns:
    samplesheet['dedup_on'] = 'c_type'

# Backwards-compatibility warnings: parameters from the bwameth-based pipeline
# that are no longer used. The pipeline ignores them but warns so users know
# their config/samplesheet has stale entries.
_DEPRECATED_CONFIG_KEYS = {
    'alignment_score_fraction': 'read alignment is now handled by bismark',
    'alignment_length_fraction': 'read alignment is now handled by bismark',
    'read1_length': 'read lengths are no longer needed; bismark handles alignment',
    'read2_length': 'read lengths are no longer needed; bismark handles alignment',
}
for _key, _reason in _DEPRECATED_CONFIG_KEYS.items():
    if _key in config:
        print(f"WARNING: config key '{_key}' is no longer used ({_reason}). "
              f"Remove it from your config file to silence this warning.",
              file=sys.stderr)

_DEPRECATED_SAMPLE_COLS = {
    'filter_contigs': 'contig filtering is no longer performed',
    'bottom_strand': 'strand orientation is handled by passing R2 as -1 to bismark',
    'ignore_bounds': 'alignment bounds checking is no longer performed',
    'no_endog_meth': 'endogenous methylation masking is now controlled via c_context',
}
for _col, _reason in _DEPRECATED_SAMPLE_COLS.items():
    if _col in samplesheet.columns:
        print(f"WARNING: samplesheet column '{_col}' is no longer used ({_reason}). "
              f"Remove it from your samplesheet to silence this warning.",
              file=sys.stderr)


def all_input(wildcards):
    wanted_input = []

    for sample in samplesheet['sample_name']:
        experiment = samplesheet.loc[sample, 'experiment']

        wanted_input.append('results/{e}/{s}/{s}.amplicon_stats.txt'.format(e=experiment, s=sample))
        wanted_input.append('results/{e}/plots/{s}.conversion_and_coverage.pdf'.format(e=experiment, s=sample))
        wanted_input.append('results/{e}/plots/{s}.single_molecules.pdf'.format(e=experiment, s=sample))
        wanted_input.append('results/{e}/plots/{s}.nuc_len_qc_plots.pdf'.format(e=experiment, s=sample))
        wanted_input.append('results/qc/fastqc/fastqc.txt')

    return wanted_input
