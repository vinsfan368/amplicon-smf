import os

WORKFLOW_DIR = workflow.basedir
SCRIPTS_DIR = os.path.join(WORKFLOW_DIR, "scripts")
ENV_DIR = os.path.join(WORKFLOW_DIR, "rules", "envs")


rule fastqc:
    input:
        lambda wildcards: samplesheet[['fastq_R1','fastq_R2']].to_numpy().flatten().tolist()
    output:
        'results/qc/fastqc/fastqc.txt'
    conda:
        os.path.join(ENV_DIR, "python3_v7.yaml")
    params:
        outdir = 'results/qc/fastqc'
    shell:
        'fastqc -o {params.outdir} -f fastq {input}; touch {output}'


def get_amplicon(wildcards):
    return samplesheet.loc[wildcards.sample, "amplicon_fa"]


rule wrap_fasta_with_NNN:
    """
    Reverse complement, uppercase, and pad each contig in the amplicon FASTA
    with NNN on each end.

    RC is required because reads are sequenced from the bottom strand: all
    paired-end reads align as OB in the original reference frame, so Cs in
    the original reference never appear as Ts in the reads.  RC'ing puts the
    cytosines of interest on the top strand, so bismark reports them as OT
    reads and merge_reads_make_matrices.py sees C (protected) or T (accessible)
    at the correct positions.

    The NNN padding prevents bismark from producing conversion artefacts at
    contig boundaries without changing the coordinate system for any scored
    cytosine (Ns are never scored).
    """
    input:
        get_amplicon
    output:
        'results/{experiment}/{sample}/tmp/{sample}.amplicon.NNN.fa'
    run:
        _comp = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
        def rc(seq):
            return seq.translate(_comp)[::-1]

        def wrap_rc_upper(lines, n_pad=3):
            header = None
            seq_chunks = []
            pad = "N" * n_pad
            for line in lines:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if header is not None:
                        yield header
                        yield pad + rc("".join(seq_chunks)).upper() + pad
                    header = line
                    seq_chunks = []
                else:
                    seq_chunks.append(line)
            if header is not None:
                yield header
                yield pad + rc("".join(seq_chunks)).upper() + pad

        with open(input[0]) as inp, open(output[0], "w") as out:
            for line in wrap_rc_upper(inp):
                out.write(line + "\n")


rule bismark_genome_preparation:
    """
    Copy the NNN-padded FASTA into a genome folder and build the
    bismark bisulfite genome index (CT and GA converted genomes).
    """
    input:
        fa='results/{experiment}/{sample}/tmp/{sample}.amplicon.NNN.fa'
    output:
        sentinel='results/{experiment}/{sample}/tmp/bismark_genome/.genome_prepared'
    params:
        genome_dir='results/{experiment}/{sample}/tmp/bismark_genome'
    conda:
        os.path.join(ENV_DIR, "python3_v7.yaml")
    shell:
        'mkdir -p {params.genome_dir} && '
        'cp {input.fa} {params.genome_dir}/ && '
        'bismark_genome_preparation {params.genome_dir} && '
        'touch {output.sentinel}'


rule bismark_align:
    """
    Paired-end bismark alignment. R2 is passed as -1 and R1 as -2 because
    reads are sequenced from the bottom strand of the amplicon.
    """
    input:
        sentinel='results/{experiment}/{sample}/tmp/bismark_genome/.genome_prepared',
        r1=lambda wc: samplesheet.loc[wc.sample, 'fastq_R1'],
        r2=lambda wc: samplesheet.loc[wc.sample, 'fastq_R2']
    output:
        bam='results/{experiment}/{sample}/{sample}_pe.bam'
    params:
        genome_dir='results/{experiment}/{sample}/tmp/bismark_genome',
        outdir='results/{experiment}/{sample}',
        parallel=config.get('threads', 10)
    log:
        'results/{experiment}/{sample}/{sample}.bismark.log'
    conda:
        os.path.join(ENV_DIR, "python3_v7.yaml")
    shell:
        'bismark {params.genome_dir} '
        '-1 {input.r2} -2 {input.r1} '
        '--maxins 700 -p {params.parallel} --local '
        '--output_dir {params.outdir} '
        '--basename {wildcards.sample} '
        '> {log} 2>&1'


def get_c_context(wildcards):
    """Return space-separated c_context list for merge_reads_make_matrices.py."""
    if samplesheet.loc[wildcards.sample, 'deaminase']:
        return 'allC'
    elif samplesheet.loc[wildcards.sample, 'include_cpg']:
        return 'GC CG'
    else:
        return 'GC'


def get_dedup_method(wildcards):
    """Map samplesheet dedup_on + experiment type to a dedup_method arg."""
    dedup_on = samplesheet.loc[wildcards.sample, 'dedup_on']
    if dedup_on == 'allC':
        return 'allC'
    # 'c_type': derive from the c_context for this sample
    if samplesheet.loc[wildcards.sample, 'deaminase']:
        return 'allC'
    elif samplesheet.loc[wildcards.sample, 'include_cpg']:
        return 'both_dimers'
    else:
        return 'GC'


rule make_conversion_matrices:
    input:
        bam='results/{experiment}/{sample}/{sample}_pe.bam',
        fa='results/{experiment}/{sample}/tmp/{sample}.amplicon.NNN.fa'
    output:
        stats='results/{experiment}/{sample}/{sample}.amplicon_stats.txt',
        conversion_pdf='results/{experiment}/plots/{sample}.conversion_and_coverage.pdf',
        sm_pdf='results/{experiment}/plots/{sample}.single_molecules.pdf'
    params:
        matrix_prefix='results/{experiment}/{sample}/matrices/{sample}.',
        c_context=get_c_context,
        dedup_method=get_dedup_method,
        unconv_frac=lambda wc: config.get('unconverted_frac', 0.85),
        subsample=lambda wc: config.get('subsample', 1000),
        invert=lambda wc: '--invert' if samplesheet.loc[wc.sample, 'deaminase'] else '',
        script=os.path.join(SCRIPTS_DIR, "merge_reads_make_matrices.py")
    conda:
        os.path.join(ENV_DIR, "python3_v7.yaml")
    shell:
        'mkdir -p results/{wildcards.experiment}/{wildcards.sample}/matrices; '
        'python {params.script} {input.bam} {input.fa} {output.stats} {params.matrix_prefix} '
        '--c_context {params.c_context} '
        '--unconv_frac {params.unconv_frac} '
        '--dedup_method {params.dedup_method} '
        '--subsample {params.subsample} '
        '--conversion_pdf {output.conversion_pdf} '
        '--sm_pdf {output.sm_pdf} '
        '{params.invert}'


rule plot_nuc_qc:
    input:
        fa='results/{experiment}/{sample}/tmp/{sample}.amplicon.NNN.fa',
        stats='results/{experiment}/{sample}/{sample}.amplicon_stats.txt'
    output:
        plot='results/{experiment}/plots/{sample}.nuc_len_qc_plots.pdf',
        stats='results/{experiment}/{sample}/stats/{sample}.nuc_len_qc.stats.txt'
    params:
        matrix_path='results/{experiment}/{sample}/matrices/{sample}',
        script=os.path.join(SCRIPTS_DIR, "plot_nucleosome_qc.py")
    conda:
        os.path.join(ENV_DIR, "python3_v7.yaml")
    shell:
        'python {params.script} --input {params.matrix_path} --amplicon {input.fa} '
        '--plot {output.plot} --results {output.stats} --gmm'
