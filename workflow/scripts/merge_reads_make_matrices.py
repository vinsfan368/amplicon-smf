import pysam
from argparse import ArgumentParser
import numpy as np


# Abort if we see more unpaired reads than this while streaming the
# BAM file. If you hit this limit, rerun with a name-sorted BAM!
MAX_CACHE = 1_000_000


def parse_args():
    parser = ArgumentParser(description="Compute C conversion matrices from a BAM file and its reference")
    parser.add_argument("bam_path", help="Path to BAM file to parse")
    parser.add_argument("fa_path", help="Path to reference FASTA used for alignment")

    return parser.parse_args()


def merge_mates(mate1: pysam.AlignedSegment,
                mate2: pysam.AlignedSegment,
                ref_len: int) -> np.ndarray:
    """
    Given two mated AlignedSegments, merge in reference coordinates:
    1. All unsequenced bases in the reference are 'N'
    2. If only one read covers a position and its base quality
       is 0, that position is assigned 'N'
    3. If only one read covers a position and its base quality
       is > 0, the position is assigned that read's base
    4. If the reads overlap and have different quality scores, the
       position is assigned the base with the greater quality score
    5. If the reads overlap and have the same quality score and the
       same base at a position, that position is assigned that base
    6. If the reads overlap and have the same quality score but
       different bases at a position, that position is assigned 'N'
    """
    # Make arrays to store bases and quality scores in ref coordinates
    seqs = np.full((2, ref_len), "N")
    quals = np.zeros_like(seqs, dtype=int)
    for i, mate in enumerate((mate1, mate2)):
        # Read qualities for this mate, if missing, leave as zeros
        q_quals = mate.query_qualities
        if q_quals is None:
            continue
        # Otherwise fill a row of the array
        q_seq = mate.query_sequence
        for qpos, rpos in mate.get_aligned_pairs(matches_only=True):
            seqs[i, rpos] = q_seq[qpos]
            quals[i, rpos] = q_quals[qpos]
    
    # Merge reads: first, assign mapped positions in cases where one
    # quality score is higher than the other. This fulfills
    # criteria 1-4 because of how the arrays were initialized.
    merge = np.full(ref_len, "N")
    merge[quals[0] > quals[1]] = seqs[0, (quals[0] > quals[1])]
    merge[quals[1] > quals[0]] = seqs[1, (quals[1] > quals[0])]

    # Then, set positions where quality score is equal but greater
    # than zero and bases agree to that consensus base, fulfilling
    # criteria 5. Number 6 is fulfilled by not changing it from an N
    mask = ((quals[0] == quals[1]) &
            (seqs[0] == seqs[1]) &
            (quals[0] > 0))
    merge[mask] = seqs[0, mask]

    return merge


def qc_and_merge_mates(bam_path: str) -> tuple[dict[str, np.ndarray], int]:
    """
    Stream a possibly unsorted BAM file. Look at SAM flags to QC, and
    for mates that pass, merge sequences in reference coordinates.

    output
    ------
    return  :   dict[str, np.ndarray], ndarrays of shape
                (n_pairs, len(reference)) containing merged sequences
                in reference coordinates, keyed by reference name
            :   int, number of alignments (not pairs) that had
                QC-failing flags, or didn't align to the same
                contig as its mate (both fail), or remained in
                the cache after the BAM file was streamed
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    n_qcfail = 0
    cache = {}  # store alignments whose mates we haven't seen yet
    arrs = {}   # refname-keyed dict of np.array, sequence of each merged read
    for aln in bam:
        # Look at flags to QC this alignment
        if aln.is_unmapped or \
           (not aln.is_proper_pair) or \
           aln.is_qcfail or \
           aln.is_secondary or \
           aln.is_supplementary:
            n_qcfail += 1
            continue
        
        # Store alignment in cache if not full and if mate is not already there
        if aln.query_name not in cache:
            if len(cache) >= MAX_CACHE:
                raise RuntimeError("Max cache size reached searching for "
                                   "mates, try again with a name-sorted BAM.")
            cache[aln.query_name] = aln
            continue
        
        # Otherwise we found mate, remove it from the cache
        mate = cache.pop(aln.query_name)
        # Sanity check: mates mapped to the same reference
        ref_name = aln.reference_name
        if ref_name != mate.reference_name:
            n_qcfail += 2
            continue

        # Merge seqs and store
        merge_seq = merge_mates(mate1=aln,
                                mate2=mate,
                                ref_len=bam.lengths[aln.reference_id])
        arrs.setdefault(ref_name, []).append(merge_seq)
    
    # return vstacked arrays and qcfailed + remaining cache length
    return ({ref: np.vstack(reads) for ref, reads in arrs.items()},
            n_qcfail + len(cache))


def main():
    args = parse_args()
    merged_reads, n_qcfail = qc_and_merge_mates(args.bam_path)
    
    fa = pysam.FastaFile(args.fa_path)



if __name__ == "__main__":
    main()
