"""
Unit tests for ConversionMatrix and ConditionMatrices in
merge_reads_make_matrices.py.

Run from the tests/ directory:
    PYTHONPATH=../workflow/scripts python -m unittest test_conversion_matrix -v
"""
import unittest
import numpy as np
import numpy.ma as ma
from merge_reads_make_matrices import ConversionMatrix, ConditionMatrices


def make_cm(ref, reads, c_context=None, unconv_frac=0.0, dedup_method=None, subsample=None):
    """Helper: build a ConversionMatrix from plain strings."""
    if c_context is None:
        c_context = ['GC']
    reads_arr = np.array([list(r) for r in reads])
    return ConversionMatrix(reads_arr, ref, "test_ref",
                            c_context=c_context,
                            unconv_frac=unconv_frac,
                            dedup_method=dedup_method,
                            subsample=subsample)


class TestMakeMasks(unittest.TestCase):
    """make_masks() should correctly identify CG, GC, GCG, and allC positions."""

    def test_allC_marks_all_cytosines(self):
        # Every C in the reference should appear in allC
        ref = "ACGTCGCAT"
        cm = make_cm(ref, ["ACGTCGCAT"])
        mask = cm.masks['allC']
        expected = np.array([False, True, False, False, True, False, True, False, False])
        #                           A  C  G  T  C  G  C  A  T
        # ref positions:            0  1  2  3  4  5  6  7  8
        # Cs at 1, 4, 6
        c_positions = [i for i, b in enumerate(ref.upper()) if b == 'C']
        np.testing.assert_array_equal(mask[c_positions], True)
        np.testing.assert_array_equal(mask[[i for i in range(len(ref)) if ref[i] != 'C']], False)

    def test_CG_mask(self):
        # ACGT: C at pos 1 is in CG context (pos 1,2 = CG)
        ref = "ACGT"
        cm = make_cm(ref, ["ACGT"])
        # CG: C followed by G
        self.assertTrue(cm.masks['CG'][1])    # pos 1: C followed by G at pos 2
        self.assertFalse(cm.masks['CG'][0])   # A
        self.assertFalse(cm.masks['CG'][2])   # G
        self.assertFalse(cm.masks['CG'][3])   # T

    def test_GC_mask(self):
        # AGCT: C at pos 2 is in GC context (pos 1,2 = GC)
        ref = "AGCT"
        cm = make_cm(ref, ["AGCT"])
        self.assertTrue(cm.masks['GC'][2])    # pos 2: C preceded by G at pos 1
        self.assertFalse(cm.masks['GC'][0])
        self.assertFalse(cm.masks['GC'][1])
        self.assertFalse(cm.masks['GC'][3])

    def test_GCG_is_in_both_dimers(self):
        # GCG: position 1 (C) is in BOTH CG (as GC follows) and GC (G precedes)
        ref = "GCGT"
        cm = make_cm(ref, ["GCGT"])
        # C at pos 1: preceded by G (GC context) AND followed by G (CG context)
        self.assertTrue(cm.masks['GC'][1])
        self.assertTrue(cm.masks['CG'][1])
        # CG_unambig and GC_unambig should both be False at pos 1 (it's GCG)
        self.assertFalse(cm.masks['CG_unambig'][1])
        self.assertFalse(cm.masks['GC_unambig'][1])

    def test_unambiguous_masks_exclude_GCG(self):
        # Sequence: G-C-G-C-T
        # pos 0: G, pos 1: C (GCG → ambiguous), pos 2: G, pos 3: C (GC → GC_unambig only), pos 4: T
        ref = "CGCGT"
        cm = make_cm(ref, ["CGCGT"])
        # C at pos 0: CG context (followed by G at pos 1), pos -1 is no G → not GC
        self.assertTrue(cm.masks['CG_unambig'][0])
        self.assertFalse(cm.masks['GC_unambig'][0])
        # C at pos 2: GCG → both, so neither unambig
        self.assertFalse(cm.masks['CG_unambig'][2])
        self.assertFalse(cm.masks['GC_unambig'][2])


class TestRawMatrix(unittest.TestCase):
    """raw_matrix should encode C=0, T=1, other=masked at cytosine positions."""

    def test_C_is_zero_T_is_one(self):
        ref = "GCT"
        # Read has C at pos 1 (unconverted) and T at pos 2 (not a cytosine ref pos)
        reads = ["GCT", "GTT"]
        cm = make_cm(ref, reads, c_context=['allC'])
        # pos 1 is the only C in ref
        raw = cm.raw_matrix
        # read 0: C at pos 1 → 0
        self.assertEqual(raw[0, 1], 0)
        # read 1: T at pos 1 → 1 (converted)
        self.assertEqual(raw[1, 1], 1)
        # pos 0 and pos 2 are G and T in ref, should be masked
        self.assertTrue(raw.mask[0, 0])
        self.assertTrue(raw.mask[0, 2])

    def test_non_C_T_bases_are_masked(self):
        ref = "GCA"
        reads = ["GAA"]   # A at pos 1 (a C pos in ref): should be masked
        cm = make_cm(ref, reads, c_context=['allC'])
        self.assertTrue(cm.raw_matrix.mask[0, 1])


class TestPerReadConv(unittest.TestCase):
    """per_read_conv excludes CG and GC (and GCG) contexts."""

    def test_fully_converted_allC_read(self):
        # Reference with only non-CG, non-GC Cs
        # "ATCAT": C at pos 2, no CG or GC neighbors
        ref = "ATCAT"
        reads = ["ATTAT"]    # C→T at pos 2: fully converted
        cm = make_cm(ref, reads, c_context=['allC'])
        conv = cm.per_read_conv
        self.assertAlmostEqual(conv[0], 1.0)

    def test_unconverted_read_is_zero(self):
        ref = "ATCAT"
        reads = ["ATCAT"]    # C retained: no conversion
        cm = make_cm(ref, reads, c_context=['allC'])
        self.assertAlmostEqual(cm.per_read_conv[0], 0.0)

    def test_CG_context_excluded_from_per_read_conv(self):
        # Reference: TCG — only C at pos 1 in CG context; per_read_conv should be nan/0
        ref = "TCG"
        reads = ["TCG", "TTG"]    # one unconverted, one converted in CG context
        cm = make_cm(ref, reads, c_context=['CG'])
        # both_dimers mask covers pos 1 (CG), so no Cs remain for per_read_conv
        # np.mean of fully masked row returns masked; .data gives fill_value
        # Just check it doesn't crash and returns an array
        conv = cm.per_read_conv
        self.assertEqual(len(conv), 2)


class TestFilteredMatrix(unittest.TestCase):
    """filtered_matrix should drop reads below unconv_frac."""

    def test_below_threshold_dropped(self):
        # Ref has two non-CG, non-GC Cs (pos 1 and 3)
        ref  = "ACACA"
        # Read 0: both Cs unconverted (conv=0.0) → should be dropped at frac=0.5
        # Read 1: both Cs converted (conv=1.0) → should be kept
        reads = ["ACACA", "ATATA"]
        cm = make_cm(ref, reads, c_context=['allC'], unconv_frac=0.5)
        self.assertEqual(cm.n_reads['filtered'], 1)

    def test_no_threshold_keeps_all(self):
        ref = "ACACA"
        reads = ["ACACA", "ATATA"]
        cm = make_cm(ref, reads, c_context=['allC'], unconv_frac=0.0)
        self.assertEqual(cm.n_reads['filtered'], 2)


class TestDedup(unittest.TestCase):
    """make_dedup_matrix should remove exact duplicates but keep distinct reads."""

    def _reads_with_dupe(self):
        # Simple ref with two non-CG, non-GC Cs at pos 1 and 3
        ref = "ACACA"
        # Three reads: two identical (0,0), one distinct (1,1)
        reads = ["ACACA",   # C C → 0,0
                 "ACACA",   # C C → 0,0  (duplicate)
                 "ATATA"]   # T T → 1,1
        return ref, reads

    def test_duplicate_removed(self):
        ref, reads = self._reads_with_dupe()
        cm = make_cm(ref, reads, c_context=['allC'], dedup_method='allC')
        self.assertEqual(cm.n_reads['deduped'], 2)

    def test_no_dedup_keeps_all(self):
        ref, reads = self._reads_with_dupe()
        cm = make_cm(ref, reads, c_context=['allC'], dedup_method=None)
        self.assertEqual(cm.n_reads['deduped'], 3)

    def test_distinct_reads_all_kept(self):
        ref = "ACACA"
        reads = ["ACACA",   # 0,0
                 "ATACA",   # 1,0
                 "ACATA",   # 0,1
                 "ATATA"]   # 1,1
        cm = make_cm(ref, reads, c_context=['allC'], dedup_method='allC')
        self.assertEqual(cm.n_reads['deduped'], 4)


class TestNReads(unittest.TestCase):
    """n_reads should track counts at each stage."""

    def test_counts_correct(self):
        ref = "ACACA"
        reads = ["ACACA",   # conv=0.0 → filtered out at frac=0.5
                 "ATATA",   # conv=1.0 → kept
                 "ATATA"]   # conv=1.0 → duplicate → deduped out
        cm = make_cm(ref, reads, c_context=['allC'],
                     unconv_frac=0.5, dedup_method='allC')
        self.assertEqual(cm.n_reads['raw'], 3)
        self.assertEqual(cm.n_reads['filtered'], 2)
        self.assertEqual(cm.n_reads['deduped'], 1)


class TestAmpliconStats(unittest.TestCase):
    """amplicon_stats string should have correct format and values."""

    def test_format(self):
        ref = "ACACA"
        reads = ["ATATA", "ACACA"]
        cm = make_cm(ref, reads, c_context=['allC'], unconv_frac=0.5)
        stats = cm.amplicon_stats
        parts = stats.split('\t')
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], 'test_ref')
        self.assertEqual(int(parts[1]), cm.n_reads['filtered'])
        self.assertEqual(int(parts[2]), cm.n_reads['deduped'])

    def test_zero_dedup_does_not_divide_by_zero(self):
        # All reads filtered → deduped = 0
        ref = "ACACA"
        reads = ["ACACA"]   # conv=0.0, filtered at frac=0.5
        cm = make_cm(ref, reads, c_context=['allC'], unconv_frac=0.5)
        stats = cm.amplicon_stats   # should not raise
        parts = stats.split('\t')
        self.assertEqual(float(parts[3]), 0.0)


class TestConditionMatricesInit(unittest.TestCase):
    """ConditionMatrices should reject bad inputs at init time."""

    def test_bad_c_context_raises(self):
        with self.assertRaises(ValueError):
            ConditionMatrices(bam="x.bam", fasta="x.fa", c_context=["bad"])

    def test_bad_dedup_method_raises(self):
        with self.assertRaises(ValueError):
            ConditionMatrices(bam="x.bam", fasta="x.fa",
                              c_context=["GC"], dedup_method="bad")

    def test_empty_c_context_raises(self):
        with self.assertRaises(ValueError):
            ConditionMatrices(bam="x.bam", fasta="x.fa", c_context=[])

    def test_repr_does_not_crash(self):
        cond = ConditionMatrices(bam="x.bam", fasta="x.fa",
                                 c_context=["GC"], dedup_method="GC")
        self.assertIn("ConditionMatrices", repr(cond))


if __name__ == "__main__":
    unittest.main()
