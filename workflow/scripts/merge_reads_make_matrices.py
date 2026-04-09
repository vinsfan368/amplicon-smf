# CLI
from argparse import ArgumentParser
import os

# Read FASTA and BAM
import pysam

# Arrays, masked arrays
import numpy as np
import numpy.ma as ma

# Save objects for later
import pickle

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as grd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns       # Confidence intervals on conversion barplot

# Cluster single molecules
from sklearn.cluster import KMeans


# Abort if we see more unpaired reads than this while streaming the BAM.
MAX_CACHE = 1_000_000


def parse_args():
    parser = ArgumentParser(description="Compute C conversion matrices from a BAM file and its reference")
    parser.add_argument("bam_path", help="Path to BAM file to parse")
    parser.add_argument("fa_path", help="Path to reference FASTA used for alignment")
    parser.add_argument("stats_out", help="Path to write per-amplicon stats TSV")
    parser.add_argument("matrix_prefix",
                        help="Filename prefix for output matrix files, "
                             "e.g. 'results/exp/sample/matrices/sample.'")
    parser.add_argument("--c_context", nargs="+", default=["GC"],
                        choices=["CG", "GC", "allC"],
                        help="C contexts to score (space-separated list)")
    parser.add_argument("--unconv_frac", type=float, default=0.0,
                        help="Minimum per-read C conversion fraction to keep a "
                             "read. Conversion is computed on non-CG, non-GC Cs.")
    parser.add_argument("--dedup_method", default=None,
                        choices=["CG", "GC", "both_dimers", "allC", "none"],
                        help="C context to use for deduplication. "
                             "Pass 'none' to skip deduplication.")
    parser.add_argument("--min_mapq", type=int, default=0,
                        help="Minimum mapping quality; reads below this are dropped.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample this many reads before clustering/plotting.")
    parser.add_argument("--conversion_pdf", default=None,
                        help="Path to save conversion + coverage line plots PDF.")
    parser.add_argument("--sm_pdf", default=None,
                        help="Path to save single-molecule heatmap PDF.")
    parser.add_argument("--invert", action="store_true",
                        help="Invert conversion coloring (plot protection instead "
                             "of accessibility).")
    parser.add_argument("--compressed", action="store_true",
                        help="In single-molecule plots, compress to only show "
                             "scored Cs (ignore real distances).")
    return parser.parse_args()


class ConversionMatrix:
    """
    Object storing alignments to a single reference DNA sequence.
    1. Computes a per-read conversion fraction (see `unconv_frac` below)
    2. Uses (1) to filter reads below `unconv_frac`
    3. De-duplicates these filtered reads (see `dedup_method` below)
    4. Scores the requested cytosines (see `c_context` below)

    Also caches number of reads and coverage (number of informative
    Cs and Ts at each position) retained after each filtering step.

    init
    ----
    base_matrix     :   np.ndarray of shape (n_reads, len(reference)),
                        the base identities of each read aligned
                        to the reference in the reference frame.
    reference_seq   :   str, reference sequence these reads align to.
    reference_name  :   str, name of the reference seq reads align to.
    c_context       :   list[str] whose elements must be in ['CG', 'GC',
                        'allC'], which Cs to score for position-wise
                        conversion percentage and C/T coverage.
                        Scoring conversion and coverage happens
                        after the two filters below. 'GC' and 'CG'
                        do NOT consider Cs in GCG contexts.
    unconv_frac     :   float, default 0. Do not consider molecules
                        with less than this C conversion fraction in
                        non-CG and non-GC (and non-GCG) contexts.
                        (Experimentally, these molecules may originate
                        from unpermeabilized nuclei.) This filter
                        is applied first to base_matrix.
    dedup_method    :   str that must be in ['CG', 'GC', 'both_dimers',
                        'allC', None], default None, which C contexts
                        to consider when identifying duplicate
                        molecules. *Here, 'CG', 'GC', and 'both_dimers'
                        all consider GCG to identify duplicates.* This
                        filter is applied second to base_matrix.
    properties
    ----------
    masks           :   str-keyed dict of np.ndarray,
                        boolean masks to use for computations
    raw_matrix      :   np.ma.MaskedArray where Cs are represented
                        by 0s, Ts by 1s, and other bases are masked
                        with a default value of -1. All non-C positions
                        in the reference also have a value of -1. All
                        MaskedArrays in this class take this form.
    per_read_conv   :   np.ndarray of float, read-wise fraction of
                        converted Cs. For this calculation, we only
                        consider non-GC and non-CG (and non-GCG) Cs.
    filtered_matrix :   np.ma.MaskedArray, contains all reads of
                        `raw_matrix` where `per_read_conv` is more
                        than `unconv_frac`.
    deduped_matrix  :   np.ma.MaskedArray, `filtered_matrix` containing
                        only unique reads according to `dedup_method`.
    coverage_arrays :   dict of np.ndarray keyed with 'raw', 'filtered',
                        and 'deduped'. Each ndarray are `reference_seq`-
                        indexed, and at each cytosine, tallies how many
                        informative bases (Cs or Ts) are at that position.
    n_reads         :   dict of np.ndarray keyed with 'raw, 'filtered',
                        and 'deduped', describing how many reads
                        survived past each filtering step.
    scored_matrices   :   dict of np.ma.MaskedArray, keyed with every
                        list element in `c_context`. These are the
                        scored matrices containing only the requested
                        C contexts.
    """
    def __init__(self,
                 base_matrix: np.ndarray,
                 reference_seq: str,
                 reference_name: str,
                 c_context: list[str],
                 unconv_frac: float=0.0,
                 dedup_method: str=None,
                 subsample: int=None):
        # Sanitize inputs and store
        if not isinstance(reference_seq, np.ndarray):
            reference_seq = np.array(list(reference_seq))
        self.reference_seq = np.char.upper(reference_seq)

        if not isinstance(c_context, list):
            c_context = [c_context]
        if len(c_context) == 0:
            raise ValueError(f'ConversionMatrix.__init__: empty ',
                             f'list passed for c_context')
        for ele in c_context:
            if ele not in ["CG", "GC", "allC"]:
                raise ValueError(f'All elements of c_context must be one of ',
                                f'"CG", "GC", or "allC", {c_context} passed.')
        self.c_context = c_context

        if dedup_method is None:
            dedup_method = False
        elif dedup_method not in ["CG", "GC", "both_dimers", "allC"]:
            raise ValueError(f'dedup must be one of "CG", "GC", "both_dimers"',
                             f', or "allC", {dedup_method} passed.')
        self.dedup = dedup_method

        # Conceivable that matrix passed is not 2D
        assert isinstance(base_matrix, np.ndarray), \
            "ConversionMatrix.__init__: base_matrix passed is not a numpy array!"
        if np.ndim(base_matrix) > 2:
            raise ValueError(f"ConversionMatrix: passed base_matrix has shape ",
                             f"{base_matrix}.shape but must have two dims!")
        elif np.ndim(base_matrix) == 1:
            base_matrix = base_matrix[np.newaxis, :]
        n_reads, ref_len = base_matrix.shape
        if len(reference_seq) != ref_len:
            msg = (f"ConversionMatrix: Reads have alignment length "
                   f"{ref_len} but reference is {len(reference_seq)}, a "
                   f"different FASTA file was probably used for alignment!")
            raise RuntimeError(msg)

        self.base_matrix = np.char.upper(base_matrix)
        self.reference_name = reference_name
        self.unconv_frac = unconv_frac
        self.ref_len = ref_len
        self.subsample = subsample

    def __repr__(self):
        """String representation of ConversionMatrix."""
        return "ConversionMatrix object:\n    {}".format("\n    ".join([
            f"{attr : <16}:    {getattr(self, attr)}"
            for attr in ['reference_name', 'ref_len', 'c_context',
                         'dedup', "unconv_frac"]]))

    ##############
    # PROPERTIES #
    ##############
    @property
    def masks(self) -> dict[str, np.ndarray]:
        """Make and cache boolean masks."""
        if not hasattr(self, "_masks"):
            self._masks = self.make_masks()
        return self._masks

    @property
    def raw_matrix(self) -> ma.array:
        """
        Make an int MaskedArray where Cs are 0 (unconverted), Ts are 1
        (converted), and other bases in the reads are masked (default
        -1), only for positions where the reference sequence is a C.
        """
        if not hasattr(self, "_raw_matrix"):
            arr = np.full_like(self.base_matrix, fill_value=-1, dtype=int)
            arr[(self.base_matrix == "C") & self.masks["allC"][None, :]] = 0
            arr[(self.base_matrix == "T") & self.masks["allC"][None, :]] = 1
            self._raw_matrix = ma.array(data=arr,
                                        dtype=int,
                                        mask=(arr < 0),
                                        fill_value=-1)
        return self._raw_matrix

    @property
    def per_read_conv(self) -> np.ndarray:
        """
        Conversion fraction (number of Ts / (Cs + Ts)) for each read.
        This will be used to filter reads below `self.unconv_frac`.
        This property and filtering invariably uses all Cs not in
        a CG or GC context to decide which reads to keep.
        """
        if not hasattr(self, "_per_read_conv"):
            marr = self.raw_matrix.copy()   # copy is necessary
            # Confusingly, mask == True means data is invalid, so
            # C in either CG or GC contexts should not be considered
            marr[:, self.masks["both_dimers"]] = ma.masked
            self._per_read_conv = np.mean(marr, axis=1).data
        return self._per_read_conv

    @property
    def filtered_matrix(self) -> ma.array:
        """Raw matrix filtered to exclude reads
        below the `unconv_frac` threshold."""
        if not hasattr(self, "_filtered_matrix"):
            if self.unconv_frac > 0:
                passes = self.per_read_conv > self.unconv_frac
                self._filtered_matrix = self.raw_matrix[passes, :]
            else:
                self._filtered_matrix = self.raw_matrix.copy()
        return self._filtered_matrix

    @property
    def deduped_matrix(self) -> ma.array:
        """
        Filtered matrix removing duplicates to exclude reads according
        to `dedup_method`, which describes which C contexts to consider
        when evaluating uniqueness. A read is culled iff all covered
        (C or T) positions in the read matches another read.
        """
        if not hasattr(self, "_deduped_matrix"):
            if self.dedup:
                self._deduped_matrix = self.make_dedup_matrix()
            else:
                self._deduped_matrix = self.filtered_matrix.copy()
        return self._deduped_matrix

    @property
    def coverage_arrays(self) -> dict[str, np.ndarray]:
        """Number of informative bases at all C positions in
        the reference sequence after various filtering steps."""
        if not hasattr(self, "_coverage_arrays"):
            cov = {}
            cov['raw'] = np.sum(~self.raw_matrix.mask, axis=0)
            cov['filtered'] = np.sum(~self.filtered_matrix.mask, axis=0)
            cov['deduped'] = np.sum(~self.deduped_matrix.mask, axis=0)
            self._coverage_arrays = cov
        return self._coverage_arrays

    @property
    def plotting_matrices(self) -> dict[str, list]:
        """
        Convenience property for plotting conversion. Cs outside of requested
        contexts are masked and only indices for Cs to plot are stored.
        """
        if not hasattr(self, "_plotting_matrices"):
            matrices = {}
            # deduped_matrix contains all Cs, so allC is just a copy
            if 'allC' in self.c_context:
                allC_matrix = self.deduped_matrix.copy()
                x = np.where(self.masks['allC'])[0]
                y = np.mean(allC_matrix[:, x], axis=0)
                matrices['allC'] = (x, y)
            # Unambiguous C in CG context
            if 'CG' in self.c_context:
                CG_matrix = self.deduped_matrix.copy()
                CG_matrix[:, ~self.masks['CG_unambig']] = ma.masked
                x = np.where(self.masks['CG_unambig'])[0]
                y = np.mean(CG_matrix[:, x], axis=0)
                matrices['CG_unambig'] = (x, y)
            # Unambiguous C in GC context
            if 'GC' in self.c_context:
                GC_matrix = self.deduped_matrix.copy()
                GC_matrix[:, ~self.masks['GC_unambig']] = ma.masked
                x = np.where(self.masks['GC_unambig'])[0]
                y = np.mean(GC_matrix[:, x], axis=0)
                matrices['GC_unambig'] = (x, y)
            self._plotting_matrices = matrices
        return self._plotting_matrices

    @property
    def n_reads(self) -> dict[str, int]:
        """Number of reads that survived various filtering steps."""
        if not hasattr(self, "_n_reads"):
            n_reads = {}
            n_reads['raw'] = self.raw_matrix.shape[0]
            n_reads['filtered'] = self.filtered_matrix.shape[0]
            n_reads['deduped'] = self.deduped_matrix.shape[0]
            self._n_reads = n_reads
        return self._n_reads

    @property
    def subsampled_matrix(self) -> ma.array:
        """Subsampled deduped matrix."""
        if not hasattr(self, "_subsampled_matrix"):
            deduped = self.deduped_matrix
            if (self.subsample is None or
                self.subsample > self.n_reads['deduped']):
                # If no subsample or fewer reads than subsample, just copy
                self._subsampled_matrix = deduped.copy()
                return self._subsampled_matrix

            # Else sample some read indices without replacement
            rng = np.random.default_rng()
            indices = rng.choice(deduped.shape[0],
                                 size=self.subsample,
                                 replace=False)
            self._subsampled_matrix = deduped[indices].copy()
        return self._subsampled_matrix

    @property
    def clustered_matrix(self) -> ma.array:
        """Clustered subsampled, deduped matrix."""
        if not hasattr(self, "_clustered_matrix"):
            # Don't attempt cluster if there's one item
            subsampled = self.subsampled_matrix.copy()
            n_aligns, n_cytosines = subsampled.shape
            # Don't attempt clustering with too few reads
            if n_aligns < 5:
                self._clustered_matrix = subsampled
                return self._clustered_matrix

            # Cluster with KMeans, sort by cluster ID then by distance
            KM = KMeans(n_clusters=4)
            labels = KM.fit_predict(subsampled.data)
            dists = KM.transform(subsampled).min(axis=1)
            order = np.lexsort((dists, labels))
            self._clustered_matrix = subsampled[order].copy()
        return self._clustered_matrix

    @property
    def amplicon_stats(self) -> str:
        """Tab-separated string: amplicon name, filtered reads, deduped
        reads, and ratio of filtered to deduped (reads per unique state)."""
        if not hasattr(self, "_amplicon_stats"):
            n_filt = self.n_reads['filtered']
            n_dedup = self.n_reads['deduped']
            s = (self.reference_name,
                 n_filt,
                 n_dedup,
                 n_filt / n_dedup if n_dedup != 0 else 0.0)
            self._amplicon_stats = "{}\t{}\t{}\t{:.2f}".format(*s)
        return self._amplicon_stats

    ###########
    # METHODS #
    ###########
    def make_masks(self) -> dict[str, np.ndarray]:
        """Make all boolean masks to cache for calculations."""
        ref = self.reference_seq

        mask_CG = np.zeros(len(ref), dtype=bool)
        mask_CG[:-1] = (ref[:-1] == "C") & (ref[1:] == "G")
        mask_GC = np.zeros(len(ref), dtype=bool)
        mask_GC[1:] = (ref[:-1] == "G") & (ref[1:] == "C")

        return {"allC": ref == "C",
                "CG": mask_CG,
                "GC": mask_GC,
                "CG_unambig": mask_CG & ~mask_GC,
                "GC_unambig": mask_GC & ~mask_CG,
                "both_dimers": mask_CG | mask_GC}

    def make_dedup_matrix(self) -> ma.array:
        """
        Dedup filtered matrix with logic `self.dedup`. Cull
        reads where all covered (C or T) positions in the
        requested context are identical to those of another
        read. Uses trie recursion logic, iterating through the
        filtered reads and making comparisons only as necessary.
        """
        def is_redundant(read: np.ndarray,
                         node: dict,
                         depth: int=0) -> bool:
            """Recursion to figure out if the current read is identical
            to any of the reads we've seen by traversing the trie."""
            # Default case: if we've reached the end of read and found
            # a match at every position on the branch, read is a dupe.
            if depth == len(read):
                return True

            # If this position is masked, continue at depth + 1
            val = read[depth]
            if val < 0:
                for branch in node:
                    # If any branch is identical, break and return True
                    if is_redundant(read, node[branch], depth + 1):
                        return True
                # Else this is a unique read, return False
                return False

            # Else position not masked, just follow one of C or T branch
            else:
                if val in node:
                    return is_redundant(read, node[val], depth + 1)
                # If no branch matches current base, read is not dupe
                return False

        def insert_read(read: np.ndarray, trie: dict):
            """Add a read to the trie."""
            node = trie
            for i in range(len(read)):
                val = read[i]
                if val not in node:
                    node[val] = {}
                node = node[val]

        # Mask `self.filtered_matrix` with the `self.dedup` logic
        mask = self.masks[self.dedup]
        dedup_contexts = self.filtered_matrix[:, mask]
        # Sort to start with most informative reads, iterate
        order = np.argsort(dedup_contexts.mask.sum(axis=1))
        trie = {}
        keep_indices = []
        for idx in order:
            read = dedup_contexts[idx].data
            # Save read to trie and note index if not duplicate
            if not is_redundant(read, trie):
                insert_read(read, trie)
                keep_indices.append(idx)
        return self.filtered_matrix[keep_indices]

    def clear(self, custom_list: list=None):
        """Clear attributes in custom_list if provided,
        otherwise delete all big matrices."""
        if custom_list is None:
            to_clear = ['_raw_matrix', '_filtered_matrix', '_deduped_matrix',
                        '_subsampled_matrix', '_clustered_matrix']
        else:
            to_clear = custom_list

        for attr in to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

    def save_matrices(self, prefix: str=""):
        """Save filtered, deduped, and clustered matrices to disk.
        The `prefix` arg can be used to specify output folder, an
        experimental condition, etc. Outputs:
        - {prefix}{reference_name}.full_unclustered.matrix
        - {prefix}{reference_name}.dedup.full_unclustered.matrix
        - {prefix}{reference_name}.clustered.matrix
        """
        header = "\t".join([f"{self.reference_name}"] + [str(i) for i in range(self.ref_len)])

        filtered_out = f"{prefix}{self.reference_name}.full_unclustered.matrix"
        mat = self.filtered_matrix.data
        n_aligns = mat.shape[0]
        fil = np.hstack([np.asarray(range(n_aligns))[:, np.newaxis], mat])
        np.savetxt(filtered_out, fil, header=header, fmt='%d', delimiter='\t', comments='#')

        deduped_out = f"{prefix}{self.reference_name}.dedup.full_unclustered.matrix"
        mat = self.deduped_matrix.data
        n_aligns = mat.shape[0]
        dedup = np.hstack([np.asarray(range(n_aligns))[:, np.newaxis], mat])
        np.savetxt(deduped_out, dedup, header=header, fmt='%d', delimiter='\t', comments='#')

        clustered_out = f"{prefix}{self.reference_name}.clustered.matrix"
        mat = self.clustered_matrix.data
        n_aligns = mat.shape[0]
        clust = np.hstack([np.asarray(range(n_aligns))[:, np.newaxis], mat])
        np.savetxt(clustered_out, clust, header=header, fmt='%d', delimiter='\t', comments='#')

    def to_pickle(self, out_path: str):
        """Pickle ConversionMatrix object."""
        with open(out_path, 'wb') as fh:
            pickle.dump(self, fh)

    ############
    # PLOTTING #
    ############
    def plot_conversion_and_coverage(self,
                                     invert: bool=False,
                                     show_plot: bool=False) -> dict[str, plt.Figure]:
        """
        Stacked line plots for:
        - Conversion/percent SMF as a function of position along the
          amplicon, the mean conversion of the unambiguous Cs in the
          requested context(s). If invert, plot 1 minus conversion.
        - Coverage as a function of position, how many unambiguous Cs
          we saw. Evaluates all Cs, no matter the context.
        """
        # Calculate C coverage, which is the same no matter c_context
        cov_x = np.where(self.masks['allC'])[0]
        cov_y = self.coverage_arrays['deduped'][cov_x]

        # Iterate over all requested C contexts
        out_figs = {}
        for c_context, vals in self.plotting_matrices.items():
            # Make two horizontal plots, 3:1 height ratio
            fig = plt.figure()
            gs = grd.GridSpec(4, 1, figure=fig)
            axs = [fig.add_subplot(gs[:3]), fig.add_subplot(gs[-1])]
            shared_xlim = (-2, self.ref_len + 2)

            # Plot mean conversion, coverage, max theoretical coverage
            y_vals = vals[1].data * 100     # percent
            if invert:
                y_vals = 100 - y_vals
            axs[0].plot(vals[0], y_vals, '-ko')
            axs[1].bar(cov_x, cov_y, width=1.0)
            axs[1].hlines(self.n_reads['deduped'],
                          xmin=shared_xlim[0],
                          xmax=shared_xlim[1],
                          colors='gray',
                          linestyles='dashed')

            # Labels and limits
            axs[0].set_title(f"{self.reference_name} amplicon:\n{c_context} context")
            axs[0].set_ylabel("% SMF")
            axs[1].set_ylabel("C coverage")
            axs[0].set_xlabel("")
            axs[0].set_xticks([])
            axs[1].set_xlabel("Amplicon position (bp)")
            axs[0].set_ylim((0, 102))
            axs[0].set_xlim(shared_xlim)
            axs[1].set_xlim(shared_xlim)

            out_figs[c_context] = fig
            if show_plot:
                plt.show()

        return out_figs

    def plot_conversion_frac_by_context(self,
                                        invert: bool=False,
                                        show_plot: bool=False) -> plt.Figure:
        """Plot fraction of Cs converted in various
        contexts. If invert, 1-fraction is plotted."""
        data = {}
        deduped = self.deduped_matrix.copy()
        data['allC'] = np.mean(deduped[:, self.masks['allC']], axis=1).data
        data['GC'] = np.mean(deduped[:, self.masks['GC_unambig']], axis=1).data
        data['CG'] = np.mean(deduped[:, self.masks['CG_unambig']], axis=1).data

        if invert:
            for context, arr in data.items():
                data[context] = 1 - arr

        fig, ax = plt.subplots()
        sns.barplot(data, ax=ax)
        ax.set_xlabel("Cytosine context")
        ax.set_title(f"{self.reference_name}")

        if invert:
            ax.set_ylabel("1 - conversion fraction")
        else:
            ax.set_ylabel("Conversion fraction")

        if show_plot:
            plt.show()

        return fig

    def plot_single_molecules(self,
                              invert: bool=False,
                              compressed: bool=False,
                              title: bool=False,
                              gray_val: float=0.8,
                              show_plot: bool=False) -> dict[str, plt.Figure]:
        """
        If invert: unconverted (0) is black, converted (1) is gray.
        Else: converted (1) is black, unconverted (0) is gray.
        Masked positions are transparent in both cases.
        If compressed, only plot the requested Cs and don't respect real distance.
        """
        n_rows, n_cols = self.clustered_matrix.shape
        if n_rows < 1 or n_cols < 1:
            print("ConversionMatrix.plot_single_molecules error:",
                  "no single molecules to plot.")
            return {}

        if invert:
            # 0 (unconv) is black, 1 (converted) is gray
            colors = [(0, 0, 0, 1), (gray_val, gray_val, gray_val, 1)]
        else:
            # 1 (converted) is black, 0 (unconv) is gray
            colors = [(gray_val, gray_val, gray_val, 1), (0, 0, 0, 1)]
        cmap = mcolors.LinearSegmentedColormap.from_list('bin', colors, N=2)
        cmap.set_bad((0, 0, 0, 0))      # masked values transparent

        out_figs = {}
        for c_context in self.c_context:
            clustered_matrix = self.clustered_matrix.copy()
            if c_context == 'CG':
                mask = self.masks['CG_unambig']
            elif c_context == 'GC':
                mask = self.masks['GC_unambig']
            elif c_context == 'allC':
                mask = self.masks['allC']

            clustered_matrix[:, ~mask] = ma.masked
            if compressed:
                clustered_matrix = clustered_matrix[:, mask]

            fig, ax = plt.subplots()
            ax.imshow(clustered_matrix,
                      cmap=cmap,
                      vmin=0,
                      vmax=1,
                      interpolation='nearest')
            ax.set_yticks([])
            xlabel = "Cytosines" if compressed else "Amplicon position (bp)"
            ax.set_xlabel(xlabel)
            plt.tight_layout()
            if title:
                plt.title(f"{self.reference_name} amplicon:\n{c_context} context")
            out_figs[c_context] = fig

            if show_plot:
                plt.show()

        return out_figs


class ConditionMatrices:
    """
    Class for analyzing data from one experimental condition. Given
    an alignment file and reference sequences, get sequences in
    reference coordinates.

    init
    ----
    bam             :   str, path to BAM file
    fasta           :   str, path to reference FASTA
    c_context       :   list[str] whose elements must be in ["CG", "GC",
                        "allC"]. See ConversionMatrix docstring.
    unconv_frac     :   float, default 0. See ConversionMatrix docstring.
    dedup_method    :   str that must be in ["CG", "GC",
                        "both_dimers", "allC", None], default
                        None, see ConversionMatrix docstring.
    min_mapq        :   int, default 0. Reject reads with mapping
                        quality lower than this number.
    subsample       :   int, default None. Subsample this many reads
                        before clustering and plotting.
    """
    def __init__(self,
                 bam: str,
                 fasta: str,
                 c_context: list[str],
                 unconv_frac: float=0.0,
                 dedup_method: str=None,
                 min_mapq: int=0,
                 subsample: int=None):
        if not isinstance(c_context, list):
            c_context = [c_context]
        if len(c_context) == 0:
            raise ValueError(f'ConditionMatrices.__init__: empty ',
                             f'list passed for c_context')
        for ele in c_context:
            if ele not in ["CG", "GC", "allC"]:
                raise ValueError(f'All elements of c_context must be one of ',
                                f'"CG", "GC", or "allC", {c_context} passed.')
        self.c_context = c_context

        if dedup_method is None:
            dedup_method = False
        elif dedup_method not in ["CG", "GC", "both_dimers", "allC"]:
            raise ValueError(f'dedup must be one of "CG", "GC", "both_dimers"',
                             f', or "allC", {dedup_method} passed.')
        self.dedup_method = dedup_method
        self.unconv_frac = unconv_frac
        self.bam = bam
        self.fasta = fasta
        self.min_mapq = min_mapq
        self.subsample = subsample

    def __repr__(self):
        """String representation of ConditionMatrices."""
        return "ConditionMatrices object:\n    {}".format("\n    ".join([
            f"{attr : <16}:    {getattr(self, attr)}"
            for attr in ['bam', 'c_context', 'dedup_method', "unconv_frac"]]))

    ##############
    # PROPERTIES #
    ##############
    @property
    def reference_dict(self) -> dict[str, np.ndarray]:
        """Dict of np.ndarray keyed by reference name, the
        reads aligning to each reference in `self.fasta`."""
        if not hasattr(self, "_reference_dict"):
            self.qc_and_merge_mates()
        return self._reference_dict

    @property
    def n_aligns(self) -> int:
        """Number of single alignments in the BAM file."""
        if not hasattr(self, "_n_aligns"):
            self.qc_and_merge_mates()
        return self._n_aligns

    @property
    def n_passing(self) -> int:
        """Number of mated reads that pass QC and mating."""
        if not hasattr(self, "_n_passing"):
            self.qc_and_merge_mates()
        return self._n_passing

    @property
    def n_qcfail(self) -> int:
        """Number of single reads in `self.bam` that failed QC."""
        if not hasattr(self, "_n_qcfail"):
            self.qc_and_merge_mates()
        return self._n_qcfail

    @property
    def n_mapqfail(self) -> int:
        """Number of single reads in `self.bam` that failed MAPQ threshold."""
        if not hasattr(self, "_n_mapqfail"):
            self.qc_and_merge_mates()
        return self._n_mapqfail

    @property
    def n_unmated(self) -> int:
        """Number of single reads in `self.bam` that
        remained unmated after the BAM file was streamed."""
        if not hasattr(self, "_n_unmated"):
            self.qc_and_merge_mates()
        return self._n_unmated

    @property
    def mapqs(self) -> np.ndarray:
        """MAPQ scores for every alignment that passed initial QC flags."""
        if not hasattr(self, "_mapqs"):
            self.qc_and_merge_mates()
        return self._mapqs

    @property
    def conv_matrices(self) -> dict[str, ConversionMatrix]:
        """Child ConversionMatrix objects, one per reference sequence."""
        if not hasattr(self, "_conv_matrices"):
            out = {}
            fa = pysam.FastaFile(self.fasta)
            for ref, reads in self.reference_dict.items():
                out[ref] = ConversionMatrix(reads,
                                            fa.fetch(ref),
                                            ref,
                                            c_context=self.c_context,
                                            unconv_frac=self.unconv_frac,
                                            dedup_method=self.dedup_method,
                                            subsample=self.subsample)
            self._conv_matrices = out
        return self._conv_matrices

    ###########
    # METHODS #
    ###########
    def qc_and_merge_mates(self):
        """
        Stream a possibly unsorted BAM file. Look at SAM flags to QC, and
        for mates that pass, merge sequences in reference coordinates."""
        def merge_mates(mate1: pysam.AlignedSegment,
                        mate2: pysam.AlignedSegment,
                        ref_len: int) -> np.ndarray:
            """
            Given two mated AlignedSegments, merge in ref coordinates:
            1. All unsequenced bases in the reference are 'N'
            2. If only one read covers a position and its base
            quality is 0, that position is assigned 'N'
            3. If only one read covers a position and its base quality
            is > 0, the position is assigned that read's base
            4. If the reads overlap and have different quality scores,
            the position is assigned the base with the better score
            5. If the reads overlap and have the same quality score and
            same base at a position, that position is assigned that base
            6. If the reads overlap and have the same quality score but
            different bases at a position, that position is assigned 'N'
            """
            seqs = np.full((2, ref_len), "N")
            quals = np.zeros_like(seqs, dtype=int)
            for i, mate in enumerate((mate1, mate2)):
                q_quals = mate.query_qualities
                if q_quals is None:
                    continue
                q_seq = mate.query_sequence
                for qpos, rpos in mate.get_aligned_pairs(matches_only=True):
                    seqs[i, rpos] = q_seq[qpos]
                    quals[i, rpos] = q_quals[qpos]

            merge = np.full(ref_len, "N")
            merge[quals[0] > quals[1]] = seqs[0, (quals[0] > quals[1])]
            merge[quals[1] > quals[0]] = seqs[1, (quals[1] > quals[0])]

            mask = ((quals[0] == quals[1]) &
                    (seqs[0] == seqs[1]) &
                    (quals[0] > 0))
            merge[mask] = seqs[0, mask]
            return merge

        fa = pysam.FastaFile(self.fasta)
        bam = pysam.AlignmentFile(self.bam, "rb")
        n_qcfail = 0
        n_mapqfail = 0
        n_aligns = 0
        all_mapqs = []
        cache = {}
        arrs = {}
        for aln in bam:
            n_aligns += 1
            if (aln.is_unmapped or
                not aln.is_proper_pair or
                aln.is_qcfail or
                aln.is_secondary or
                aln.is_supplementary):
                n_qcfail += 1
                continue

            all_mapqs.append(aln.mapping_quality)
            if aln.mapping_quality < self.min_mapq:
                n_mapqfail += 1
                continue

            if aln.query_name not in cache:
                if len(cache) >= MAX_CACHE:
                    raise RuntimeError("Max cache size reached searching for "
                                       "mates, try again with a name-sorted BAM.")
                cache[aln.query_name] = aln
                continue

            mate = cache.pop(aln.query_name)
            ref_name = aln.reference_name
            if ref_name != mate.reference_name:
                n_qcfail += 2
                continue

            merge_seq = merge_mates(mate1=aln,
                                    mate2=mate,
                                    ref_len=bam.lengths[aln.reference_id])
            arrs.setdefault(ref_name, []).append(merge_seq)

        n_passing = 0
        ref_dict = {}
        for ref in fa.references:
            try:
                ref_dict[ref] = np.vstack(arrs[ref])
                n_passing += ref_dict[ref].shape[0]
            except Exception:
                print(f"No passing, mated alignments to {ref}.")
        self._reference_dict = ref_dict
        self._n_aligns = n_aligns
        self._n_passing = n_passing
        self._n_qcfail = n_qcfail
        self._n_mapqfail = n_mapqfail
        self._n_unmated = len(cache)
        self._mapqs = np.asarray(all_mapqs)

    def save_all_amplicon_stats(self,
                                out_path: str=None,
                                header: bool=True,
                                stdout: bool=False):
        """Write per-amplicon read counts to file and/or stdout."""
        if (not stdout) and (out_path is None):
            return

        str_list = []
        if header:
            h = "amplicon\ttotal_reads\tobserved_states\treads_per_state"
            str_list.append(h)
        str_list.extend([cm.amplicon_stats for cm in self.conv_matrices.values()])
        joined = "\n".join(str_list)

        if out_path is not None:
            with open(out_path, 'w') as fh:
                fh.write(joined + "\n")
        if stdout:
            print(joined)

    def save_all_matrices(self, prefix: str=""):
        """Save filtered, deduped, and clustered matrices for every amplicon."""
        for cm in self.conv_matrices.values():
            cm.save_matrices(prefix=prefix)

    def to_pickle(self, out_path: str):
        """Pickle ConditionMatrices object."""
        with open(out_path, 'wb') as fh:
            pickle.dump(self, fh)

    ############
    # PLOTTING #
    ############
    def plot_all_conversion_and_coverage(self, out_pdf: str, invert: bool=False):
        with PdfPages(out_pdf) as pdf:
            for cm in self.conv_matrices.values():
                if cm.n_reads['deduped'] < 1:
                    continue
                for fig in cm.plot_conversion_and_coverage(invert=invert).values():
                    pdf.savefig(fig)
                    plt.close(fig)

    def plot_all_conversion_frac_by_context(self,
                                            out_pdf: str,
                                            invert: bool=False):
        with PdfPages(out_pdf) as pdf:
            for conv_matrix in self.conv_matrices.values():
                if conv_matrix.n_reads['deduped'] < 1:
                    continue
                fig = conv_matrix.plot_conversion_frac_by_context(invert=invert)
                pdf.savefig(fig)
                plt.close(fig)

    def plot_aggregate_conversion_fracs(self,
                                        out_path: str=None,
                                        bins: int=15,
                                        show_plot: bool=False):
        if (not out_path) and (not show_plot):
            return

        frac_arrs = []
        for conv_matrix in self.conv_matrices.values():
            frac_arrs.append(conv_matrix.per_read_conv)
        all_conv = np.concatenate(frac_arrs)

        plt.hist(all_conv, bins=bins)
        plt.axvline(self.unconv_frac, color='r', linestyle='--')

        if out_path is not None:
            plt.savefig(out_path, dpi=600)
        if show_plot:
            plt.show()
        plt.close()

    def plot_mapqs(self, out_path: str=None, bins: int=20, show_plot: bool=False):
        """Plot MAPQs for every alignment that passed initial QC checks."""
        if (not out_path) and (not show_plot):
            return

        plt.hist(self.mapqs, bins=bins)
        plt.axvline(self.min_mapq, color='r', linestyle='--')
        plt.xlabel("MAPQ")

        if out_path is not None:
            plt.savefig(out_path, dpi=600)
        if show_plot:
            plt.show()
        plt.close()

    def plot_all_amplicon_loss(self, out_path: str=None, show_plot: bool=False):
        if (not out_path) and (not show_plot):
            return

        _, ax = plt.subplots(figsize=(8, 5))
        for i, conv_matrix in enumerate(self.conv_matrices.values()):
            refname = conv_matrix.reference_name
            ax.bar(refname, conv_matrix.n_reads['raw'], zorder=0, color='orange',
                   label='unconverted reads' if i == 0 else None)
            ax.bar(refname, conv_matrix.n_reads['filtered'], zorder=1, color='gray',
                   label='converted reads' if i == 0 else None)
            ax.bar(refname, conv_matrix.n_reads['deduped'], zorder=2, color='blue',
                   label='deduped reads' if i == 0 else None)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()

        if out_path is not None:
            plt.savefig(out_path, dpi=600)
        if show_plot:
            plt.show()
        plt.close()

    def plot_aggregate_loss(self, out_path: str=None, show_plot: bool=False):
        if (not out_path) and (not show_plot):
            return

        aggregate_n_reads = {'raw': 0, 'filtered': 0, 'deduped': 0}
        for conv_matrix in self.conv_matrices.values():
            for filter_name, n_reads in conv_matrix.n_reads.items():
                aggregate_n_reads[filter_name] += n_reads

        plt.bar('Total alignments', self.n_aligns / 2)
        plt.bar('Retained reads', aggregate_n_reads['raw'])
        plt.bar('Converted reads', aggregate_n_reads['filtered'])
        plt.bar('Unique reads', aggregate_n_reads['deduped'])

        if out_path is not None:
            plt.savefig(out_path, dpi=600)
        if show_plot:
            plt.show()
        plt.close()

    def plot_all_single_molecule_matrices(self,
                                          out_pdf: str,
                                          invert: bool=False,
                                          compressed: bool=False,
                                          gray_val: float=0.75):
        with PdfPages(out_pdf) as pdf:
            for cm in self.conv_matrices.values():
                figs = cm.plot_single_molecules(invert=invert,
                                                compressed=compressed,
                                                gray_val=gray_val)
                for fig in figs.values():
                    pdf.savefig(fig)
                    plt.close(fig)


def main():
    args = parse_args()
    dedup = None if args.dedup_method == 'none' else args.dedup_method
    cond = ConditionMatrices(
        bam=args.bam_path,
        fasta=args.fa_path,
        c_context=args.c_context,
        unconv_frac=args.unconv_frac,
        dedup_method=dedup,
        min_mapq=args.min_mapq,
        subsample=args.subsample,
    )
    matrix_dir = os.path.dirname(args.matrix_prefix)
    if matrix_dir:
        os.makedirs(matrix_dir, exist_ok=True)
    cond.save_all_matrices(prefix=args.matrix_prefix)
    cond.save_all_amplicon_stats(out_path=args.stats_out, stdout=True)
    if args.conversion_pdf:
        cond.plot_all_conversion_and_coverage(args.conversion_pdf, invert=args.invert)
    if args.sm_pdf:
        cond.plot_all_single_molecule_matrices(
            args.sm_pdf,
            invert=args.invert,
            compressed=args.compressed,
        )


if __name__ == "__main__":
    main()
