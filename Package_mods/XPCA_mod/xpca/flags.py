
from enum import IntFlag

"""
For 4XP-Z the warnings are summarized as: 0 = BAD_PIXEL_COUNT,
1 = LOW_SNR, 2 = NOT_ENOUGH_PEAKS, 3 = LINALG_ERROR,
4 = NO_CHI2_MIN, 5 = NO_ERROR_ESTIMATE, 6 = NO_UNIQUE_Z,
7 = HIGH_CHI2, 8 = LOW_PEAK_SIG
"""

class Flags(IntFlag):
    """16-bit integer flag"""
    NO_UNIQUE_Z       = 2**0    # More than one other model is consistent with the best-fit
    MINIMUM_AT_EDGE   = 2**1    # The minimum chi^2 is at the edge of the redshift grid
    LOW_REDCHI2       = 2**2    # placeholder
    LINALG_ERROR      = 2**3    # No solution to the PCA matrix fit, Chi2 = -999
    BAD_CHI2FIT       = 2**4    # No solution to the parabolic fit of the redshift grid
    NO_CHI2_CROSSINGS = 2**5    # Couldn't estimate redshift uncertainty from redshift grid
                                # usually because the grid isn't a minimum or has two extrema
    NEGATIVE_CHI2     = 2**6    # Negative chi^2 value from fitted parabola
    Z_OUTSIDE_RANGE   = 2**7    # Best-fit redshift is outside the range of the grid
    PEAKS_WARNING     = 2**8    # Fewer than `N_PEAKS` peaks in the cross-correlation function
    NO_PEAKS_ERROR    = 2**9    # No peaks found in the cross-correlation function
    NO_PEAK_SOLUTION  = 2**10   # No solution of parabolic fit to cross-correlation peak
    BAD_SPECTRUM      = 2**11   # Corrupted input spectrum

    def get_flags(self):
        return [flag.name for flag in Flags if flag in self]

    def to_string(self, delimiter='; '):
        return delimiter.join(self.get_flags())
