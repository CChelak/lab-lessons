import unittest
import numpy as np

def cumulative_sum(arr):
    """Perform a cumulative sum on the elements of an array"""
    cumulative_arr = []
    sum = 0
    for val in arr:
        sum += val
        cumulative_arr.append(sum)

    return cumulative_arr

# We create a test class as follows
class TestCumulativeSum(unittest.TestCase):
    def test_ones(self):
        """Pass some ones into the function and see if it increments"""
        ones = [1,1,1,1,1]
        res = np.cumsum(ones)
        self.assertTrue(np.array_equal(res, [1,2,3,4,5]))

    def test_custom(self):
        """Test our custom cumulative sum"""
        arr = [2, 3, 6]
        res = cumulative_sum(arr)
        self.assertEqual(res, [2, 5, 11])

if __name__ == "__main__":
    unittest.main()