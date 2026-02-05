import unittest
from cv01.test.test import TestMnist
import generate_results

if __name__ == '__main__':
    # Run tests without exiting
    unittest.main(exit=False)
    # Then run generate_results
    generate_results.main()

