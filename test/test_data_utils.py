import unittest
import os
import sys
sys.path.append('../')

from torchlib.datasets import utils

class TestUtils( unittest.TestCase ):
    def test_read( self ):
 
        pathname = '~/.datasets/txt'
        pathfile = 'para-nmt-50m-demo/para-nmt-50m-small.txt'
        pathname = os.path.expanduser(pathname)
        pathfile = os.path.expanduser(pathfile)

        pairs = utils.read_paraphraser( os.path.join( pathname, pathfile ) )
        print( len(pairs) )
        print( pairs[1] )

    def test_normalize_string( self ):    
        s = 'This Is a tEst ... '
        s_norm = utils.normalizeString(s)
        print( '>> {}'.format(s) )
        print( '<< {}'.format(s_norm) )



if __name__ == '__main__':
    unittest.main()