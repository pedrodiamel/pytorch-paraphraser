import unittest
import os
import sys
sys.path.append('../')
from torchlib.datasets.downloads import download_data, extract

class TestDownload( unittest.TestCase ):
    # def test_data(self):
    #     idfile = '1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD'
    #     namefile = 'para-nmt-50m-demo.zip'
    #     pathname = os.path.expanduser( '~/.datasets/txt' )
    #     #download_data( namefile, idfile, pathname, ext=True )

    def test_extrac(self):
        namefile = 'para-nmt-50m-demo.tar.gz'
        pathname = os.path.expanduser( '~/.datasets/txt' )  
        pathname = os.path.expanduser(pathname) 

        print( namefile )
        print( pathname )
        
        extract( pathname, namefile )
        self.assertTrue( os.path.exists( os.path.join( 'para-nmt-50m-demo', pathname ) ) )


if __name__ == '__main__':
    unittest.main()
    


