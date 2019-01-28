import unittest
import os
import sys
sys.path.append('../')

from phrases.model.paraphraser import Paraphrases

class TestParaphrasesModel( unittest.TestCase ):
    
    def test_model( self ):
        net = Paraphrases()
        net.run(pathconfigurate='../modelsconfig.json' )
                
        sentence = 'open camera ... ' 
        phrase, score = net(sentence)
        
        print('>> ', sentence)
        print('<< ', phrase)
        print(score)
        
        
    
if __name__ == '__main__':
    unittest.main()
        