import unittest
import pickle
import os

class TestFakeNewsModel(unittest.TestCase):
    def setUp(self):
        # Path to the saved model
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
        
    def test_model_output_type(self):
        """Check if the model correctly returns a 'real' or 'fake' string."""
        if not os.path.exists(self.model_path):
            self.skipTest("model.pkl not found. Run train_model.py first.")
            
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
            
        test_text = "The government announced a new policy today."
        prediction = model.predict([test_text])[0]
        
        # Verify the prediction is one of the allowed labels
        self.assertIn(prediction, ['real', 'fake'])

if __name__ == '__main__':
    unittest.main()
