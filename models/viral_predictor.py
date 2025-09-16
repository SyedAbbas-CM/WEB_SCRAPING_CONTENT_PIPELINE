class ViralPredictor:
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
    
    def predict(self, content, platform: str = 'unknown'):
        text = content.get('text') or content.get('title') or ''
        score = min(100, max(0, len(text) % 100))
        return {'score': score, 'confidence': 0.7}

