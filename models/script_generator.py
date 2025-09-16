class ScriptGenerator:
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
    
    def generate(self, content, style: str = 'engaging', target_duration: int = 60):
        text = content.get('text') or content.get('title') or 'Interesting topic'
        main = text[:200]
        return {
            'sections': [
                {'id': 'hook', 'type': 'hook', 'text': f"You won't believe this: {text[:60]}...", 'duration': 5},
                {'id': 'main', 'type': 'story', 'text': main, 'duration': max(5, target_duration - 10)},
                {'id': 'cta', 'type': 'call_to_action', 'text': 'Follow for more!', 'duration': 5}
            ],
            'duration': target_duration,
            'word_count': max(120, len(main.split())),
            'hooks': ['You won\'t believe'],
            'cta': 'Follow for more!'
        }

