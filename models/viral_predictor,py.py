# models/viral_predictor.py
"""
Viral Content Prediction Model
Uses multiple signals to predict viral potential
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime

class ViralPredictor:
    """
    Predicts viral potential using:
    - Content features (text, sentiment, topics)
    - Temporal features (posting time, trends)
    - Platform-specific signals
    - Historical performance data
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger('viral_predictor')
        
        # Load pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
        
        # Load or initialize viral prediction head
        self.prediction_head = ViralPredictionHead().to(device)
        
        if model_path:
            self.load_model(model_path)
        
        # Platform-specific weights
        self.platform_weights = {
            'reddit': {'text': 0.4, 'timing': 0.2, 'sentiment': 0.2, 'trends': 0.2},
            'twitter': {'text': 0.3, 'timing': 0.3, 'sentiment': 0.2, 'trends': 0.2},
            'tiktok': {'text': 0.2, 'timing': 0.3, 'sentiment': 0.2, 'trends': 0.3},
            'instagram': {'text': 0.2, 'timing': 0.3, 'sentiment': 0.3, 'trends': 0.2}
        }
        
        # Viral indicators database
        self.viral_indicators = self._load_viral_indicators()
        
    def predict(self, content: Dict, platform: str) -> Dict:
        """Predict viral potential of content"""
        # Extract features
        features = self._extract_features(content, platform)
        
        # Get text embeddings
        text_embedding = self._get_text_embedding(content.get('text', ''))
        
        # Combine all features
        combined_features = torch.cat([
            text_embedding,
            torch.tensor(features['temporal'], device=self.device),
            torch.tensor(features['engagement'], device=self.device),
            torch.tensor(features['sentiment'], device=self.device)
        ])
        
        # Make prediction
        with torch.no_grad():
            viral_score = self.prediction_head(combined_features)
            confidence = self._calculate_confidence(features)
        
        # Calculate sub-scores
        sub_scores = self._calculate_sub_scores(features, platform)
        
        return {
            'score': float(viral_score.item()),
            'confidence': float(confidence),
            'sub_scores': sub_scores,
            'key_factors': self._identify_key_factors(features, viral_score),
            'recommendations': self._generate_recommendations(features, viral_score, platform)
        }
    
    def _extract_features(self, content: Dict, platform: str) -> Dict:
        """Extract all features from content"""
        features = {
            'temporal': self._extract_temporal_features(content),
            'engagement': self._extract_engagement_features(content, platform),
            'sentiment': self._extract_sentiment_features(content),
            'content': self._extract_content_features(content),
            'trends': self._extract_trend_features(content, platform)
        }
        
        return features
    
    def _extract_temporal_features(self, content: Dict) -> np.ndarray:
        """Extract time-based features"""
        features = []
        
        # Hour of day (0-23)
        if 'created_utc' in content:
            dt = datetime.fromtimestamp(content['created_utc'])
            features.append(dt.hour / 23.0)
            features.append(dt.weekday() / 6.0)  # Day of week
            features.append(int(dt.weekday() >= 5))  # Is weekend
        else:
            features.extend([0.5, 0.5, 0])
        
        # Time since creation (freshness)
        age_hours = (time.time() - content.get('created_utc', time.time())) / 3600
        features.append(min(age_hours / 168, 1.0))  # Normalize to week
        
        return np.array(features, dtype=np.float32)
    
    def _extract_engagement_features(self, content: Dict, platform: str) -> np.ndarray:
        """Extract engagement-based features"""
        features = []
        
        if platform == 'reddit':
            score = content.get('score', 0)
            comments = content.get('num_comments', 0)
            
            # Normalize scores
            features.append(min(score / 10000, 1.0))
            features.append(min(comments / 1000, 1.0))
            
            # Engagement ratio
            if score > 0:
                features.append(min(comments / score, 1.0))
            else:
                features.append(0)
            
            # Awards (strong viral signal)
            features.append(min(content.get('awards', 0) / 10, 1.0))
            
        elif platform == 'twitter':
            likes = content.get('like_count', 0)
            retweets = content.get('retweet_count', 0)
            replies = content.get('reply_count', 0)
            
            features.append(min(likes / 10000, 1.0))
            features.append(min(retweets / 5000, 1.0))
            features.append(min(replies / 1000, 1.0))
            
            # Retweet ratio (virality indicator)
            if likes > 0:
                features.append(min(retweets / likes, 1.0))
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_sentiment_features(self, content: Dict) -> np.ndarray:
        """Extract sentiment and emotion features"""
        from textblob import TextBlob
        
        text = content.get('text', '') or content.get('title', '')
        
        if not text:
            return np.zeros(5, dtype=np.float32)
        
        # Basic sentiment
        blob = TextBlob(text)
        polarity = (blob.sentiment.polarity + 1) / 2  # Normalize to 0-1
        subjectivity = blob.sentiment.subjectivity
        
        # Emotional triggers
        emotions = {
            'surprise': len([w for w in ['wow', 'amazing', 'incredible', 'unbelievable'] if w in text.lower()]),
            'controversy': len([w for w in ['controversial', 'debate', 'opinion', 'unpopular'] if w in text.lower()]),
            'urgency': len([w for w in ['breaking', 'urgent', 'now', 'today'] if w in text.lower()])
        }
        
        features = [
            polarity,
            subjectivity,
            min(emotions['surprise'] / 3, 1.0),
            min(emotions['controversy'] / 3, 1.0),
            min(emotions['urgency'] / 3, 1.0)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_content_features(self, content: Dict) -> Dict:
        """Extract content structure features"""
        text = content.get('text', '') or content.get('title', '')
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'question_marks': text.count('?'),
            'exclamations': text.count('!'),
            'capitals_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'has_media': int(bool(content.get('url') and not content['url'].startswith('https://www.reddit.com'))),
            'has_video': int(content.get('is_video', False))
        }
        
        return features
    
    def _extract_trend_features(self, content: Dict, platform: str) -> np.ndarray:
        """Extract trending topic features"""
        text = content.get('text', '') or content.get('title', '')
        
        # Check for trending keywords
        trending_score = 0
        for keyword, weight in self.viral_indicators['trending_keywords'].items():
            if keyword.lower() in text.lower():
                trending_score += weight
        
        # Check for viral hashtags
        hashtag_score = 0
        if platform in ['twitter', 'instagram', 'tiktok']:
            hashtags = self._extract_hashtags(text)
            for tag in hashtags:
                if tag in self.viral_indicators['viral_hashtags']:
                    hashtag_score += 1
        
        features = [
            min(trending_score, 1.0),
            min(hashtag_score / 5, 1.0)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for text"""
        if not text:
            return torch.zeros(768, device=self.device)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding
    
    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate prediction confidence"""
        confidence_factors = []
        
        # Data completeness
        data_completeness = sum(
            1 for v in features.values() 
            if v is not None and (not isinstance(v, np.ndarray) or v.size > 0)
        ) / len(features)
        confidence_factors.append(data_completeness)
        
        # Feature strength
        engagement_strength = np.mean(features['engagement'])
        confidence_factors.append(engagement_strength)
        
        # Trend alignment
        trend_strength = np.mean(features['trends'])
        confidence_factors.append(trend_strength)
        
        return float(np.mean(confidence_factors))
    
    def _calculate_sub_scores(self, features: Dict, platform: str) -> Dict:
        """Calculate detailed sub-scores"""
        weights = self.platform_weights[platform]
        
        return {
            'content_quality': float(np.mean(features['sentiment']) * weights['text']),
            'timing_score': float(np.mean(features['temporal']) * weights['timing']),
            'sentiment_score': float(features['sentiment'][0] * weights['sentiment']),
            'trend_alignment': float(np.mean(features['trends']) * weights['trends'])
        }
    
    def _identify_key_factors(self, features: Dict, viral_score: torch.Tensor) -> List[str]:
        """Identify key factors contributing to viral score"""
        factors = []
        
        # High engagement
        if np.mean(features['engagement']) > 0.7:
            factors.append("High existing engagement")
        
        # Good timing
        temporal = features['temporal']
        if temporal[0] > 0.6 and temporal[0] < 0.9:  # Peak hours
            factors.append("Posted during peak hours")
        
        # Emotional content
        if features['sentiment'][2] > 0.5 or features['sentiment'][3] > 0.5:
            factors.append("Strong emotional triggers")
        
        # Trending alignment
        if features['trends'][0] > 0.5:
            factors.append("Contains trending topics")
        
        return factors
    
    def _generate_recommendations(self, features: Dict, viral_score: torch.Tensor, platform: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        score = viral_score.item()
        
        if score < 50:
            # Low viral potential
            if features['sentiment'][0] < 0.5:
                recommendations.append("Add more positive emotional content")
            
            if features['content']['question_marks'] == 0:
                recommendations.append("Consider asking engaging questions")
            
            if features['trends'][0] < 0.3:
                recommendations.append("Incorporate trending topics or hashtags")
        
        elif score < 75:
            # Medium potential
            if features['temporal'][0] < 0.5 or features['temporal'][0] > 0.9:
                recommendations.append("Post during peak engagement hours (6-9 PM)")
            
            if not features['content']['has_media']:
                recommendations.append("Add images or video for higher engagement")
        
        else:
            # High potential
            recommendations.append("Content has high viral potential - post immediately!")
            
            if platform == 'tiktok' and features['trends'][1] < 0.5:
                recommendations.append("Use trending sounds/hashtags for maximum reach")
        
        return recommendations
    
    def _load_viral_indicators(self) -> Dict:
        """Load database of viral indicators"""
        return {
            'trending_keywords': {
                'breaking': 0.3,
                'just in': 0.3,
                'update': 0.2,
                'exclusive': 0.25,
                'leaked': 0.3,
                'shocking': 0.25,
                'unbelievable': 0.2,
                'you won\'t believe': 0.25
            },
            'viral_hashtags': {
                '#fyp', '#foryou', '#viral', '#trending',
                '#breakingnews', '#mustwatch', '#omg'
            }
        }
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        import re
        return re.findall(r'#\w+', text)
    
    def load_model(self, path: str):
        """Load saved model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.prediction_head.load_state_dict(checkpoint['prediction_head'])
        self.logger.info(f"Loaded model from {path}")
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'prediction_head': self.prediction_head.state_dict()
        }, path)


class ViralPredictionHead(nn.Module):
    """Neural network head for viral prediction"""
    
    def __init__(self, input_dim: int = 768 + 20):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x) * 100  # Scale to 0-100


# models/content_analyzer.py
"""
Content Analysis Model
Deep analysis of scraped content
"""

import torch
import numpy as np
from typing import Dict, List, Any
from transformers import pipeline
import spacy
from collections import Counter
import logging

class ContentAnalyzer:
    """
    Comprehensive content analysis:
    - Topic extraction
    - Sentiment analysis
    - Entity recognition
    - Readability scoring
    - Audience identification
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger('content_analyzer')
        
        # Load models
        self.sentiment_analyzer = pipeline('sentiment-analysis', device=0 if device == 'cuda' else -1)
        self.ner_model = spacy.load('en_core_web_sm')
        self.topic_model = self._load_topic_model()
        
        # Analysis components
        self.readability_calculator = ReadabilityCalculator()
        self.audience_identifier = AudienceIdentifier()
        
    def analyze(self, content: Dict) -> Dict:
        """Perform comprehensive content analysis"""
        text = self._extract_text(content)
        
        if not text:
            return {'error': 'No text content found'}
        
        # Run all analyses
        analysis = {
            'content_id': content.get('id', 'unknown'),
            'sentiment': self._analyze_sentiment(text),
            'emotions': self._analyze_emotions(text),
            'topics': self._extract_topics(text),
            'entities': self._extract_entities(text),
            'keywords': self._extract_keywords(text),
            'readability': self.readability_calculator.calculate(text),
            'audience': self.audience_identifier.identify(text),
            'content_type': self._classify_content_type(content),
            'quality_score': self._calculate_quality_score(text)
        }
        
        return analysis
    
    def _extract_text(self, content: Dict) -> str:
        """Extract all text from content"""
        text_parts = []
        
        # Title
        if 'title' in content:
            text_parts.append(content['title'])
        
        # Body text
        if 'text' in content:
            text_parts.append(content['text'])
        elif 'selftext' in content:
            text_parts.append(content['selftext'])
        elif 'body' in content:
            text_parts.append(content['body'])
        
        # Comments (if analyzing post with comments)
        if 'comments' in content and isinstance(content['comments'], list):
            for comment in content['comments'][:10]:  # Top 10 comments
                if isinstance(comment, dict) and 'body' in comment:
                    text_parts.append(comment['body'])
        
        return ' '.join(text_parts)
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Detailed sentiment analysis"""
        # Overall sentiment
        result = self.sentiment_analyzer(text[:512])  # Truncate for model
        
        # Sentence-level sentiment
        sentences = text.split('.')[:10]  # Analyze first 10 sentences
        sentence_sentiments = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                sent_result = self.sentiment_analyzer(sentence)
                sentence_sentiments.append(sent_result[0])
        
        # Calculate sentiment distribution
        positive_ratio = sum(1 for s in sentence_sentiments if s['label'] == 'POSITIVE') / max(len(sentence_sentiments), 1)
        
        return {
            'overall': result[0]['label'].lower(),
            'score': result[0]['score'],
            'positive_ratio': positive_ratio,
            'sentence_sentiments': sentence_sentiments
        }
    
    def _analyze_emotions(self, text: str) -> Dict:
        """Analyze emotional content"""
        # Emotion keywords
        emotions = {
            'joy': ['happy', 'joy', 'excited', 'amazing', 'wonderful', 'fantastic'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'frustrated', 'outraged'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous'],
            'sadness': ['sad', 'depressed', 'crying', 'miserable', 'heartbroken'],
            'surprise': ['shocked', 'surprised', 'astonished', 'amazed', 'unexpected']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = min(score / 10, 1.0)  # Normalize
        
        # Dominant emotion
        dominant = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'scores': emotion_scores,
            'dominant': dominant[0] if dominant[1] > 0 else 'neutral'
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        # Simple topic extraction using noun phrases
        doc = self.ner_model(text)
        
        # Extract noun phrases
        topics = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Max 3 words
                topics.append(chunk.text.lower())
        
        # Count frequency
        topic_counts = Counter(topics)
        
        # Return top topics
        return [topic for topic, _ in topic_counts.most_common(10)]
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities"""
        doc = self.ner_model(text)
        
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'other': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['people'].append(ent.text)
            elif ent.label_ in ['ORG', 'COMPANY']:
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['LOC', 'GPE']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            else:
                entities['other'].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF approach"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Simple keyword extraction
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = [(feature_names[i], scores[i]) for i in range(len(scores))]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, _ in keyword_scores[:10]]
        except:
            # Fallback to simple word frequency
            words = text.lower().split()
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(10) if len(word) > 3]
    
    def _classify_content_type(self, content: Dict) -> str:
        """Classify type of content"""
        text = self._extract_text(content)
        
        # Check for questions
        if '?' in text[:100]:  # Question in beginning
            return 'question'
        
        # Check for stories
        story_indicators = ['tifu', 'story time', 'this happened', 'years ago']
        if any(indicator in text.lower() for indicator in story_indicators):
            return 'story'
        
        # Check for news
        news_indicators = ['breaking', 'announced', 'report', 'according to']
        if any(indicator in text.lower() for indicator in news_indicators):
            return 'news'
        
        # Check for opinion
        opinion_indicators = ['i think', 'in my opinion', 'unpopular opinion', 'cmv']
        if any(indicator in text.lower() for indicator in opinion_indicators):
            return 'opinion'
        
        return 'general'
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate overall content quality score"""
        scores = []
        
        # Length score
        word_count = len(text.split())
        if 100 <= word_count <= 500:
            scores.append(1.0)
        elif 50 <= word_count <= 1000:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # Grammar score (simplified)
        doc = self.ner_model(text)
        grammar_errors = sum(1 for token in doc if token.dep_ == 'ROOT' and token.pos_ != 'VERB')
        grammar_score = max(0, 1 - (grammar_errors / 10))
        scores.append(grammar_score)
        
        # Readability score
        readability = self.readability_calculator.calculate(text)
        if 60 <= readability['flesch_ease'] <= 80:
            scores.append(1.0)
        elif 40 <= readability['flesch_ease'] <= 90:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        return float(np.mean(scores) * 100)
    
    def _load_topic_model(self):
        """Load topic modeling component"""
        # Placeholder for actual topic model
        return None


class ReadabilityCalculator:
    """Calculate readability metrics"""
    
    def calculate(self, text: str) -> Dict:
        """Calculate various readability scores"""
        sentences = text.split('.')
        words = text.split()
        
        # Basic metrics
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Syllable counting (simplified)
        total_syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / max(len(words), 1)
        
        # Flesch Reading Ease
        flesch_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        flesch_ease = max(0, min(100, flesch_ease))  # Bound between 0-100
        
        # Grade level
        grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        grade_level = max(0, min(18, grade_level))  # Bound between 0-18
        
        return {
            'flesch_ease': flesch_ease,
            'grade_level': grade_level,
            'avg_sentence_length': avg_sentence_length,
            'complexity': 'simple' if flesch_ease > 60 else 'complex'
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiou'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Minimum of 1 syllable
        return max(1, syllable_count)


class AudienceIdentifier:
    """Identify target audience for content"""
    
    def identify(self, text: str) -> Dict:
        """Identify likely audience characteristics"""
        text_lower = text.lower()
        
        # Age indicators
        age_groups = {
            'teen': ['high school', 'homework', 'parents', 'teenager', 'prom'],
            'young_adult': ['college', 'university', 'student loan', 'internship', 'entry level'],
            'adult': ['mortgage', 'career', 'marriage', 'kids', 'retirement'],
            'senior': ['grandchildren', 'retired', 'medicare', 'pension']
        }
        
        detected_age = 'general'
        for age, keywords in age_groups.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_age = age
                break
        
        # Interest indicators
        interests = {
            'tech': ['programming', 'software', 'ai', 'crypto', 'startup'],
            'gaming': ['game', 'ps5', 'xbox', 'nintendo', 'steam'],
            'finance': ['invest', 'stock', 'crypto', 'money', 'wealth'],
            'fitness': ['workout', 'gym', 'diet', 'fitness', 'health'],
            'entertainment': ['movie', 'show', 'netflix', 'music', 'concert']
        }
        
        detected_interests = []
        for interest, keywords in interests.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_interests.append(interest)
        
        # Gender indicators (be careful with assumptions)
        gender_neutral_score = 1.0
        
        return {
            'age_group': detected_age,
            'interests': detected_interests[:3],  # Top 3 interests
            'gender_neutral_score': gender_neutral_score,
            'complexity_level': 'advanced' if len(text.split()) > 200 else 'casual'
        }


# models/script_generator.py
"""
Script Generation Model
Generates video scripts from analyzed content
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Optional
import logging
import re

class ScriptGenerator:
    """
    Generates engaging video scripts:
    - Multiple style templates
    - Platform-specific formatting
    - Hook generation
    - Call-to-action optimization
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger('script_generator')
        
        # Load language model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
        
        # Script templates
        self.templates = self._load_templates()
        
        # Platform-specific settings
        self.platform_settings = {
            'tiktok': {'max_duration': 60, 'style': 'fast_paced'},
            'instagram': {'max_duration': 90, 'style': 'visual'},
            'youtube': {'max_duration': 180, 'style': 'detailed'}
        }
    
    def generate(self, content: Dict, style: str = 'engaging', 
                 target_duration: int = 60) -> Dict:
        """Generate video script from content"""
        
        # Select template based on content type
        template = self._select_template(content, style)
        
        # Generate script sections
        script = {
            'id': f"script_{int(time.time())}",
            'sections': [],
            'total_duration': 0,
            'word_count': 0,
            'style': style
        }
        
        # Generate hook
        hook = self._generate_hook(content, template)
        script['sections'].append(hook)
        
        # Generate main content
        main_content = self._generate_main_content(content, template, target_duration)
        script['sections'].extend(main_content)
        
        # Generate call-to-action
        cta = self._generate_cta(content, template)
        script['sections'].append(cta)
        
        # Calculate totals
        script['total_duration'] = sum(s['duration'] for s in script['sections'])
        script['word_count'] = sum(s['word_count'] for s in script['sections'])
        
        # Extract key elements
        script['hooks'] = [hook['text']]
        script['cta'] = cta['text']
        
        # Add production notes
        script['production_notes'] = self._generate_production_notes(script)
        
        return script
    
    def _select_template(self, content: Dict, style: str) -> Dict:
        """Select appropriate template"""
        content_type = content.get('content_type', 'general')
        
        # Get base template
        if content_type == 'story':
            template = self.templates['story']
        elif content_type == 'question':
            template = self.templates['question']
        elif content_type == 'news':
            template = self.templates['news']
        else:
            template = self.templates['general']
        
        # Apply style modifications# models/viral_predictor.py
"""
Viral Content Prediction Model
Uses multiple signals to predict viral potential
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime

class ViralPredictor:
    """
    Predicts viral potential using:
    - Content features (text, sentiment, topics)
    - Temporal features (posting time, trends)
    - Platform-specific signals
    - Historical performance data
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger('viral_predictor')
        
        # Load pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
        
        # Load or initialize viral prediction head
        self.prediction_head = ViralPredictionHead().to(device)
        
        if model_path:
            self.load_model(model_path)
        
        # Platform-specific weights
        self.platform_weights = {
            'reddit': {'text': 0.4, 'timing': 0.2, 'sentiment': 0.2, 'trends': 0.2},
            'twitter': {'text': 0.3, 'timing': 0.3, 'sentiment': 0.2, 'trends': 0.2},
            'tiktok': {'text': 0.2, 'timing': 0.3, 'sentiment': 0.2, 'trends': 0.3},
            'instagram': {'text': 0.2, 'timing': 0.3, 'sentiment': 0.3, 'trends': 0.2}
        }
        
        # Viral indicators database
        self.viral_indicators = self._load_viral_indicators()
        
    def predict(self, content: Dict, platform: str) -> Dict:
        """Predict viral potential of content"""
        # Extract features
        features = self._extract_features(content, platform)
        
        # Get text embeddings
        text_embedding = self._get_text_embedding(content.get('text', ''))
        
        # Combine all features
        combined_features = torch.cat([
            text_embedding,
            torch.tensor(features['temporal'], device=self.device),
            torch.tensor(features['engagement'], device=self.device),
            torch.tensor(features['sentiment'], device=self.device)
        ])
        
        # Make prediction
        with torch.no_grad():
            viral_score = self.prediction_head(combined_features)
            confidence = self._calculate_confidence(features)
        
        # Calculate sub-scores
        sub_scores = self._calculate_sub_scores(features, platform)
        
        return {
            'score': float(viral_score.item()),
            'confidence': float(confidence),
            'sub_scores': sub_scores,
            'key_factors': self._identify_key_factors(features, viral_score),
            'recommendations': self._generate_recommendations(features, viral_score, platform)
        }
    
    def _extract_features(self, content: Dict, platform: str) -> Dict:
        """Extract all features from content"""
        features = {
            'temporal': self._extract_temporal_features(content),
            'engagement': self._extract_engagement_features(content, platform),
            'sentiment': self._extract_sentiment_features(content),
            'content': self._extract_content_features(content),
            'trends': self._extract_trend_features(content, platform)
        }
        
        return features
    
    def _extract_temporal_features(self, content: Dict) -> np.ndarray:
        """Extract time-based features"""
        features = []
        
        # Hour of day (0-23)
        if 'created_utc' in content:
            dt = datetime.fromtimestamp(content['created_utc'])
            features.append(dt.hour / 23.0)
            features.append(dt.weekday() / 6.0)  # Day of week
            features.append(int(dt.weekday() >= 5))  # Is weekend
        else:
            features.extend([0.5, 0.5, 0])
        
        # Time since creation (freshness)
        age_hours = (time.time() - content.get('created_utc', time.time())) / 3600
        features.append(min(age_hours / 168, 1.0))  # Normalize to week
        
        return np.array(features, dtype=np.float32)
    
    def _extract_engagement_features(self, content: Dict, platform: str) -> np.ndarray:
        """Extract engagement-based features"""
        features = []
        
        if platform == 'reddit':
            score = content.get('score', 0)
            comments = content.get('num_comments', 0)
            
            # Normalize scores
            features.append(min(score / 10000, 1.0))
            features.append(min(comments / 1000, 1.0))
            
            # Engagement ratio
            if score > 0:
                features.append(min(comments / score, 1.0))
            else:
                features.append(0)
            
            # Awards (strong viral signal)
            features.append(min(content.get('awards', 0) / 10, 1.0))
            
        elif platform == 'twitter':
            likes = content.get('like_count', 0)
            retweets = content.get('retweet_count', 0)
            replies = content.get('reply_count', 0)
            
            features.append(min(likes / 10000, 1.0))
            features.append(min(retweets / 5000, 1.0))
            features.append(min(replies / 1000, 1.0))
            
            # Retweet ratio (virality indicator)
            if likes > 0:
                features.append(min(retweets / likes, 1.0))
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_sentiment_features(self, content: Dict) -> np.ndarray:
        """Extract sentiment and emotion features"""
        from textblob import TextBlob
        
        text = content.get('text', '') or content.get('title', '')
        
        if not text:
            return np.zeros(5, dtype=np.float32)
        
        # Basic sentiment
        blob = TextBlob(text)
        polarity = (blob.sentiment.polarity + 1) / 2  # Normalize to 0-1
        subjectivity = blob.sentiment.subjectivity
        
        # Emotional triggers
        emotions = {
            'surprise': len([w for w in ['wow', 'amazing', 'incredible', 'unbelievable'] if w in text.lower()]),
            'controversy': len([w for w in ['controversial', 'debate', 'opinion', 'unpopular'] if w in text.lower()]),
            'urgency': len([w for w in ['breaking', 'urgent', 'now', 'today'] if w in text.lower()])
        }
        
        features = [
            polarity,
            subjectivity,
            min(emotions['surprise'] / 3, 1.0),
            min(emotions['controversy'] / 3, 1.0),
            min(emotions['urgency'] / 3, 1.0)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_content_features(self, content: Dict) -> Dict:
        """Extract content structure features"""
        text = content.get('text', '') or content.get('title', '')
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'question_marks': text.count('?'),
            'exclamations': text.count('!'),
            'capitals_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'has_media': int(bool(content.get('url') and not content['url'].startswith('https://www.reddit.com'))),
            'has_video': int(content.get('is_video', False))
        }
        
        return features
    
    def _extract_trend_features(self, content: Dict, platform: str) -> np.ndarray:
        """Extract trending topic features"""
        text = content.get('text', '') or content.get('title', '')
        
        # Check for trending keywords
        trending_score = 0
        for keyword, weight in self.viral_indicators['trending_keywords'].items():
            if keyword.lower() in text.lower():
                trending_score += weight
        
        # Check for viral hashtags
        hashtag_score = 0
        if platform in ['twitter', 'instagram', 'tiktok']:
            hashtags = self._extract_hashtags(text)
            for tag in hashtags:
                if tag in self.viral_indicators['viral_hashtags']:
                    hashtag_score += 1
        
        features = [
            min(trending_score, 1.0),
            min(hashtag_score / 5, 1.0)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for text"""
        if not text:
            return torch.zeros(768, device=self.device)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding
    
    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate prediction confidence"""
        confidence_factors = []
        
        # Data completeness
        data_completeness = sum(
            1 for v in features.values() 
            if v is not None and (not isinstance(v, np.ndarray) or v.size > 0)
        ) / len(features)
        confidence_factors.append(data_completeness)
        
        # Feature strength
        engagement_strength = np.mean(features['engagement'])
        confidence_factors.append(engagement_strength)
        
        # Trend alignment
        trend_strength = np.mean(features['trends'])
        confidence_factors.append(trend_strength)
        
        return float(np.mean(confidence_factors))
    
    def _calculate_sub_scores(self, features: Dict, platform: str) -> Dict:
        """Calculate detailed sub-scores"""
        weights = self.platform_weights[platform]
        
        return {
            'content_quality': float(np.mean(features['sentiment']) * weights['text']),
            'timing_score': float(np.mean(features['temporal']) * weights['timing']),
            'sentiment_score': float(features['sentiment'][0] * weights['sentiment']),
            'trend_alignment': float(np.mean(features['trends']) * weights['trends'])
        }
    
    def _identify_key_factors(self, features: Dict, viral_score: torch.Tensor) -> List[str]:
        """Identify key factors contributing to viral score"""
        factors = []
        
        # High engagement
        if np.mean(features['engagement']) > 0.7:
            factors.append("High existing engagement")
        
        # Good timing
        temporal = features['temporal']
        if temporal[0] > 0.6 and temporal[0] < 0.9:  # Peak hours
            factors.append("Posted during peak hours")
        
        # Emotional content
        if features['sentiment'][2] > 0.5 or features['sentiment'][3] > 0.5:
            factors.append("Strong emotional triggers")
        
        # Trending alignment
        if features['trends'][0] > 0.5:
            factors.append("Contains trending topics")
        
        return factors
    
    def _generate_recommendations(self, features: Dict, viral_score: torch.Tensor, platform: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        score = viral_score.item()
        
        if score < 50:
            # Low viral potential
            if features['sentiment'][0] < 0.5:
                recommendations.append("Add more positive emotional content")
            
            if features['content']['question_marks'] == 0:
                recommendations.append("Consider asking engaging questions")
            
            if features['trends'][0] < 0.3:
                recommendations.append("Incorporate trending topics or hashtags")
        
        elif score < 75:
            # Medium potential
            if features['temporal'][0] < 0.5 or features['temporal'][0] > 0.9:
                recommendations.append("Post during peak engagement hours (6-9 PM)")
            
            if not features['content']['has_media']:
                recommendations.append("Add images or video for higher engagement")
        
        else:
            # High potential
            recommendations.append("Content has high viral potential - post immediately!")
            
            if platform == 'tiktok' and features['trends'][1] < 0.5:
                recommendations.append("Use trending sounds/hashtags for maximum reach")
        
        return recommendations
    
    def _load_viral_indicators(self) -> Dict:
        """Load database of viral indicators"""
        return {
            'trending_keywords': {
                'breaking': 0.3,
                'just in': 0.3,
                'update': 0.2,
                'exclusive': 0.25,
                'leaked': 0.3,
                'shocking': 0.25,
                'unbelievable': 0.2,
                'you won\'t believe': 0.25
            },
            'viral_hashtags': {
                '#fyp', '#foryou', '#viral', '#trending',
                '#breakingnews', '#mustwatch', '#omg'
            }
        }
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        import re
        return re.findall(r'#\w+', text)
    
    def load_model(self, path: str):
        """Load saved model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.prediction_head.load_state_dict(checkpoint['prediction_head'])
        self.logger.info(f"Loaded model from {path}")
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'prediction_head': self.prediction_head.state_dict()
        }, path)


class ViralPredictionHead(nn.Module):
    """Neural network head for viral prediction"""
    
    def __init__(self, input_dim: int = 768 + 20):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x) * 100  # Scale to 0-100


# models/content_analyzer.py
"""
Content Analysis Model
Deep analysis of scraped content
"""

import torch
import numpy as np
from typing import Dict, List, Any
from transformers import pipeline
import spacy
from collections import Counter
import logging

class ContentAnalyzer:
    """
    Comprehensive content analysis:
    - Topic extraction
    - Sentiment analysis
    - Entity recognition
    - Readability scoring
    - Audience identification
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger('content_analyzer')
        
        # Load models
        self.sentiment_analyzer = pipeline('sentiment-analysis', device=0 if device == 'cuda' else -1)
        self.ner_model = spacy.load('en_core_web_sm')
        self.topic_model = self._load_topic_model()
        
        # Analysis components
        self.readability_calculator = ReadabilityCalculator()
        self.audience_identifier = AudienceIdentifier()
        
    def analyze(self, content: Dict) -> Dict:
        """Perform comprehensive content analysis"""
        text = self._extract_text(content)
        
        if not text:
            return {'error': 'No text content found'}
        
        # Run all analyses
        analysis = {
            'content_id': content.get('id', 'unknown'),
            'sentiment': self._analyze_sentiment(text),
            'emotions': self._analyze_emotions(text),
            'topics': self._extract_topics(text),
            'entities': self._extract_entities(text),
            'keywords': self._extract_keywords(text),
            'readability': self.readability_calculator.calculate(text),
            'audience': self.audience_identifier.identify(text),
            'content_type': self._classify_content_type(content),
            'quality_score': self._calculate_quality_score(text)
        }
        
        return analysis
    
    def _extract_text(self, content: Dict) -> str:
        """Extract all text from content"""
        text_parts = []
        
        # Title
        if 'title' in content:
            text_parts.append(content['title'])
        
        # Body text
        if 'text' in content:
            text_parts.append(content['text'])
        elif 'selftext' in content:
            text_parts.append(content['selftext'])
        elif 'body' in content:
            text_parts.append(content['body'])
        
        # Comments (if analyzing post with comments)
        if 'comments' in content and isinstance(content['comments'], list):
            for comment in content['comments'][:10]:  # Top 10 comments
                if isinstance(comment, dict) and 'body' in comment:
                    text_parts.append(comment['body'])
        
        return ' '.join(text_parts)
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Detailed sentiment analysis"""
        # Overall sentiment
        result = self.sentiment_analyzer(text[:512])  # Truncate for model
        
        # Sentence-level sentiment
        sentences = text.split('.')[:10]  # Analyze first 10 sentences
        sentence_sentiments = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                sent_result = self.sentiment_analyzer(sentence)
                sentence_sentiments.append(sent_result[0])
        
        # Calculate sentiment distribution
        positive_ratio = sum(1 for s in sentence_sentiments if s['label'] == 'POSITIVE') / max(len(sentence_sentiments), 1)
        
        return {
            'overall': result[0]['label'].lower(),
            'score': result[0]['score'],
            'positive_ratio': positive_ratio,
            'sentence_sentiments': sentence_sentiments
        }
    
    def _analyze_emotions(self, text: str) -> Dict:
        """Analyze emotional content"""
        # Emotion keywords
        emotions = {
            'joy': ['happy', 'joy', 'excited', 'amazing', 'wonderful', 'fantastic'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'frustrated', 'outraged'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous'],
            'sadness': ['sad', 'depressed', 'crying', 'miserable', 'heartbroken'],
            'surprise': ['shocked', 'surprised', 'astonished', 'amazed', 'unexpected']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = min(score / 10, 1.0)  # Normalize
        
        # Dominant emotion
        dominant = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'scores': emotion_scores,
            'dominant': dominant[0] if dominant[1] > 0 else 'neutral'
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        # Simple topic extraction using noun phrases
        doc = self.ner_model(text)
        
        # Extract noun phrases
        topics = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Max 3 words
                topics.append(chunk.text.lower())
        
        # Count frequency
        topic_counts = Counter(topics)
        
        # Return top topics
        return [topic for topic, _ in topic_counts.most_common(10)]
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities"""
        doc = self.ner_model(text)
        
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'other': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['people'].append(ent.text)
            elif ent.label_ in ['ORG', 'COMPANY']:
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['LOC', 'GPE']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            else:
                entities['other'].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF approach"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Simple keyword extraction
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = [(feature_names[i], scores[i]) for i in range(len(scores))]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, _ in keyword_scores[:10]]
        except:
            # Fallback to simple word frequency
            words = text.lower().split()
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(10) if len(word) > 3]
    
    def _classify_content_type(self, content: Dict) -> str:
        """Classify type of content"""
        text = self._extract_text(content)
        
        # Check for questions
        if '?' in text[:100]:  # Question in beginning
            return 'question'
        
        # Check for stories
        story_indicators = ['tifu', 'story time', 'this happened', 'years ago']
        if any(indicator in text.lower() for indicator in story_indicators):
            return 'story'
        
        # Check for news
        news_indicators = ['breaking', 'announced', 'report', 'according to']
        if any(indicator in text.lower() for indicator in news_indicators):
            return 'news'
        
        # Check for opinion
        opinion_indicators = ['i think', 'in my opinion', 'unpopular opinion', 'cmv']
        if any(indicator in text.lower() for indicator in opinion_indicators):
            return 'opinion'
        
        return 'general'
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate overall content quality score"""
        scores = []
        
        # Length score
        word_count = len(text.split())
        if 100 <= word_count <= 500:
            scores.append(1.0)
        elif 50 <= word_count <= 1000:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # Grammar score (simplified)
        doc = self.ner_model(text)
        grammar_errors = sum(1 for token in doc if token.dep_ == 'ROOT' and token.pos_ != 'VERB')
        grammar_score = max(0, 1 - (grammar_errors / 10))
        scores.append(grammar_score)
        
        # Readability score
        readability = self.readability_calculator.calculate(text)
        if 60 <= readability['flesch_ease'] <= 80:
            scores.append(1.0)
        elif 40 <= readability['flesch_ease'] <= 90:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        return float(np.mean(scores) * 100)
    
    def _load_topic_model(self):
        """Load topic modeling component"""
        # Placeholder for actual topic model
        return None


class ReadabilityCalculator:
    """Calculate readability metrics"""
    
    def calculate(self, text: str) -> Dict:
        """Calculate various readability scores"""
        sentences = text.split('.')
        words = text.split()
        
        # Basic metrics
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Syllable counting (simplified)
        total_syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / max(len(words), 1)
        
        # Flesch Reading Ease
        flesch_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        flesch_ease = max(0, min(100, flesch_ease))  # Bound between 0-100
        
        # Grade level
        grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        grade_level = max(0, min(18, grade_level))  # Bound between 0-18
        
        return {
            'flesch_ease': flesch_ease,
            'grade_level': grade_level,
            'avg_sentence_length': avg_sentence_length,
            'complexity': 'simple' if flesch_ease > 60 else 'complex'
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiou'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Minimum of 1 syllable
        return max(1, syllable_count)


class AudienceIdentifier:
    """Identify target audience for content"""
    
    def identify(self, text: str) -> Dict:
        """Identify likely audience characteristics"""
        text_lower = text.lower()
        
        # Age indicators
        age_groups = {
            'teen': ['high school', 'homework', 'parents', 'teenager', 'prom'],
            'young_adult': ['college', 'university', 'student loan', 'internship', 'entry level'],
            'adult': ['mortgage', 'career', 'marriage', 'kids', 'retirement'],
            'senior': ['grandchildren', 'retired', 'medicare', 'pension']
        }
        
        detected_age = 'general'
        for age, keywords in age_groups.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_age = age
                break
        
        # Interest indicators
        interests = {
            'tech': ['programming', 'software', 'ai', 'crypto', 'startup'],
            'gaming': ['game', 'ps5', 'xbox', 'nintendo', 'steam'],
            'finance': ['invest', 'stock', 'crypto', 'money', 'wealth'],
            'fitness': ['workout', 'gym', 'diet', 'fitness', 'health'],
            'entertainment': ['movie', 'show', 'netflix', 'music', 'concert']
        }
        
        detected_interests = []
        for interest, keywords in interests.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_interests.append(interest)
        
        # Gender indicators (be careful with assumptions)
        gender_neutral_score = 1.0
        
        return {
            'age_group': detected_age,
            'interests': detected_interests[:3],  # Top 3 interests
            'gender_neutral_score': gender_neutral_score,
            'complexity_level': 'advanced' if len(text.split()) > 200 else 'casual'
        }


# models/script_generator.py
"""
Script Generation Model
Generates video scripts from analyzed content
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Optional
import logging
import re

class ScriptGenerator:
    """
    Generates engaging video scripts:
    - Multiple style templates
    - Platform-specific formatting
    - Hook generation
    - Call-to-action optimization
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger('script_generator')
        
        # Load language model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
        
        # Script templates
        self.templates = self._load_templates()
        
        # Platform-specific settings
        self.platform_settings = {
            'tiktok': {'max_duration': 60, 'style': 'fast_paced'},
            'instagram': {'max_duration': 90, 'style': 'visual'},
            'youtube': {'max_duration': 180, 'style': 'detailed'}
        }
    
    def generate(self, content: Dict, style: str = 'engaging', 
                 target_duration: int = 60) -> Dict:
        """Generate video script from content"""
        
        # Select template based on content type
        template = self._select_template(content, style)
        
        # Generate script sections
        script = {
            'id': f"script_{int(time.time())}",
            'sections': [],
            'total_duration': 0,
            'word_count': 0,
            'style': style
        }
        
        # Generate hook
        hook = self._generate_hook(content, template)
        script['sections'].append(hook)
        
        # Generate main content
        main_content = self._generate_main_content(content, template, target_duration)
        script['sections'].extend(main_content)
        
        # Generate call-to-action
        cta = self._generate_cta(content, template)
        script['sections'].append(cta)
        
        # Calculate totals
        script['total_duration'] = sum(s['duration'] for s in script['sections'])
        script['word_count'] = sum(s['word_count'] for s in script['sections'])
        
        # Extract key elements
        script['hooks'] = [hook['text']]
        script['cta'] = cta['text']
        
        # Add production notes
        script['production_notes'] = self._generate_production_notes(script)
        
        return script
    
    def _select_template(self, content: Dict, style: str) -> Dict:
        """Select appropriate template"""
        content_type = content.get('content_type', 'general')
        
        # Get base template
        if content_type == 'story':
            template = self.templates['story']
        elif content_type == 'question':
            template = self.templates['question']
        elif content_type == 'news':
            template = self.templates['news']
        else:
            template = self.templates['general']
        
        # Apply style modifications
        if style == 'fast_paced':
            template['pace'] = 'quick'
            template['sentence_length'] = 'short'
        elif style == 'dramatic':
            template['pace'] = 'varied'
            template['emotion'] = 'high'
        elif style == 'educational':
            template['pace'] = 'moderate'
            template['clarity'] = 'high'
        
        return template
    
    def _generate_hook(self, content: Dict, template: Dict) -> Dict:
        """Generate attention-grabbing hook"""
        hooks = []
        
        # Question hooks
        if content.get('content_type') == 'story':
            hooks.extend([
                "You won't believe what happened when...",
                "This is the craziest story I've heard...",
                "Wait until you hear what this person did..."
            ])
        elif '?' in content.get('title', ''):
            hooks.extend([
                "Here's the answer everyone's looking for...",
                "The truth might surprise you...",
                "Let's settle this once and for all..."
            ])
        
        # Emotion-based hooks
        emotions = content.get('emotions', {}).get('dominant', 'neutral')
        if emotions == 'surprise':
            hooks.extend([
                "This completely shocked me...",
                "I can't believe this is real...",
                "Nobody expected this to happen..."
            ])
        elif emotions == 'joy':
            hooks.extend([
                "This will make your day...",
                "The most wholesome thing happened...",
                "Get ready to smile..."
            ])
        
        # Select best hook
        hook_text = self._select_best_hook(hooks, content) if hooks else "Here's something interesting..."
        
        return {
            'type': 'hook',
            'text': hook_text,
            'duration': 3,
            'word_count': len(hook_text.split()),
            'delivery': 'energetic',
            'visual_note': 'Quick cuts, zoom in'
        }
    
    def _generate_main_content(self, content: Dict, template: Dict, 
                              target_duration: int) -> List[Dict]:
        """Generate main script content"""
        sections = []
        remaining_duration = target_duration - 8  # Reserve time for hook and CTA
        
        # Extract key points
        key_points = self._extract_key_points(content)
        
        # Distribute time among points
        time_per_point = remaining_duration / max(len(key_points), 1)
        
        for i, point in enumerate(key_points):
            # Generate section text
            if template.get('pace') == 'quick':
                section_text = self._condense_point(point)
            else:
                section_text = self._expand_point(point)
            
            # Calculate speaking duration (150 words per minute average)
            word_count = len(section_text.split())
            duration = (word_count / 150) * 60
            
            section = {
                'type': 'main_content',
                'text': section_text,
                'duration': min(duration, time_per_point),
                'word_count': word_count,
                'delivery': template.get('delivery', 'conversational'),
                'visual_note': self._get_visual_suggestion(point, i)
            }
            
            sections.append(section)
            
            # Add transition if not last point
            if i < len(key_points) - 1:
                transition = self._generate_transition(i, template)
                sections.append(transition)
        
        return sections
    
    def _generate_cta(self, content: Dict, template: Dict) -> Dict:
        """Generate call-to-action"""
        platform = content.get('platform', 'general')
        
        cta_options = {
            'tiktok': [
                "Follow for more stories like this!",
                "What would you do? Let me know below!",
                "Share this if it surprised you too!"
            ],
            'instagram': [
                "Double tap if you agree!",
                "Save this for later!",
                "Tag someone who needs to see this!"
            ],
            'youtube': [
                "Subscribe for more content like this!",
                "What's your take? Comment below!",
                "Hit the bell for notifications!"
            ]
        }
        
        # Select appropriate CTA
        cta_list = cta_options.get(platform, ["Thanks for watching!"])
        cta_text = cta_list[0]  # Can randomize later
        
        return {
            'type': 'cta',
            'text': cta_text,
            'duration': 3,
            'word_count': len(cta_text.split()),
            'delivery': 'upbeat',
            'visual_note': 'End screen with subscribe button'
        }
    
    def _extract_key_points(self, content: Dict) -> List[Dict]:
        """Extract main points from content"""
        key_points = []
        
        # Get main text
        text = content.get('text', '') or content.get('selftext', '')
        
        if not text:
            # Use title as single point
            key_points.append({
                'text': content.get('title', ''),
                'importance': 'high'
            })
        else:
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Take most important paragraphs
            for i, para in enumerate(paragraphs[:5]):  # Max 5 points
                if len(para) > 50:  # Meaningful content
                    key_points.append({
                        'text': para,
                        'importance': 'high' if i == 0 else 'medium'
                    })
        
        # Add top comments if available
        if 'comments' in content and content['comments']:
            top_comment = content['comments'][0]
            if isinstance(top_comment, dict) and 'body' in top_comment:
                key_points.append({
                    'text': f"Top comment: {top_comment['body']}",
                    'importance': 'medium'
                })
        
        return key_points
    
    def _condense_point(self, point: Dict) -> str:
        """Condense point for fast-paced delivery"""
        text = point['text']
        
        # Remove filler words
        filler_words = ['basically', 'actually', 'literally', 'just', 'really']
        for word in filler_words:
            text = text.replace(f' {word} ', ' ')
        
        # Shorten sentences
        sentences = text.split('.')
        condensed = []
        
        for sentence in sentences[:3]:  # Max 3 sentences
            if len(sentence.split()) > 15:
                # Take first 15 words
                words = sentence.split()[:15]
                condensed.append(' '.join(words) + '...')
            else:
                condensed.append(sentence)
        
        return '. '.join(condensed)
    
    def _expand_point(self, point: Dict) -> str:
        """Expand point with additional context"""
        text = point['text']
        
        # Add emphasis phrases
        if point['importance'] == 'high':
            text = f"This is important - {text}"
        
        # Add clarification
        if '?' in text:
            text += " Let me explain..."
        
        return text
    
    def _generate_transition(self, index: int, template: Dict) -> Dict:
        """Generate transition between sections"""
        transitions = [
            "But wait, it gets better...",
            "Here's where it gets interesting...",
            "But that's not all...",
            "And then...",
            "What happened next..."
        ]
        
        transition_text = transitions[index % len(transitions)]
        
        return {
            'type': 'transition',
            'text': transition_text,
            'duration': 1,
            'word_count': len(transition_text.split()),
            'delivery': 'building',
            'visual_note': 'Quick transition effect'
        }
    
    def _get_visual_suggestion(self, point: Dict, index: int) -> str:
        """Suggest visuals for script section"""
        suggestions = [
            "Text overlay with key words highlighted",
            "Relevant stock footage or images",
            "Animated text reveal",
            "Split screen with reaction",
            "Zoom in on important details"
        ]
        
        # Match visual to content
        if 'shock' in point['text'].lower():
            return "Shocked reaction GIF/video"
        elif 'happy' in point['text'].lower():
            return "Celebration or happy footage"
        elif '?' in point['text']:
            return "Question mark animation"
        
        return suggestions[index % len(suggestions)]
    
    def _select_best_hook(self, hooks: List[str], content: Dict) -> str:
        """Select most appropriate hook"""
        # Score each hook based on relevance
        scored_hooks = []
        
        for hook in hooks:
            score = 0
            
            # Length score (shorter is better for hooks)
            score += (20 - len(hook.split())) * 0.1
            
            # Emotion alignment
            if content.get('emotions', {}).get('dominant') in hook.lower():
                score += 2
            
            # Question alignment
            if '?' in hook and '?' in content.get('title', ''):
                score += 1
            
            scored_hooks.append((hook, score))
        
        # Sort by score and return best
        scored_hooks.sort(key=lambda x: x[1], reverse=True)
        return scored_hooks[0][0]
    
    def _generate_production_notes(self, script: Dict) -> Dict:
        """Generate production notes for video creation"""
        notes = {
            'pacing': [],
            'visuals': [],
            'audio': [],
            'effects': []
        }
        
        # Analyze script for pacing
        total_duration = script['total_duration']
        if total_duration < 30:
            notes['pacing'].append("Very fast paced - quick cuts every 2-3 seconds")
        elif total_duration < 60:
            notes['pacing'].append("Moderate pace - cuts every 4-5 seconds")
        else:
            notes['pacing'].append("Relaxed pace - longer shots allowed")
        
        # Visual recommendations
        if script['style'] == 'fast_paced':
            notes['visuals'].extend([
                "Use jump cuts and zooms",
                "Add motion graphics",
                "Include reaction clips"
            ])
        elif script['style'] == 'dramatic':
            notes['visuals'].extend([
                "Use dramatic lighting",
                "Slow motion for emphasis",
                "Close-up shots for emotion"
            ])
        
        # Audio recommendations
        notes['audio'].extend([
            "Background music: upbeat/energetic" if script['style'] == 'fast_paced' else "subtle/atmospheric",
            "Sound effects for transitions",
            "Emphasize key words with audio cues"
        ])
        
        # Effects based on platform
        platform = script.get('platform', 'general')
        if platform == 'tiktok':
            notes['effects'].extend([
                "Use trending TikTok effects",
                "Add captions with pop animation",
                "Include emoji reactions"
            ])
        
        return notes
    
    def _load_templates(self) -> Dict:
        """Load script templates"""
        return {
            'story': {
                'structure': ['hook', 'setup', 'conflict', 'resolution', 'cta'],
                'pace': 'moderate',
                'emotion': 'high',
                'delivery': 'narrative'
            },
            'question': {
                'structure': ['hook', 'question', 'exploration', 'answer', 'cta'],
                'pace': 'moderate',
                'clarity': 'high',
                'delivery': 'explanatory'
            },
            'news': {
                'structure': ['hook', 'headline', 'details', 'impact', 'cta'],
                'pace': 'quick',
                'clarity': 'high',
                'delivery': 'informative'
            },
            'general': {
                'structure': ['hook', 'main_points', 'cta'],
                'pace': 'moderate',
                'delivery': 'conversational'
            }
        }


# Example usage and integration
if __name__ == "__main__":
    # Initialize models
    viral_predictor = ViralPredictor(device='cuda' if torch.cuda.is_available() else 'cpu')
    content_analyzer = ContentAnalyzer(device='cuda' if torch.cuda.is_available() else 'cpu')
    script_generator = ScriptGenerator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example content
    example_content = {
        'id': 'test123',
        'title': 'TIFU by accidentally sending my boss a meme meant for my friend',
        'text': 'So this happened yesterday and I\'m still mortified...',
        'score': 5432,
        'num_comments': 234,
        'created_utc': time.time() - 3600,
        'platform': 'reddit'
    }
    
    # Analyze content
    print("Analyzing content...")
    analysis = content_analyzer.analyze(example_content)
    print(f"Sentiment: {analysis['sentiment']['overall']}")
    print(f"Topics: {analysis['topics'][:5]}")
    
    # Predict virality
    print("\nPredicting viral potential...")
    viral_prediction = viral_predictor.predict(example_content, 'reddit')
    print(f"Viral Score: {viral_prediction['score']:.1f}/100")
    print(f"Key Factors: {', '.join(viral_prediction['key_factors'])}")
    
    # Generate script if viral
    if viral_prediction['score'] > 70:
        print("\nGenerating video script...")
        script = script_generator.generate(
            {**example_content, **analysis},
            style='fast_paced',
            target_duration=60
        )
        
        print(f"\nScript sections: {len(script['sections'])}")
        print(f"Total duration: {script['total_duration']} seconds")
        print(f"\nHook: {script['hooks'][0]}")
        print(f"CTA: {script['cta']}")# models/viral_predictor.py
"""
Viral Content Prediction Model
Uses multiple signals to predict viral potential
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime

class ViralPredictor:
    """
    Predicts viral potential using:
    - Content features (text, sentiment, topics)
    - Temporal features (posting time, trends)
    - Platform-specific signals
    - Historical performance data
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger('viral_predictor')
        
        # Load pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
        
        # Load or initialize viral prediction head
        self.prediction_head = ViralPredictionHead().to(device)
        
        if model_path:
            self.load_model(model_path)
        
        # Platform-specific weights
        self.platform_weights = {
            'reddit': {'text': 0.4, 'timing': 0.2, 'sentiment': 0.2, 'trends': 0.2},
            'twitter': {'text': 0.3, 'timing': 0.3, 'sentiment': 0.2, 'trends': 0.2},
            'tiktok': {'text': 0