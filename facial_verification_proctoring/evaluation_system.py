import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class EvaluationSystem:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def preprocess_text(self, text):
        """Preprocess text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
        
    def evaluate_answer(self, student_answer, model_answer, max_marks):
        """Evaluate a theory answer against a model answer"""
        # Preprocess both answers
        processed_student = self.preprocess_text(student_answer)
        processed_model = self.preprocess_text(model_answer)
        
        # Vectorize the answers
        vectors = self.vectorizer.fit_transform([processed_student, processed_model])
        
        # Calculate similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        # Calculate marks based on similarity
        marks = similarity * max_marks
        
        # Additional checks for answer quality
        quality_score = self._check_answer_quality(student_answer)
        
        # Final marks (80% similarity, 20% quality)
        final_marks = (marks * 0.8) + (quality_score * max_marks * 0.2)
        
        return {
            'marks': round(final_marks, 2),
            'similarity_score': round(similarity, 2),
            'quality_score': round(quality_score, 2),
            'feedback': self._generate_feedback(similarity, quality_score)
        }
        
    def _check_answer_quality(self, answer):
        """Check the quality of the answer"""
        # Tokenize into sentences
        sentences = sent_tokenize(answer)
        
        # Calculate average sentence length
        avg_sentence_length = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
        
        # Check for key indicators of good answers
        indicators = {
            'examples': len(re.findall(r'for example|for instance|such as', answer.lower())),
            'explanations': len(re.findall(r'because|therefore|thus|hence', answer.lower())),
            'structure': len(re.findall(r'first|second|finally|in conclusion', answer.lower()))
        }
        
        # Calculate quality score (0-1)
        quality_score = (
            0.4 * min(avg_sentence_length / 15, 1) +  # Sentence length (max 15 words)
            0.2 * min(indicators['examples'] / 2, 1) +  # Examples (max 2)
            0.2 * min(indicators['explanations'] / 2, 1) +  # Explanations (max 2)
            0.2 * min(indicators['structure'] / 2, 1)  # Structure indicators (max 2)
        )
        
        return min(quality_score, 1)
        
    def _generate_feedback(self, similarity, quality_score):
        """Generate feedback based on evaluation metrics"""
        feedback = []
        
        # Similarity feedback
        if similarity < 0.3:
            feedback.append("Your answer shows low similarity to the expected response.")
        elif similarity < 0.6:
            feedback.append("Your answer shows moderate similarity to the expected response.")
        else:
            feedback.append("Your answer shows high similarity to the expected response.")
            
        # Quality feedback
        if quality_score < 0.3:
            feedback.append("Consider providing more examples and explanations.")
        elif quality_score < 0.6:
            feedback.append("Good effort! Try to structure your answer better with clear points.")
        else:
            feedback.append("Well-structured answer with good examples and explanations.")
            
        return ' '.join(feedback) 