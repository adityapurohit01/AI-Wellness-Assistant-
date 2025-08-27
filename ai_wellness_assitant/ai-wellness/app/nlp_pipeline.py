"""
Medical NLP Pipeline with Smart Fallbacks
Handles missing dependencies gracefully and provides demonstration functionality
"""

import logging
from typing import Dict, List, Optional
import re
import json

# Try to import advanced NLP libraries with fallbacks
try:
    import spacy
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import scispacy
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalNLPPipeline:
    """Medical NLP Pipeline with graceful fallbacks for missing dependencies"""

    def __init__(self):
        """Initialize the NLP pipeline with available components"""
        self.nlp = None
        self.intent_classifier = None
        self.capabilities = {
            'spacy_core': False,
            'scispacy_medical': False,
            'umls_linker': False,
            'intent_classifier': False
        }
        self._initialize_components()

    def _initialize_components(self):
        """Initialize available NLP components"""

        # Try to load spaCy
        if SPACY_AVAILABLE:
            try:
                # Try different spaCy models in order of preference
                models_to_try = ['en_core_sci_sm', 'en_core_web_sm', 'en_core_web_md']

                for model in models_to_try:
                    try:
                        self.nlp = spacy.load(model)
                        self.capabilities['spacy_core'] = True
                        if 'sci' in model:
                            self.capabilities['scispacy_medical'] = True
                        logger.info(f"✅ Loaded spaCy model: {model}")
                        break
                    except IOError:
                        continue

                # Basic UMLS-like functionality
                if SCISPACY_AVAILABLE and self.nlp:
                    try:
                        # Add custom attribute extensions
                        if not Doc.has_extension("umls_ents"):
                            Doc.set_extension("umls_ents", default=[])
                        if not Span.has_extension("umls_ents"):
                            Span.set_extension("umls_ents", default=[])
                        
                        # Mark as having UMLS capabilities
                        self.capabilities['umls_linker'] = True
                        logger.info("✅ Added UMLS-like entity support")
                    except Exception as e:
                        logger.warning(f"UMLS support not available: {e}")

            except Exception as e:
                logger.warning(f"spaCy initialization failed: {e}")

        # Try to load intent classifier
        if TRANSFORMERS_AVAILABLE:
            try:
                self.intent_classifier = pipeline("text-classification", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment",
                                                 return_all_scores=True)
                self.capabilities['intent_classifier'] = True
                logger.info("✅ Loaded intent classifier")
            except Exception as e:
                logger.warning(f"Intent classifier not available: {e}")

        logger.info(f"NLP Pipeline initialized with capabilities: {self.capabilities}")

    def get_capabilities(self) -> Dict[str, bool]:
        """Get current pipeline capabilities"""
        return self.capabilities.copy()

    def process_symptoms(self, text: str) -> Dict:
        """
        Process symptoms with available NLP components
        Falls back to rule-based processing if advanced components unavailable
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty")

        logger.info(f"Processing medical text with available components...")

        try:
            if self.nlp and self.capabilities['spacy_core']:
                return self._advanced_nlp_processing(text)
            else:
                return self._fallback_processing(text)

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return self._fallback_processing(text)

    def _advanced_nlp_processing(self, text: str) -> Dict:
        """Advanced NLP processing using spaCy/scispaCy"""
        doc = self.nlp(text)

        # Extract entities
        entities = []
        for ent in doc.ents:
            entity_data = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "cui": None,
                "umls_description": None,
                "confidence": 0.8
            }

            # Try to get UMLS linking if available
            if self.capabilities['umls_linker'] and hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                cui, score = ent._.kb_ents[0]
                entity_data["cui"] = cui
                entity_data["confidence"] = score

                if hasattr(ent._, 'kb_ents_') and ent._.kb_ents_:
                    entity_data["umls_description"] = getattr(
                        ent._.kb_ents_[0], 'canonical_name', 'N/A'
                    )

            entities.append(entity_data)

        # Classify intent
        intent = self._classify_intent(text)

        # Map to conditions
        conditions = self._map_to_conditions(entities, text)

        return {
            "original_text": text,
            "intent": intent,
            "medical_entities": entities,
            "probable_conditions": conditions,
            "entity_count": len(entities),
            "confidence": self._calculate_confidence(entities, intent),
            "processing_method": "advanced_nlp"
        }

    def _fallback_processing(self, text: str) -> Dict:
        """Fallback processing using rule-based medical entity recognition"""
        logger.info("Using fallback rule-based processing")

        # Medical keyword dictionary with realistic medical information
        medical_entities = {
            # Symptoms
            'tired': {'type': 'SYMPTOM', 'cui': 'C0015672', 'description': 'Fatigue'},
            'fatigue': {'type': 'SYMPTOM', 'cui': 'C0015672', 'description': 'Fatigue'},
            'dizzy': {'type': 'SYMPTOM', 'cui': 'C0012833', 'description': 'Dizziness'},  
            'dizziness': {'type': 'SYMPTOM', 'cui': 'C0012833', 'description': 'Dizziness'},
            'headache': {'type': 'SYMPTOM', 'cui': 'C0018681', 'description': 'Headache'},
            'nausea': {'type': 'SYMPTOM', 'cui': 'C0027497', 'description': 'Nausea'},
            'nauseous': {'type': 'SYMPTOM', 'cui': 'C0027497', 'description': 'Nausea'},
            'fever': {'type': 'SYMPTOM', 'cui': 'C0015967', 'description': 'Fever'},
            'pain': {'type': 'SYMPTOM', 'cui': 'C0030193', 'description': 'Pain'},
            'ache': {'type': 'SYMPTOM', 'cui': 'C0030193', 'description': 'Pain'},
            'cough': {'type': 'SYMPTOM', 'cui': 'C0010200', 'description': 'Cough'},
            'chest pain': {'type': 'SYMPTOM', 'cui': 'C0008031', 'description': 'Chest Pain'},
            'back pain': {'type': 'SYMPTOM', 'cui': 'C0004604', 'description': 'Back Pain'},
            'stomach pain': {'type': 'SYMPTOM', 'cui': 'C0024905', 'description': 'Abdominal Pain'},
            'anxious': {'type': 'SYMPTOM', 'cui': 'C0003467', 'description': 'Anxiety'},
            'anxiety': {'type': 'SYMPTOM', 'cui': 'C0003467', 'description': 'Anxiety'},
            'insomnia': {'type': 'SYMPTOM', 'cui': 'C0917801', 'description': 'Insomnia'},
            'trouble sleeping': {'type': 'SYMPTOM', 'cui': 'C0917801', 'description': 'Sleep Disorder'},
            'difficulty breathing': {'type': 'SYMPTOM', 'cui': 'C0013404', 'description': 'Dyspnea'},

            # Body parts
            'head': {'type': 'ANATOMY', 'cui': 'C0018670', 'description': 'Head'},
            'chest': {'type': 'ANATOMY', 'cui': 'C0817096', 'description': 'Chest'},
            'back': {'type': 'ANATOMY', 'cui': 'C0004600', 'description': 'Back'},
            'stomach': {'type': 'ANATOMY', 'cui': 'C0038351', 'description': 'Stomach'},
            'heart': {'type': 'ANATOMY', 'cui': 'C0018787', 'description': 'Heart'},
        }

        # Extract entities using keyword matching
        entities = []
        text_lower = text.lower()

        # Sort by length (longest first) to catch phrases like "chest pain" before "pain"
        sorted_keywords = sorted(medical_entities.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            if keyword in text_lower:
                start_pos = text_lower.find(keyword)
                entity_info = medical_entities[keyword]

                entities.append({
                    "text": keyword,
                    "label": entity_info['type'],
                    "start": start_pos,
                    "end": start_pos + len(keyword),
                    "cui": entity_info['cui'],
                    "umls_description": entity_info['description'],
                    "confidence": 0.7
                })

        # Remove duplicates (keep first occurrence)
        seen_texts = set()
        unique_entities = []
        for entity in entities:
            if entity['text'] not in seen_texts:
                unique_entities.append(entity)
                seen_texts.add(entity['text'])

        # Classify intent
        intent = self._classify_intent(text)

        # Map to conditions
        conditions = self._map_to_conditions(unique_entities, text)

        return {
            "original_text": text,
            "intent": intent,
            "medical_entities": unique_entities,
            "probable_conditions": conditions,
            "entity_count": len(unique_entities),
            "confidence": self._calculate_confidence(unique_entities, intent),
            "processing_method": "rule_based_fallback"
        }

    def _classify_intent(self, text: str) -> str:
        """Classify user intent with medical context"""
        text_lower = text.lower()

        # Emergency keywords
        emergency_keywords = [
            'chest pain', 'difficulty breathing', 'severe pain', 'emergency', 
            'urgent', 'unconscious', 'bleeding', 'stroke', 'heart attack',
            'seizure', 'choking', 'overdose'
        ]

        # Symptom keywords
        symptom_keywords = [
            'feel', 'feeling', 'pain', 'hurt', 'ache', 'tired', 'dizzy', 
            'nausea', 'headache', 'fever', 'cough', 'symptoms', 'sick',
            'unwell', 'discomfort', 'problem'
        ]

        # Question keywords
        question_keywords = ['what', 'how', 'why', 'when', 'where', '?', 'explain']

        # Priority-based classification
        if any(keyword in text_lower for keyword in emergency_keywords):
            return "emergency"
        elif any(keyword in text_lower for keyword in symptom_keywords):
            return "symptom_check"
        elif any(keyword in text_lower for keyword in question_keywords):
            return "general_inquiry"
        else:
            return "symptom_check"  # Default assumption

    def _map_to_conditions(self, entities: List[Dict], text: str) -> List[Dict]:
        """Map entities to probable medical conditions"""
        conditions = []

        # Medical knowledge mapping
        symptom_to_conditions = {
            'fatigue': [
                {'condition': 'Iron Deficiency Anemia', 'probability': 0.3},
                {'condition': 'Hypothyroidism', 'probability': 0.25},
                {'condition': 'Sleep Disorder', 'probability': 0.2},
                {'condition': 'Depression', 'probability': 0.15}
            ],
            'tired': [
                {'condition': 'Iron Deficiency Anemia', 'probability': 0.3},
                {'condition': 'Sleep Disorder', 'probability': 0.25},
                {'condition': 'Hypothyroidism', 'probability': 0.2}
            ],
            'dizziness': [
                {'condition': 'Orthostatic Hypotension', 'probability': 0.35},
                {'condition': 'Inner Ear Problem', 'probability': 0.25},
                {'condition': 'Dehydration', 'probability': 0.2},
                {'condition': 'Anemia', 'probability': 0.15}
            ],
            'dizzy': [
                {'condition': 'Orthostatic Hypotension', 'probability': 0.35},
                {'condition': 'Dehydration', 'probability': 0.25},
                {'condition': 'Inner Ear Problem', 'probability': 0.2}
            ],
            'headache': [
                {'condition': 'Tension Headache', 'probability': 0.4},
                {'condition': 'Migraine', 'probability': 0.25},
                {'condition': 'Eye Strain', 'probability': 0.2},
                {'condition': 'Dehydration', 'probability': 0.15}
            ],
            'nausea': [
                {'condition': 'Gastroenteritis', 'probability': 0.3},
                {'condition': 'Motion Sickness', 'probability': 0.25},
                {'condition': 'Migraine', 'probability': 0.2},
                {'condition': 'Food Poisoning', 'probability': 0.15}
            ],
            'chest pain': [
                {'condition': 'Muscle Strain', 'probability': 0.3},
                {'condition': 'Gastroesophageal Reflux', 'probability': 0.25},
                {'condition': 'Anxiety', 'probability': 0.2},
                {'condition': 'Cardiac Issue (Requires Evaluation)', 'probability': 0.25}
            ],
            'back pain': [
                {'condition': 'Muscle Strain', 'probability': 0.4},
                {'condition': 'Poor Posture', 'probability': 0.3},
                {'condition': 'Herniated Disc', 'probability': 0.2},
                {'condition': 'Arthritis', 'probability': 0.1}
            ],
            'anxiety': [
                {'condition': 'Generalized Anxiety Disorder', 'probability': 0.35},
                {'condition': 'Stress Response', 'probability': 0.3},
                {'condition': 'Panic Disorder', 'probability': 0.2},
                {'condition': 'Depression with Anxiety', 'probability': 0.15}
            ]
        }

        # Collect conditions from entities
        condition_scores = {}

        for entity in entities:
            entity_text = entity['text'].lower()

            # Check for direct matches or partial matches
            for symptom, condition_list in symptom_to_conditions.items():
                if symptom in entity_text or entity_text in symptom:
                    for condition_info in condition_list:
                        condition_name = condition_info['condition']
                        base_prob = condition_info['probability']

                        if condition_name in condition_scores:
                            # Increase probability if multiple supporting symptoms
                            condition_scores[condition_name] = min(0.95, 
                                condition_scores[condition_name] + (base_prob * 0.5))
                        else:
                            condition_scores[condition_name] = base_prob

        # Convert to list format
        for condition, probability in sorted(condition_scores.items(), 
                                           key=lambda x: x[1], reverse=True):
            conditions.append({
                "condition": condition,
                "probability_score": probability,
                "supporting_entities": len([e for e in entities 
                                          if any(symptom in e['text'].lower() 
                                               for symptom in symptom_to_conditions.keys())])
            })

        return conditions[:5]  # Top 5 conditions

    def _calculate_confidence(self, entities: List[Dict], intent: str) -> float:
        """Calculate processing confidence"""
        if not entities:
            return 0.3

        # Base confidence from entity count and quality
        entity_confidence = min(0.8, len(entities) * 0.15)

        # Intent confidence
        intent_confidence = {
            "symptom_check": 0.9,
            "emergency": 0.95,
            "general_inquiry": 0.7
        }.get(intent, 0.6)

        # Entity quality confidence (higher if CUIs available)
        quality_confidence = 0.8 if any(e.get('cui') for e in entities) else 0.6

        final_confidence = (entity_confidence + intent_confidence + quality_confidence) / 3
        return min(1.0, max(0.2, final_confidence))

# Create global instance
medical_nlp = MedicalNLPPipeline()
