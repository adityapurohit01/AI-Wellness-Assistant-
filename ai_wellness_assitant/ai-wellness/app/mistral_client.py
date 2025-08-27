"""
Corrected Mistral Client with Fallbacks
Handles missing Ollama gracefully and provides quality rule-based recommendations
"""

import logging
import time
from typing import Dict, List, Optional

# Try to import Ollama with fallback
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class MistralRecommendationEngine:
    """Mistral-7B recommendation engine with intelligent fallbacks"""

    def __init__(self):
        """Initialize recommendation engine"""
        self.model_name = "mistral:7b"
        self.client = None
        self.ollama_available = False

        if OLLAMA_AVAILABLE:
            try:
                self.client = ollama.Client()
                # Test connection without pulling model
                try:
                    models = self.client.list()
                    self.ollama_available = True
                    logger.info("✅ Ollama connection established")

                    # Check if model is available
                    if any(self.model_name in model.get('name', '') for model in models.get('models', [])):
                        logger.info(f"✅ {self.model_name} model available")
                    else:
                        logger.info(f"⚠️ {self.model_name} model not found (will use fallback)")

                except Exception as e:
                    logger.warning(f"Ollama service not running: {e}")
                    self.ollama_available = False
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}")
        else:
            logger.info("Ollama package not available - using advanced rule-based recommendations")

    def generate_wellness_plan(self, nlp_results: Dict, user_context: Optional[Dict] = None) -> Dict:
        """Generate wellness plan using available AI or fallback to advanced rules"""

        logger.info("Generating wellness plan...")
        start_time = time.time()

        # Try Mistral-7B if available, otherwise use advanced rule-based system
        if self.ollama_available and self.client:
            try:
                wellness_plan = self._generate_with_mistral(nlp_results, user_context)
                wellness_plan['model_used'] = self.model_name
            except Exception as e:
                logger.warning(f"Mistral generation failed: {e}")
                wellness_plan = self._advanced_rule_based_recommendations(nlp_results, user_context)
                wellness_plan['model_used'] = 'advanced_rules_fallback'
        else:
            wellness_plan = self._advanced_rule_based_recommendations(nlp_results, user_context)
            wellness_plan['model_used'] = 'advanced_rule_based'

        # Add metadata
        wellness_plan.update({
            'generation_time_ms': int((time.time() - start_time) * 1000),
            'nlp_confidence': nlp_results.get('confidence', 0.5),
            'ollama_available': self.ollama_available
        })

        return wellness_plan

    def _generate_with_mistral(self, nlp_results: Dict, context: Optional[Dict] = None) -> Dict:
        """Generate recommendations using Mistral-7B"""

        # Build comprehensive prompt
        prompt = self._build_medical_prompt(nlp_results, context)

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 600
                }
            )

            return self._parse_mistral_response(response['message']['content'])

        except Exception as e:
            logger.error(f"Mistral generation error: {e}")
            raise

    def _advanced_rule_based_recommendations(self, nlp_results: Dict, context: Optional[Dict] = None) -> Dict:
        """Generate high-quality recommendations using advanced rule-based system"""

        entities = nlp_results.get('medical_entities', [])
        conditions = nlp_results.get('probable_conditions', [])
        intent = nlp_results.get('intent', 'symptom_check')
        original_text = nlp_results.get('original_text', '')

        # Extract symptom information
        symptoms = [e['text'].lower() for e in entities if e.get('label') == 'SYMPTOM']

        # Generate comprehensive recommendations
        return {
            "condition_summary": self._generate_condition_summary(symptoms, conditions, original_text),
            "precautions": self._generate_precautions(symptoms, intent),
            "yoga_plan": self._generate_yoga_recommendations(symptoms),
            "diet_plan": self._generate_diet_recommendations(symptoms),
            "lifestyle_tips": self._generate_lifestyle_recommendations(symptoms),
            "medication_guidance": self._generate_medical_guidance(symptoms, intent),
            "raw_response": f"Generated using advanced rule-based system with {len(entities)} medical entities"
        }

    def _generate_condition_summary(self, symptoms: List[str], conditions: List[Dict], original_text: str) -> str:
        """Generate comprehensive condition summary"""

        if not symptoms and not conditions:
            return "Thank you for sharing your health concerns. While specific medical symptoms weren't clearly identified in your description, your overall wellness is important and the following recommendations may help support your general health."

        summary_parts = []

        # Primary symptom analysis
        if symptoms:
            primary_symptoms = ", ".join(symptoms[:3])  # Top 3 symptoms
            summary_parts.append(f"Based on your reported symptoms ({primary_symptoms})")

        # Condition analysis
        if conditions:
            top_condition = conditions[0]['condition']
            probability = conditions[0].get('probability_score', 0)
            summary_parts.append(f", you may be experiencing {top_condition.lower()} or related conditions (confidence: {probability:.0%})")
        else:
            summary_parts.append(", you may be experiencing general health concerns that could benefit from lifestyle modifications")

        # Symptom-specific insights
        insight = ""
        if any(s in symptoms for s in ['tired', 'fatigue']):
            insight = " Fatigue can be related to various factors including sleep quality, nutrition, stress levels, or underlying medical conditions."
        elif any(s in symptoms for s in ['dizzy', 'dizziness']):
            insight = " Dizziness often relates to blood pressure changes, dehydration, inner ear issues, or medication effects."
        elif any(s in symptoms for s in ['headache']):
            insight = " Headaches commonly result from tension, eye strain, dehydration, or stress, though other causes should be considered."
        elif any(s in symptoms for s in ['nausea', 'nauseous']):
            insight = " Nausea can indicate digestive issues, motion sensitivity, medication effects, or other medical conditions."
        elif any(s in symptoms for s in ['anxiety', 'anxious']):
            insight = " Anxiety symptoms can be managed through lifestyle changes, stress reduction techniques, and professional support when needed."

        # Combine summary
        full_summary = "".join(summary_parts) + "." + insight

        # Add important disclaimer
        full_summary += " This assessment is for informational purposes only and should not replace professional medical evaluation."

        return full_summary

    def _generate_precautions(self, symptoms: List[str], intent: str) -> str:
        """Generate safety precautions"""

        if intent == "emergency":
            return "🚨 SEEK IMMEDIATE MEDICAL ATTENTION - Your symptoms may indicate a medical emergency. Call emergency services or visit the nearest emergency room immediately."

        precautions = []

        # General precautions
        precautions.append("Monitor your symptoms closely and note any changes in intensity, frequency, or new symptoms that develop.")

        # Symptom-specific precautions
        if any(s in symptoms for s in ['chest pain', 'difficulty breathing']):
            precautions.append("🚨 Chest pain and breathing difficulties require immediate medical evaluation. Do not delay seeking emergency care.")

        if any(s in symptoms for s in ['dizzy', 'dizziness']):
            precautions.extend([
                "Avoid driving or operating machinery until dizziness resolves completely.",
                "Move slowly when changing positions (lying to sitting, sitting to standing) to prevent falls.",
                "Stay well-hydrated and avoid alcohol."
            ])

        if any(s in symptoms for s in ['severe', 'pain']):
            precautions.append("For severe or worsening pain, seek medical evaluation promptly rather than waiting.")

        if any(s in symptoms for s in ['fever']):
            precautions.append("Monitor your temperature regularly and seek medical care if fever exceeds 101.3°F (38.5°C) or persists.")

        # Timeline guidance
        precautions.append("Consult a healthcare provider if symptoms persist for more than 3-5 days, worsen significantly, or interfere with your daily activities.")

        return " ".join(precautions)

    def _generate_yoga_recommendations(self, symptoms: List[str]) -> str:
        """Generate evidence-based yoga and movement recommendations"""

        recommendations = []

        # Base recommendations
        recommendations.extend([
            "Practice gentle, mindful movement for 15-20 minutes daily, focusing on breath awareness and body comfort.",
            "Begin with basic grounding poses: Mountain Pose (Tadasana) for stability and Child's Pose (Balasana) for relaxation."
        ])

        # Symptom-specific yoga
        if any(s in symptoms for s in ['tired', 'fatigue']):
            recommendations.extend([
                "Try gentle energizing sequences: Cat-Cow stretches (Marjaryasana-Bitilasana) to mobilize the spine.",
                "Practice Legs-Up-Wall pose (Viparita Karani) for 5-10 minutes to improve circulation and reduce fatigue.",
                "Include gentle backbends like Camel Pose (Ustrasana) modification against a wall to boost energy.",
                "End with Corpse Pose (Savasana) for complete relaxation and restoration."
            ])

        if any(s in symptoms for s in ['dizzy', 'dizziness']):
            recommendations.extend([
                "Focus on seated or grounding poses initially: Easy Pose (Sukhasana) with spinal breathing.",
                "Practice Tree Pose (Vrksasana) near a wall for balance training once dizziness improves.",
                "Avoid rapid movements, inversions, and quick transitions between poses.",
                "Include neck and shoulder releases to address potential cervical spine contributions."
            ])

        if any(s in symptoms for s in ['headache']):
            recommendations.extend([
                "Practice gentle neck stretches and shoulder rolls to release tension.",
                "Try Forward Fold (Uttanasana) modification with hands on blocks for mild inversion benefits.",
                "Include Eye Yoga: gentle eye movements and palming to reduce eye strain.",
                "Practice Supported Fish Pose (Matsyasana) with bolster to open chest and neck area."
            ])

        if any(s in symptoms for s in ['back pain']):
            recommendations.extend([
                "Focus on spine mobility: Cat-Cow stretches and gentle spinal twists in seated position.",
                "Practice Hip Flexor stretches: Low Lunge (Anjaneyasana) modifications.",
                "Strengthen core gently: Modified Plank and Bridge Pose (Setu Bandhasana).",
                "End with Knee-to-Chest pose (Apanasana) and gentle spinal twists."
            ])

        if any(s in symptoms for s in ['anxiety', 'anxious', 'trouble sleeping']):
            recommendations.extend([
                "Emphasize breathwork: Practice 4-7-8 breathing (inhale 4, hold 7, exhale 8 counts).",
                "Include restorative poses: Supported Child's Pose and Reclined Butterfly (Supta Baddha Konasana).",
                "Practice gentle forward folds for introspection: Seated Forward Fold (Paschimottanasana).",
                "End with extended Savasana (15-20 minutes) with body scan meditation."
            ])

        # Safety reminder
        recommendations.append("Always listen to your body, move within comfortable ranges, and stop if any pose causes pain or worsens symptoms.")

        return " ".join(recommendations)

    def _generate_diet_recommendations(self, symptoms: List[str]) -> str:
        """Generate evidence-based nutritional recommendations"""

        recommendations = []

        # Foundation nutrition
        recommendations.extend([
            "Follow a balanced, whole-foods diet emphasizing fresh fruits, vegetables, whole grains, lean proteins, and healthy fats.",
            "Maintain consistent meal timing with balanced portions to support stable blood sugar and energy levels.",
            "Stay adequately hydrated with 8-10 glasses of water daily, adjusting for activity level and climate."
        ])

        # Symptom-specific nutrition
        if any(s in symptoms for s in ['tired', 'fatigue']):
            recommendations.extend([
                "Include iron-rich foods: lean red meat, poultry, fish, lentils, spinach, and pumpkin seeds.",
                "Add vitamin B12 sources: fish, eggs, dairy, nutritional yeast, and fortified plant milks.",
                "Consume complex carbohydrates for sustained energy: quinoa, brown rice, sweet potatoes, and oats.",
                "Include magnesium-rich foods: almonds, avocados, dark chocolate, and leafy greens.",
                "Consider vitamin D assessment and foods like fatty fish, egg yolks, and fortified foods."
            ])

        if any(s in symptoms for s in ['dizzy', 'dizziness']):
            recommendations.extend([
                "Maintain stable blood sugar with balanced meals every 3-4 hours containing protein, healthy fats, and complex carbs.",
                "Limit caffeine and alcohol, which can affect blood pressure and hydration status.",
                "Include potassium-rich foods: bananas, oranges, potatoes, and yogurt for blood pressure support.",
                "Ensure adequate sodium intake (within healthy limits) especially if you exercise or sweat frequently."
            ])

        if any(s in symptoms for s in ['headache']):
            recommendations.extend([
                "Identify and avoid potential trigger foods: aged cheeses, processed meats, artificial sweeteners, and MSG.",
                "Include magnesium-rich foods: nuts, seeds, dark leafy greens, and dark chocolate.",
                "Maintain regular meal timing to prevent hunger-triggered headaches.",
                "Stay well-hydrated as dehydration is a common headache trigger.",
                "Consider riboflavin (B2) sources: dairy, eggs, leafy greens, and almonds."
            ])

        if any(s in symptoms for s in ['nausea', 'nauseous']):
            recommendations.extend([
                "Try ginger: fresh ginger tea, crystallized ginger, or ginger supplements for nausea relief.",
                "Eat small, frequent meals with bland, easily digestible foods: crackers, toast, rice, bananas.",
                "Include electrolyte-rich foods: coconut water, broths, and fruits if tolerated.",
                "Avoid greasy, spicy, or strong-smelling foods that may worsen nausea.",
                "Consider peppermint tea or aromatherapy for additional nausea relief."
            ])

        if any(s in symptoms for s in ['anxiety', 'anxious']):
            recommendations.extend([
                "Include omega-3 fatty acids: fatty fish, walnuts, chia seeds, and flaxseeds for brain health.",
                "Consume magnesium and B-vitamin rich foods: leafy greens, nuts, seeds, and whole grains.",
                "Limit caffeine and sugar which can worsen anxiety symptoms in sensitive individuals.",
                "Include probiotics: yogurt, kefir, fermented foods for gut-brain axis support.",
                "Consider L-theanine sources: green tea for calm, focused energy."
            ])

        # General wellness foods
        recommendations.extend([
            "Include anti-inflammatory foods: berries, fatty fish, turmeric, and colorful vegetables.",
            "Limit processed foods, excessive sugar, and trans fats which can negatively impact energy and mood."
        ])

        return " ".join(recommendations)

    def _generate_lifestyle_recommendations(self, symptoms: List[str]) -> str:
        """Generate comprehensive lifestyle modification recommendations"""

        recommendations = []

        # Sleep foundation
        recommendations.extend([
            "Prioritize 7-9 hours of quality sleep nightly with a consistent sleep schedule, even on weekends.",
            "Create an optimal sleep environment: dark, cool (65-68°F), quiet room with comfortable bedding.",
            "Establish a relaxing bedtime routine: dim lights, avoid screens 1 hour before bed, try reading or gentle stretching."
        ])

        # Stress management
        recommendations.extend([
            "Practice stress reduction techniques: deep breathing exercises, progressive muscle relaxation, or mindfulness meditation for 10-15 minutes daily.",
            "Maintain work-life balance with clear boundaries between work and personal time."
        ])

        # Symptom-specific lifestyle modifications
        if any(s in symptoms for s in ['tired', 'fatigue']):
            recommendations.extend([
                "Optimize your sleep hygiene: avoid caffeine after 2 PM, limit alcohol, and keep a sleep diary to identify patterns.",
                "Consider strategic 20-30 minute power naps before 3 PM if needed, but avoid longer or later naps.",
                "Gradually increase physical activity as tolerated - even 10-15 minutes of walking can boost energy.",
                "Evaluate your workspace ergonomics and take regular breaks from sedentary activities."
            ])

        if any(s in symptoms for s in ['headache']):
            recommendations.extend([
                "Implement the 20-20-20 rule for screen work: every 20 minutes, look at something 20 feet away for 20 seconds.",
                "Ensure proper lighting at your workspace and consider blue light filtering glasses.",
                "Maintain good posture, especially neck and shoulder alignment during desk work.",
                "Practice regular stress management as tension is a common headache trigger."
            ])

        if any(s in symptoms for s in ['anxiety', 'anxious', 'trouble sleeping']):
            recommendations.extend([
                "Establish a daily relaxation practice: try apps like Headspace or Calm for guided meditations.",
                "Limit news consumption and social media exposure, especially before bedtime.",
                "Create a worry journal: write down concerns earlier in the day to prevent bedtime rumination.",
                "Consider professional counseling or therapy for additional anxiety management strategies."
            ])

        if any(s in symptoms for s in ['back pain']):
            recommendations.extend([
                "Evaluate and improve your workspace ergonomics: proper chair height, monitor position, and keyboard placement.",
                "Take movement breaks every 30-60 minutes if you have a desk job.",
                "Focus on core strengthening exercises and proper lifting techniques in daily activities.",
                "Consider your sleep surface - ensure your mattress and pillows provide adequate support."
            ])

        # Activity and movement
        recommendations.extend([
            "Incorporate regular physical activity: aim for 150 minutes of moderate exercise weekly, as appropriate for your condition.",
            "Spend time outdoors daily for natural light exposure and fresh air when possible."
        ])

        # Social and environmental
        recommendations.extend([
            "Maintain social connections and seek support from friends, family, or support groups when dealing with health concerns.",
            "Create a calm, organized living environment that supports relaxation and well-being."
        ])

        return " ".join(recommendations)

    def _generate_medical_guidance(self, symptoms: List[str], intent: str) -> str:
        """Generate when to seek medical care guidance"""

        if intent == "emergency":
            return "🚨 This appears to be a medical emergency. Call 911 or emergency services immediately. Do not drive yourself - call for emergency transportation."

        guidance = []

        # Immediate care scenarios
        if any(s in symptoms for s in ['chest pain', 'difficulty breathing']):
            guidance.append("🚨 Seek immediate emergency medical care for chest pain or difficulty breathing. Call 911 or go to the nearest emergency room.")

        # Urgent care (within 24 hours)
        urgent_symptoms = ['severe pain', 'high fever', 'persistent vomiting']
        if any(urgent in ' '.join(symptoms) for urgent in urgent_symptoms):
            guidance.append("Seek medical care within 24 hours for severe or rapidly worsening symptoms.")

        # Routine medical consultation
        guidance.extend([
            "Schedule a medical consultation if symptoms persist for more than 5-7 days without improvement.",
            "Contact your healthcare provider if symptoms significantly worsen or new concerning symptoms develop.",
            "Seek medical evaluation if symptoms interfere with your daily activities, work, or sleep quality."
        ])

        # Specific symptom guidance
        if any(s in symptoms for s in ['dizzy', 'dizziness']):
            guidance.append("For dizziness: seek care if accompanied by hearing changes, severe headache, chest pain, or if episodes are frequent or severe.")

        if any(s in symptoms for s in ['headache']):
            guidance.append("For headaches: seek immediate care if sudden, severe, or accompanied by fever, stiff neck, vision changes, or confusion.")

        if any(s in symptoms for s in ['tired', 'fatigue']):
            guidance.append("For persistent fatigue: consider medical evaluation to rule out conditions like anemia, thyroid disorders, or sleep disorders.")

        # Professional consultation types
        guidance.extend([
            "Consider consulting specialists as recommended by your primary care provider (cardiologist, neurologist, etc.).",
            "Don't hesitate to seek a second opinion if you have ongoing concerns about your symptoms or treatment plan."
        ])

        # Important disclaimer
        guidance.extend([
            "Remember: This guidance is educational only and should not replace your clinical judgment or professional medical advice.",
            "Always trust your instincts - if something feels seriously wrong, seek medical care promptly regardless of these general guidelines."
        ])

        return " ".join(guidance)

    def _build_medical_prompt(self, nlp_results: Dict, context: Optional[Dict] = None) -> str:
        """Build medical prompt for Mistral"""
        # Implementation for Mistral prompt building
        return f"Medical analysis for: {nlp_results.get('original_text', '')}"

    def _get_system_prompt(self) -> str:
        """Get system prompt for Mistral"""
        return "You are a medical wellness advisor providing evidence-based health recommendations."

    def _parse_mistral_response(self, response_text: str) -> Dict:
        """Parse Mistral response"""
        # Basic parsing - in a real implementation this would be more sophisticated
        return {
            "condition_summary": response_text[:200] + "...",
            "precautions": "Monitor symptoms and consult healthcare provider.",
            "yoga_plan": "Practice gentle yoga as tolerated.",
            "diet_plan": "Follow a balanced, nutritious diet.",
            "lifestyle_tips": "Maintain good sleep and stress management.",
            "medication_guidance": "Consult healthcare provider for medical evaluation.",
            "raw_response": response_text
        }

# Global instance
mistral_engine = MistralRecommendationEngine()
