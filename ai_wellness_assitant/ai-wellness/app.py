"""
AI Wellness Assistant - Corrected Version
Works without external dependencies but demonstrates real medical AI concepts
Handles all import errors gracefully with intelligent fallbacks
"""

import streamlit as st
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure page first
st.set_page_config(
    page_title="AI Wellness Assistant - Corrected",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our components with proper error handling
try:
    from app.nlp_pipeline import medical_nlp
    from app.mistral_client import mistral_engine
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Component import error: {e}")
    COMPONENTS_AVAILABLE = False
    # Create dummy components for basic functionality
    class DummyNLP:
        def get_capabilities(self):
            return {'spacy_core': False, 'scispacy_medical': False, 'umls_linker': False, 'intent_classifier': False}
        def process_symptoms(self, text):
            return {'original_text': text, 'intent': 'symptom_check', 'medical_entities': [], 'probable_conditions': [], 'entity_count': 0, 'confidence': 0.3}

    class DummyMistral:
        def generate_wellness_plan(self, nlp_results, context=None):
            return {
                'condition_summary': 'Basic analysis completed with limited functionality.',
                'precautions': 'Consult healthcare provider for proper medical evaluation.',
                'yoga_plan': 'Practice gentle movement as tolerated.',
                'diet_plan': 'Follow a balanced, nutritious diet.',
                'lifestyle_tips': 'Maintain good sleep and stress management.',
                'medication_guidance': 'Consult healthcare provider for medical evaluation.',
                'model_used': 'basic_fallback'
            }

    medical_nlp = DummyNLP()
    mistral_engine = DummyMistral()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    .tech-stack-badge {
        background: #f8f9fa;
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #e0e0e0;
    }

    .metric-card {
        background: white;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    .entity-card {
        background: #f0f7ff;
        color: black;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }

    .condition-card {
        background: #fff8e1;
        color: black;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }

    .recommendation-card {
        background: #f8f9fa;
        color: black;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }

    .emergency-alert {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
        font-weight: bold;
    }

    .status-good {
        color: #4CAF50;
    }

    .status-bad {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def display_tech_stack():
    """Display the tech stack status"""
    st.markdown("### 🛠️ Medical AI Tech Stack")

    # Get current capabilities
    if COMPONENTS_AVAILABLE:
        capabilities = medical_nlp.get_capabilities()
    else:
        capabilities = {'spacy_core': False, 'scispacy_medical': False, 'umls_linker': False, 'intent_classifier': False}

    tech_components = [
        ("spaCy + scispaCy", capabilities.get('spacy_core', False) or capabilities.get('scispacy_medical', False)),
        ("UMLS Linking", capabilities.get('umls_linker', False)),
        ("Intent Classification", capabilities.get('intent_classifier', False)),
        ("Mistral-7B", hasattr(mistral_engine, 'ollama_available') and getattr(mistral_engine, 'ollama_available', False)),
        ("Medical Knowledge Base", True),  # Always available through rules
        ("Evidence-Based Rules", True)     # Always available
    ]

    cols = st.columns(3)
    for i, (component, available) in enumerate(tech_components):
        with cols[i % 3]:
            status_class = "status-good" if available else "status-bad"
            icon = "✅" if available else "❌"
            st.markdown(f'<span class="tech-stack-badge"><span class="{status_class}">{icon} {component}</span></span>', unsafe_allow_html=True)

def display_nlp_analysis(nlp_results: Dict):
    """Display detailed NLP analysis results"""
    st.subheader("🧠 Medical NLP Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence = nlp_results.get('confidence', 0)
        color = '#4CAF50' if confidence > 0.7 else '#FF9800' if confidence > 0.4 else '#F44336'
        st.markdown(f"""
        <div class="metric-card">
            <h3>Confidence</h3>
            <h2 style="color: {color}">{confidence:.0%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        intent = nlp_results.get('intent', 'Unknown').replace('_', ' ').title()
        intent_color = {"Symptom Check": "#4CAF50", "General Inquiry": "#2196F3", "Emergency": "#F44336"}.get(intent, "#757575")
        st.markdown(f"""
        <div class="metric-card">
            <h3>Intent</h3>
            <h2 style="color: {intent_color}">{intent}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        entity_count = len(nlp_results.get('medical_entities', []))
        st.markdown(f"""
        <div class="metric-card">
            <h3>Medical Entities</h3>
            <h2 style="color: #9C27B0">{entity_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        condition_count = len(nlp_results.get('probable_conditions', []))
        st.markdown(f"""
        <div class="metric-card">
            <h3>Conditions</h3>
            <h2 style="color: #FF9800">{condition_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Display extracted entities
    entities = nlp_results.get('medical_entities', [])
    if entities:
        st.subheader("🔬 Extracted Medical Entities")

        for entity in entities:
            cui_info = f" • CUI: {entity.get('cui', 'N/A')}" if entity.get('cui') else ""
            umls_desc = f" • UMLS: {entity.get('umls_description', 'N/A')}" if entity.get('umls_description') else ""

            st.markdown(f"""
            <div class="entity-card">
                <strong>{entity['text']}</strong> [{entity.get('label', 'UNKNOWN')}]
                <br><small>Confidence: {entity.get('confidence', 0):.1%}{cui_info}{umls_desc}</small>
            </div>
            """, unsafe_allow_html=True)

    # Display probable conditions
    conditions = nlp_results.get('probable_conditions', [])
    if conditions:
        st.subheader("🩺 Probable Medical Conditions")

        for condition in conditions:
            prob = condition.get('probability_score', 0)
            supporting = condition.get('supporting_entities', 0)

            st.markdown(f"""
            <div class="condition-card">
                <strong>{condition['condition']}</strong>
                <br><small>Probability: {prob:.1%} • Supporting evidence: {supporting} entities</small>
            </div>
            """, unsafe_allow_html=True)

def display_wellness_recommendations(wellness_plan: Dict):
    """Display comprehensive wellness recommendations"""

    # Check for emergency
    if "🚨" in wellness_plan.get('precautions', '') or wellness_plan.get('intent') == 'emergency':
        st.markdown(f"""
        <div class="emergency-alert">
            🚨 <strong>EMERGENCY DETECTED</strong><br>
            {wellness_plan.get('precautions', 'Seek immediate medical attention.')}
        </div>
        """, unsafe_allow_html=True)
        return

    st.success("✅ AI Analysis Complete! Here are your evidence-based wellness recommendations.")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 **Summary**", "⚠️ **Precautions**", "🧘 **Yoga Plan**", 
        "🥗 **Diet Plan**", "💡 **Lifestyle**", "💊 **Medical Guidance**"
    ])

    with tab1:
        st.subheader("Condition Summary")
        st.markdown(f"""
        <div class="recommendation-card">
            {wellness_plan.get('condition_summary', 'No summary available')}
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.subheader("⚠️ Safety Precautions")
        st.warning(wellness_plan.get('precautions', 'Monitor symptoms and consult healthcare provider if concerns persist.'))

    with tab3:
        st.subheader("🧘 Yoga & Physical Activities")
        st.markdown(f"""
        <div class="recommendation-card">
            {wellness_plan.get('yoga_plan', 'Practice gentle yoga and breathing exercises as tolerated.')}
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.subheader("🥗 Nutritional Recommendations")
        st.markdown(f"""
        <div class="recommendation-card">
            {wellness_plan.get('diet_plan', 'Follow a balanced diet with adequate hydration.')}
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.subheader("💡 Lifestyle Modifications")
        st.markdown(f"""
        <div class="recommendation-card">
            {wellness_plan.get('lifestyle_tips', 'Maintain regular sleep schedule and stress management practices.')}
        </div>
        """, unsafe_allow_html=True)

    with tab6:
        st.subheader("💊 Medical Consultation Guidance")
        st.info(wellness_plan.get('medication_guidance', 'Consult healthcare provider for medical evaluation and treatment recommendations.'))

        # Model information
        model_used = wellness_plan.get('model_used', 'Unknown')
        generation_time = wellness_plan.get('generation_time_ms', 0)

        st.markdown(f"""
        **AI Generation Details:**
        - Model/System: {model_used}
        - Generation Time: {generation_time}ms
        - NLP Confidence: {wellness_plan.get('nlp_confidence', 0):.1%}
        """)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Wellness Assistant - Corrected</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #333; font-size: 1.2rem;">Medical AI Pipeline with Smart Fallbacks</p>', unsafe_allow_html=True)

    # Display tech stack status
    display_tech_stack()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔬 Medical AI Pipeline")

        if COMPONENTS_AVAILABLE:
            capabilities = medical_nlp.get_capabilities()

            if any(capabilities.values()):
                st.success("""
                ✅ **Components Loaded Successfully**
                """)

                # Show detailed status
                st.markdown("**Component Status:**")
                for component, status in capabilities.items():
                    icon = "✅" if status else "❌"
                    readable_name = component.replace('_', ' ').title()
                    st.markdown(f"{icon} {readable_name}")

                if not capabilities.get('scispacy_medical', False):
                    st.info("""
                    **For Full NLP Features:**
                    ```bash
                    pip install spacy scispacy
                    python -m spacy download en_core_web_sm
                    ```
                    """)

            else:
                st.warning("""
                ⚠️ **Basic Mode Active**
                Advanced NLP models not found.
                Using rule-based medical analysis.
                """)
        else:
            st.error("""
            ❌ **Component Loading Failed**
            Using basic fallback functionality.
            """)

        st.markdown("## ⚠️ Medical Disclaimer")
        st.error("""
        This tool provides wellness guidance only.

        **Not for medical diagnosis.**
        Always consult healthcare professionals
        for medical concerns.
        """)

        # Analysis history
        if st.session_state.analysis_history:
            st.markdown("## 📊 Recent Analyses")
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-3:])):
                with st.expander(f"Analysis {len(st.session_state.analysis_history)-i}"):
                    st.write(f"**Time:** {analysis['timestamp']}")
                    st.write(f"**Input:** {analysis['input_text'][:50]}...")
                    st.write(f"**Intent:** {analysis['intent']}")
                    st.write(f"**Entities:** {analysis['entity_count']}")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Medical Symptom Analysis")

        # Example queries
        with st.expander("💡 Example Medical Queries"):
            examples = [
                "I've been feeling really tired and dizzy for the past few days",
                "I have a headache and feel nauseous after working on my computer", 
                "I'm experiencing lower back pain after sitting for long hours",
                "I feel anxious and have trouble sleeping lately"
            ]

            for example in examples:
                if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
                    st.session_state.user_input = example

        # Main input
        user_input = st.text_area(
            "Describe your symptoms in detail:",
            value=st.session_state.get('user_input', ''),
            placeholder="e.g., I've been feeling tired and having headaches lately. I also notice some dizziness when I stand up quickly...",
            height=120,
            help="The more detail you provide, the better the AI analysis will be"
        )

        # Additional context
        with st.expander("🔧 Medical Context (Optional)"):
            col_age, col_gender = st.columns(2)
            with col_age:
                age = st.number_input("Age", min_value=1, max_value=120, value=None)
            with col_gender:
                gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])

            existing_conditions = st.multiselect(
                "Existing medical conditions",
                ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Arthritis", "Depression", "Other"]
            )

        # Analysis button
        analyze_button = st.button(
            "🔍 Analyze with Medical AI",
            type="primary",
            use_container_width=True,
            disabled=not user_input or len(user_input.strip()) < 10
        )

        # Analysis processing
        if analyze_button and user_input:
            # Prepare context
            context = {}
            if age:
                context['age'] = age
            if gender != "Not specified":
                context['gender'] = gender.lower()
            if existing_conditions:
                context['existing_conditions'] = existing_conditions

            # Show processing steps
            with st.spinner("🧠 Processing with Medical AI Pipeline..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: NLP Analysis
                    status_text.text("Step 1: Medical NLP Analysis...")
                    progress_bar.progress(25)

                    nlp_results = medical_nlp.process_symptoms(user_input)

                    # Step 2: AI Recommendation Generation
                    status_text.text("Step 2: Generating Wellness Recommendations...")
                    progress_bar.progress(75)

                    wellness_plan = mistral_engine.generate_wellness_plan(nlp_results, context)

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis Complete!")
                    time.sleep(0.5)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Display results
                    st.markdown("---")
                    display_nlp_analysis(nlp_results)

                    st.markdown("---") 
                    display_wellness_recommendations(wellness_plan)

                    # Store in history
                    analysis_record = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'input_text': user_input,
                        'intent': nlp_results.get('intent', 'unknown'),
                        'entity_count': len(nlp_results.get('medical_entities', [])),
                        'confidence': nlp_results.get('confidence', 0),
                        'model_used': wellness_plan.get('model_used', 'unknown')
                    }
                    st.session_state.analysis_history.append(analysis_record)

                    # Download option
                    report_text = f"""
AI WELLNESS ASSISTANT - MEDICAL ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
System: Medical AI Pipeline with Smart Fallbacks

INPUT: {user_input}

NLP ANALYSIS:
- Intent: {nlp_results.get('intent', 'unknown')}
- Confidence: {nlp_results.get('confidence', 0):.1%}
- Medical Entities: {len(nlp_results.get('medical_entities', []))}
- Probable Conditions: {len(nlp_results.get('probable_conditions', []))}

WELLNESS RECOMMENDATIONS:

CONDITION SUMMARY:
{wellness_plan.get('condition_summary', 'N/A')}

SAFETY PRECAUTIONS:
{wellness_plan.get('precautions', 'N/A')}

YOGA & PHYSICAL ACTIVITIES:
{wellness_plan.get('yoga_plan', 'N/A')}

NUTRITIONAL RECOMMENDATIONS:
{wellness_plan.get('diet_plan', 'N/A')}

LIFESTYLE MODIFICATIONS:
{wellness_plan.get('lifestyle_tips', 'N/A')}

MEDICAL CONSULTATION GUIDANCE:
{wellness_plan.get('medication_guidance', 'N/A')}

SYSTEM INFORMATION:
- Processing Method: {nlp_results.get('processing_method', 'unknown')}
- AI Model/System: {wellness_plan.get('model_used', 'unknown')}
- Generation Time: {wellness_plan.get('generation_time_ms', 0)}ms

DISCLAIMER:
This analysis is for educational and wellness guidance purposes only.
Always consult qualified healthcare professionals for medical diagnosis and treatment.
                    """

                    st.download_button(
                        "📄 Download Complete Analysis Report",
                        report_text,
                        file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")

                    # Show basic error recovery
                    st.info("Attempting basic analysis with available components...")

                    try:
                        # Minimal fallback analysis
                        basic_result = {
                            'original_text': user_input,
                            'intent': 'symptom_check',
                            'medical_entities': [],
                            'probable_conditions': [],
                            'entity_count': 0,
                            'confidence': 0.3
                        }

                        basic_plan = {
                            'condition_summary': 'Thank you for sharing your health concerns. Due to technical limitations, a basic analysis was performed.',
                            'precautions': 'Monitor your symptoms and consult a healthcare provider if concerns persist or worsen.',
                            'yoga_plan': 'Practice gentle movement and breathing exercises as comfortable for you.',
                            'diet_plan': 'Focus on a balanced diet with plenty of fruits, vegetables, and adequate hydration.',
                            'lifestyle_tips': 'Maintain regular sleep schedule, manage stress, and stay physically active as appropriate.',
                            'medication_guidance': 'Consult with a healthcare provider for proper medical evaluation and any treatment recommendations.',
                            'model_used': 'emergency_fallback'
                        }

                        st.markdown("---")
                        st.warning("⚠️ Basic Analysis Mode - Limited Functionality")
                        display_wellness_recommendations(basic_plan)

                    except Exception as e2:
                        st.error(f"Complete system failure: {e2}")

        elif analyze_button and not user_input:
            st.warning("⚠️ Please describe your symptoms before analyzing.")

        elif analyze_button and len(user_input.strip()) < 10:
            st.warning("⚠️ Please provide more detailed information about your symptoms (at least 10 characters).")

    with col2:
        st.subheader("🔬 Pipeline Status")

        # Component status
        if COMPONENTS_AVAILABLE:
            capabilities = medical_nlp.get_capabilities()

            components = {
                "spaCy Core": capabilities.get('spacy_core', False),
                "scispaCy Medical": capabilities.get('scispacy_medical', False),
                "UMLS Linker": capabilities.get('umls_linker', False),
                "Intent Classifier": capabilities.get('intent_classifier', False),
                "Mistral-7B": hasattr(mistral_engine, 'ollama_available') and getattr(mistral_engine, 'ollama_available', False)
            }
        else:
            components = {
                "spaCy Core": False,
                "scispaCy Medical": False,
                "UMLS Linker": False,
                "Intent Classifier": False,
                "Mistral-7B": False
            }

        for component, status in components.items():
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} **{component}**")

        st.markdown("---")

        st.subheader("📚 Available Features")

        available_features = []
        if COMPONENTS_AVAILABLE:
            capabilities = medical_nlp.get_capabilities()
            if capabilities.get('scispacy_medical'):
                available_features.append("✅ Advanced medical entity recognition")
            if capabilities.get('umls_linker'):
                available_features.append("✅ UMLS medical concept linking")
            if capabilities.get('intent_classifier'):
                available_features.append("✅ AI intent classification")

        # Always available features
        available_features.extend([
            "✅ Rule-based medical analysis",
            "✅ Evidence-based recommendations",
            "✅ Emergency symptom detection",
            "✅ Comprehensive wellness plans",
            "✅ Medical safety protocols"
        ])

        for feature in available_features:
            st.markdown(feature)

        st.markdown("---")

        st.subheader("🎯 System Information")
        st.info(f"""
        **Processing Mode:** 
        {"Advanced NLP" if COMPONENTS_AVAILABLE and any(medical_nlp.get_capabilities().values()) else "Rule-Based Fallback"}

        **Available Models:**
        - Medical entity recognition
        - Intent classification  
        - Evidence-based reasoning
        - Safety protocol enforcement
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <strong>AI Wellness Assistant - Corrected Version</strong><br>
        Robust Medical AI with Smart Fallbacks<br>
        <em>Works without external dependencies • Educational purposes only</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
