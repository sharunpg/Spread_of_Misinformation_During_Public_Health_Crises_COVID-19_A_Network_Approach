"""
Demonstration Script

Shows all system features:
1. Basic classification
2. Intent detection and protection
3. Multilingual support
4. Testing mode with feedback
5. Knowledge ingestion pipeline

Run: python demo.py
"""
import sys

def demo_basic_classification():
    """Demo 1: Basic classification"""
    print("\n" + "="*60)
    print("DEMO 1: BASIC CLASSIFICATION")
    print("="*60)
    
    from classifier import MisinformationClassifier
    
    classifier = MisinformationClassifier()
    
    test_claims = [
        "COVID-19 vaccines are safe and effective",
        "5G networks spread coronavirus",
        "Drinking hot water cures COVID",
        "Wearing masks helps prevent transmission",
    ]
    
    for claim in test_claims:
        print(f"\n📝 Claim: {claim}")
        result = classifier.classify(claim)
        print(f"   Label: {result.label}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Fact Score: {result.fact_similarity:.3f}")
        print(f"   Misinfo Score: {result.misinfo_similarity:.3f}")

def demo_intent_protection():
    """Demo 2: Intent detection and protection"""
    print("\n" + "="*60)
    print("DEMO 2: INTENT PROTECTION")
    print("="*60)
    
    from intent_detector import IntentDetector
    from classifier import MisinformationClassifier
    
    detector = IntentDetector()
    classifier = MisinformationClassifier()
    
    # Show the turmeric example
    claims = [
        ("Turmeric reduces inflammation", "Symptom relief - TRUE"),
        ("Turmeric cures COVID", "Cure claim - FALSE"),
    ]
    
    print("\n🎯 Intent Detection Demo:")
    for claim, description in claims:
        intent = detector.detect_intent(claim)
        print(f"\n   Claim: '{claim}'")
        print(f"   Expected: {description}")
        print(f"   Detected Intent: {intent.level.name}")
        print(f"   Patterns: {intent.matched_patterns[:2]}")
    
    print("\n🛡️ Intent Protection Demo:")
    print("\nClassifying 'Turmeric cures COVID':")
    result = classifier.classify("Turmeric cures COVID")
    print(f"   Label: {result.label}")
    print(f"   Claim Intent: {result.claim_intent}")
    print(f"   Intent Mismatch: {result.intent_mismatch}")
    if result.intent_warning:
        print(f"   Warning: {result.intent_warning[:80]}...")

def demo_intent_hierarchy():
    """Demo 3: Show intent hierarchy"""
    print("\n" + "="*60)
    print("DEMO 3: INTENT HIERARCHY")
    print("="*60)
    
    from intent_detector import IntentDetector, IntentLevel
    from config import INTENT_HIERARCHY
    
    print("\nIntent Levels (lowest to highest):")
    for level in IntentLevel:
        print(f"   {level.value}. {level.name}")
    
    print("\nValidation Rules:")
    print("   Evidence can only validate claims at SAME or LOWER levels")
    print("\n   SYMPTOM_MANAGEMENT evidence CAN validate:")
    for allowed in INTENT_HIERARCHY[IntentLevel.SYMPTOM_MANAGEMENT]:
        print(f"      ✓ {allowed.name}")
    print("\n   SYMPTOM_MANAGEMENT evidence CANNOT validate:")
    for level in [IntentLevel.PREVENTION, IntentLevel.TREATMENT, IntentLevel.CURE]:
        print(f"      ✗ {level.name}")

def demo_multilingual():
    """Demo 4: Multilingual support"""
    print("\n" + "="*60)
    print("DEMO 4: MULTILINGUAL SUPPORT")
    print("="*60)
    
    from classifier import MisinformationClassifier
    
    classifier = MisinformationClassifier()
    
    multilingual_claims = [
        ("Las vacunas COVID son seguras", "Spanish"),
        ("Le vaccin contient des micropuces", "French"),
        ("5G verbreitet Corona", "German"),
    ]
    
    for claim, lang in multilingual_claims:
        print(f"\n📝 [{lang}] {claim}")
        result = classifier.classify(claim)
        print(f"   Detected Language: {result.detected_language}")
        print(f"   Translated: {result.translated_text[:50]}...")
        print(f"   Label: {result.label}")

def demo_tester_feedback():
    """Demo 5: Testing mode with feedback"""
    print("\n" + "="*60)
    print("DEMO 5: TESTER FEEDBACK SYSTEM")
    print("="*60)
    
    from tester_feedback import get_feedback_manager, FeedbackAction
    from classifier import MisinformationClassifier
    
    print("\n🔐 Simulating tester login...")
    manager = get_feedback_manager()
    
    # Authenticate
    success = manager.authenticate("tester1", "covid_test_2024")
    print(f"   Login successful: {success}")
    
    if success:
        classifier = MisinformationClassifier()
        
        # Classify a claim
        claim = "Honey helps soothe cough symptoms"
        result = classifier.classify(claim)
        
        print(f"\n📝 Test Claim: {claim}")
        print(f"   Predicted: {result.label} ({result.confidence:.0%})")
        
        # Simulate approval
        print("\n📋 Simulating tester approval...")
        response = manager.process_feedback(
            claim_text=claim,
            cleaned_text=result.cleaned_text,
            predicted_label=result.label,
            predicted_confidence=result.confidence,
            feedback=FeedbackAction.APPROVED,
            tester_note="Verified: honey is known to soothe coughs"
        )
        
        print(f"   Result: {response['message']}")
        print(f"   Was stored: {response['was_stored']}")
        if response['was_stored']:
            print(f"   Storage location: {response['storage_location']}")
        
        # Show session stats
        stats = manager.get_session_stats()
        print(f"\n📊 Session Stats:")
        print(f"   Claims reviewed: {stats['claims_reviewed']}")
        print(f"   Claims approved: {stats['claims_approved']}")
        
        # Logout
        manager.logout()
        print("\n✓ Logged out")

def demo_knowledge_ingestion():
    """Demo 6: Knowledge ingestion pipeline"""
    print("\n" + "="*60)
    print("DEMO 6: KNOWLEDGE INGESTION PIPELINE")
    print("="*60)
    
    from knowledge_ingestion import KnowledgeIngestionPipeline
    
    pipeline = KnowledgeIngestionPipeline()
    
    print("\n📁 Creating source directory structure...")
    print(f"   Sources directory: {pipeline.SOURCES_DIR}")
    
    print("\n📄 Supported file formats:")
    print("   - PDF documents")
    print("   - HTML pages (saved locally)")
    print("   - CSV files with claims")
    print("   - Plain text files")
    
    print("\n🔍 Claim extraction features:")
    print("   - COVID relevance filtering")
    print("   - Sentence splitting")
    print("   - Category detection (vaccine, transmission, symptoms, etc.)")
    print("   - Intent level detection")
    
    print("\n⚠️ Conflict detection:")
    print("   - Checks new claims against existing KB")
    print("   - Flags potential contradictions")
    print("   - Requires human review before adding")
    
    print("\n✅ Safety features:")
    print("   - Offline only (no live web scraping)")
    print("   - Source attribution required")
    print("   - Full audit trail")
    print("   - Dry-run mode for testing")

def demo_all():
    """Run all demos"""
    print("\n" + "#"*60)
    print("#" + " "*20 + "FULL SYSTEM DEMO" + " "*20 + "#")
    print("#"*60)
    
    demos = [
        ("Basic Classification", demo_basic_classification),
        ("Intent Protection", demo_intent_protection),
        ("Intent Hierarchy", demo_intent_hierarchy),
        ("Multilingual Support", demo_multilingual),
        ("Tester Feedback", demo_tester_feedback),
        ("Knowledge Ingestion", demo_knowledge_ingestion),
    ]
    
    for name, func in demos:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo run the web interface: streamlit run app.py")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        demos = {
            "basic": demo_basic_classification,
            "intent": demo_intent_protection,
            "hierarchy": demo_intent_hierarchy,
            "multilingual": demo_multilingual,
            "feedback": demo_tester_feedback,
            "ingestion": demo_knowledge_ingestion,
            "all": demo_all,
        }
        if demo_name in demos:
            demos[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available: {', '.join(demos.keys())}")
    else:
        demo_all()