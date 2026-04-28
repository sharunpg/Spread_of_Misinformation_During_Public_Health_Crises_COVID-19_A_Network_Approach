"""
COVID-19 Misinformation Detector - v4
Dark theme, LLM Intelligence (Ollama), Fact Gathering, Image Support, Network Analysis
"""
import streamlit as st
from datetime import datetime
import json
import os

st.set_page_config(
    page_title="Spread of Misinformation during Public Health Crises COVID-19 : A Network Approach",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1, h2, h3, h4, h5, h6 { color: #e6edf3 !important; font-weight: 500; }
    p, span, label, .stMarkdown { color: #e6edf3 !important; }
    .stTextArea textarea, .stTextInput input {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 6px;
    }
    .stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover { background-color: #2ea043 !important; }
    .mode-badge {
        display: inline-block; padding: 4px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 500; margin-left: 12px;
    }
    .badge-verify { background-color: #238636; color: #fff; }
    .badge-gather { background-color: #1f6feb; color: #fff; }
    .result-card {
        background-color: #161b22; border-radius: 8px;
        padding: 20px; margin: 16px 0; border-left: 4px solid;
    }
    .result-correct { border-left-color: #238636; }
    .result-misinfo { border-left-color: #da3633; }
    .result-unverified { border-left-color: #d29922; }
    .streamlit-expanderHeader { background-color: #161b22 !important; color: #e6edf3 !important; }
    .streamlit-expanderContent { background-color: #0d1117 !important; }
    .stProgress > div > div { background-color: #238636 !important; }
    .footer-text {
        color: #8b949e !important; font-size: 13px;
        padding: 24px 0; border-top: 1px solid #21262d; margin-top: 40px;
    }
    .source-indicator {
        background-color: #161b22; padding: 12px 16px;
        border-radius: 6px; margin: 12px 0; border: 1px solid #30363d;
    }
    .llm-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 11px; margin-left: 8px;
    }
    .llm-on { background-color: #238636; color: #fff; }
    .llm-off { background-color: #6e7681; color: #fff; }
    .network-badge { background-color: #8957e5; color: #fff; }
    .gather-card {
        background-color: #161b22; padding: 16px;
        border-radius: 8px; margin: 8px 0; border: 1px solid #30363d;
    }
    .fact-tag { color: #238636; font-weight: 500; }
    .myth-tag { color: #da3633; font-weight: 500; }
    .input-type-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 11px; background-color: #1f6feb; color: #fff;
    }
    .network-card {
        background-color: #1c1f26; padding: 12px 16px;
        border-radius: 6px; margin: 8px 0; border: 1px solid #8957e5;
    }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

from config import DATA_DIR

# Session state
if 'mode' not in st.session_state:
    st.session_state.mode = "verify"
if 'result' not in st.session_state:
    st.session_state.result = None
if 'gather_report' not in st.session_state:
    st.session_state.gather_report = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None

@st.cache_resource
def load_classifier():
    from classifier import MisinformationClassifier
    return MisinformationClassifier()

@st.cache_resource
def load_input_handler():
    from input_handler import InputHandler
    return InputHandler()

@st.cache_resource
def check_ollama():
    """Check if Ollama is available"""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False

@st.cache_resource
def check_ocr():
    """Check if OCR is available"""
    try:
        import easyocr
        return True
    except:
        return False

def get_network_stats():
    """Get network statistics"""
    try:
        from network_analysis import get_claim_network
        network = get_claim_network()
        return network.get_network_stats()
    except:
        return None

# Header with mode toggle
col_title, col_mode = st.columns([3, 2])

with col_title:
    mode_label = "Verify" if st.session_state.mode == "verify" else "Gather"
    badge_class = "badge-verify" if st.session_state.mode == "verify" else "badge-gather"
    
    ollama_available = check_ollama()
    llm_badge = '<span class="llm-badge llm-on">LLM ON</span>' if ollama_available else '<span class="llm-badge llm-off">LLM OFF</span>'
    
    # Network badge (NEW)
    network_stats = get_network_stats()
    if network_stats:
        network_badge = f'<span class="llm-badge network-badge">Network: {network_stats["total_claims"]} claims</span>'
    else:
        network_badge = '<span class="llm-badge llm-off">Network OFF</span>'
    
    st.markdown(f"""
        <h1 style="margin-bottom: 0; display: inline-flex; align-items: center; flex-wrap: wrap;">
            Spread of Misinformation during Public Health Crises COVID-19 : A Network Approach
            <span class="mode-badge {badge_class}">{mode_label}</span>
            {llm_badge}
            {network_badge}
        </h1>
    """, unsafe_allow_html=True)

with col_mode:
    st.write("")
    mc1, mc2, _ = st.columns([1, 1, 1])
    with mc1:
        if st.button("Verify", use_container_width=True,
                     type="primary" if st.session_state.mode == "verify" else "secondary",
                     key="mode_verify"):
            st.session_state.mode = "verify"
            st.rerun()
    with mc2:
        if st.button("Gather", use_container_width=True,
                     type="primary" if st.session_state.mode == "gather" else "secondary",
                     key="mode_gather"):
            st.session_state.mode = "gather"
            st.rerun()

st.markdown("---")

# ==================== VERIFY MODE ====================
if st.session_state.mode == "verify":
    st.markdown("### Verify a Claim")
    st.markdown("<p style='color: #8b949e;'>Enter text, paste a URL (including Twitter/X), or upload an image</p>", unsafe_allow_html=True)
    
    # Input tabs
    tab_text, tab_image = st.tabs(["📝 Text/URL", "🖼️ Image"])
    
    with tab_text:
        user_input = st.text_area(
            "Input", height=100,
            placeholder="Example: Garlic cures COVID-19\nOr paste a URL: https://twitter.com/...",
            label_visibility="collapsed", key="claim_input"
        )
        
        col1, col2, _ = st.columns([1, 2, 2])
        with col1:
            analyze_btn = st.button("Verify", type="primary", use_container_width=True, key="verify_btn")
        
        if analyze_btn and user_input.strip():
            input_handler = load_input_handler()
            
            with st.spinner("Analyzing..."):
                from input_handler import InputType
                processed = input_handler.process_input(user_input.strip())
                
                if not processed['success']:
                    st.error(f"❌ {processed['error']}")
                    st.session_state.result = None
                    
                    if 'twitter' in user_input.lower() or 'x.com' in user_input.lower():
                        st.info("💡 **Alternatives for Twitter/X posts:**\n"
                               "1. **Copy & Paste**: Copy the tweet text and paste it above\n"
                               "2. **Screenshot**: Take a screenshot and use the Image tab\n"
                               "3. **Mobile**: Use 'Copy text' option from tweet menu")
                else:
                    if processed['input_type'] == InputType.URL:
                        method_info = processed.get('fetch_method', '')
                        if method_info:
                            st.success(f"✓ Successfully fetched content via {method_info}")
                        st.markdown(f"""
                            <div class="source-indicator">
                                <strong>URL Content</strong> | Domain: {processed.get('domain', 'Unknown')}
                                {f" | {processed.get('title', '')}" if processed.get('title') else ""}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📄 Extracted Text", expanded=False):
                            st.text(processed['text'][:500] + "..." if len(processed['text']) > 500 else processed['text'])
                        
                        st.session_state.extracted_text = processed['text']
                    
                    classifier = load_classifier()
                    result = classifier.classify(processed['text'])
                    st.session_state.result = result
    
    with tab_image:
        ocr_available = check_ocr()
        
        if not ocr_available:
            st.warning("⚠️ OCR not available. Install with: `pip install easyocr`")
        
        uploaded_file = st.file_uploader(
            "Upload an image (screenshot of tweet, news article, etc.)",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            key="image_upload"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            col1, _ = st.columns([1, 4])
            with col1:
                analyze_img_btn = st.button("Verify Image", type="primary", use_container_width=True, key="verify_img_btn")
            
            if analyze_img_btn:
                if not ocr_available:
                    st.error("OCR is not available. Please install easyocr.")
                else:
                    with st.spinner("Extracting text from image..."):
                        input_handler = load_input_handler()
                        
                        image_bytes = uploaded_file.getvalue()
                        processed = input_handler.process_image_bytes(image_bytes, uploaded_file.name)
                        
                        if not processed['success']:
                            st.error(f"Error: {processed['error']}")
                        else:
                            st.markdown(f"""
                                <div class="source-indicator">
                                    <span class="input-type-badge">IMAGE</span> | Extracted text from: {uploaded_file.name}
                                </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("📄 Extracted Text", expanded=True):
                                st.text(processed['text'])
                            
                            st.session_state.extracted_text = processed['text']
                            
                            classifier = load_classifier()
                            result = classifier.classify(processed['text'])
                            st.session_state.result = result
    
    # Display results
    if st.session_state.result:
        result = st.session_state.result
        
        if 'MISINFO' in result.label:
            card_class, icon, status = "result-misinfo", "⛔", "Potential Misinformation"
        elif 'CORRECT' in result.label:
            card_class, icon, status = "result-correct", "✓", "Verified Information"
        else:
            card_class, icon, status = "result-unverified", "?", "Unverified"
        
        llm_indicator = ""
        if result.llm_used:
            llm_indicator = '<span class="llm-badge llm-on" style="margin-left: 10px;">AI Verified</span>'
        
        # Network indicator (NEW)
        network_indicator = ""
        if hasattr(result, 'similar_claims_count') and result.similar_claims_count > 0:
            network_indicator = f'<span class="llm-badge network-badge" style="margin-left: 10px;">Network: {result.similar_claims_count} similar</span>'
        
        st.markdown(f"""
            <div class="result-card {card_class}">
                <h3 style="margin: 0 0 8px 0;">{icon} {status} {llm_indicator} {network_indicator}</h3>
                <p style="margin: 0; color: #8b949e;">Confidence: {result.confidence:.0%}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Analysis details
        with st.expander("Analysis Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Claim Type**")
                intent_display = result.claim_intent
                if result.is_negation:
                    intent_display += " (Negation)"
                st.markdown(f"`{intent_display}`")
                
                if result.llm_reasoning:
                    st.markdown("**AI Reasoning**")
                    st.markdown(f"_{result.llm_reasoning}_")
            
            with col2:
                if result.matched_facts:
                    st.markdown("**Related Verified Facts**")
                    for fact in result.matched_facts[:3]:
                        st.markdown(f"• {fact[:80]}...")
        
        # Network Analysis section (NEW)
        if hasattr(result, 'similar_claims_count') and result.similar_claims_count > 0:
            with st.expander("🔗 Network Analysis", expanded=True):
                st.markdown(f"""
                    <div class="network-card">
                        <strong>Network Risk Score:</strong> {result.network_risk_score:.0%}<br>
                        <strong>Similar Claims Found:</strong> {result.similar_claims_count}<br>
                        <strong>Misinformation Neighbors:</strong> {result.misinfo_neighbors}<br>
                        <strong>Verified Neighbors:</strong> {result.correct_neighbors}<br>
                        <strong>Analysis:</strong> {result.network_explanation}
                    </div>
                """, unsafe_allow_html=True)
                
                # Visual risk bar
                st.markdown("**Network Risk Level**")
                st.progress(min(1.0, result.network_risk_score))
                
                if result.network_risk_score >= 0.7:
                    st.warning("⚠️ High network risk - this claim is similar to many known misinformation patterns")
                elif result.network_risk_score >= 0.5:
                    st.info("ℹ️ Moderate network risk - mixed signals from similar claims")
                elif result.correct_neighbors > result.misinfo_neighbors:
                    st.success("✓ Positive network signal - similar to verified claims")
        
        # Technical Details expander
        with st.expander("Technical Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Fact Similarity**")
                st.progress(min(1.0, result.fact_similarity))
                st.caption(f"{result.fact_similarity:.1%}")
            with col2:
                st.markdown("**Misinfo Similarity**")
                st.progress(min(1.0, result.misinfo_similarity))
                st.caption(f"{result.misinfo_similarity:.1%}")
            
            st.markdown(f"**Explanation:** {result.explanation}")
            
            if st.session_state.extracted_text:
                st.markdown("**Analyzed Text:**")
                st.text(st.session_state.extracted_text[:500] + "..." if len(st.session_state.extracted_text) > 500 else st.session_state.extracted_text)
        
        if 'MISINFO' in result.label:
            st.markdown("""
                <div style="
                    margin-top: 20px;
                    padding: 10px 16px;
                    background-color: rgba(218, 54, 51, 0.1);
                    border-left: 3px solid #da3633;
                    border-radius: 4px;
                    font-size: 13px;
                    color: #f85149;
                ">
                    ⚠️ This content is identified as potential misinformation and should not be published or shared without verification from official health sources.
                </div>
            """, unsafe_allow_html=True)

# ==================== GATHER MODE ====================
else:
    st.markdown("### Gather Facts from Official Sources")
    st.markdown("<p style='color: #8b949e;'>Fetch verified facts from WHO/CDC</p>", unsafe_allow_html=True)
    
    col1, col2, _ = st.columns([1, 1, 2])
    
    with col1:
        gather_btn = st.button("Fetch Facts", type="primary", use_container_width=True, key="gather_btn")
    with col2:
        auto_approve = st.checkbox("Auto-add to KB", value=True, key="auto_approve")
    
    if gather_btn:
        with st.spinner("Gathering facts from WHO/CDC..."):
            from fact_gatherer import run_fact_gathering
            from knowledge_base import refresh_knowledge_base
            
            result = run_fact_gathering(auto_approve=auto_approve)
            st.session_state.gather_report = result
            
            if auto_approve:
                refresh_knowledge_base()
                st.success("Facts added to knowledge base!")
    
    # Show staged claims if not auto-approved
    staged_path = os.path.join(DATA_DIR, "staged_claims.json")
    if os.path.exists(staged_path) and not auto_approve:
        with open(staged_path, 'r') as f:
            staged = json.load(f)
        
        if staged.get('status') == 'pending_review':
            st.markdown("---")
            st.markdown("### Review Staged Claims")
            
            facts = [c for c in staged['claims'] if c['claim_type'] == 'fact']
            myths = [c for c in staged['claims'] if c['claim_type'] == 'myth']
            
            tab1, tab2 = st.tabs([f"Facts ({len(facts)})", f"Myths ({len(myths)})"])
            
            with tab1:
                for claim in facts[:15]:
                    st.markdown(f"""
                        <div class="gather-card">
                            <span class="fact-tag">FACT</span> | {claim['source']}<br>
                            {claim['text'][:150]}
                        </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                for claim in myths[:15]:
                    st.markdown(f"""
                        <div class="gather-card">
                            <span class="myth-tag">MYTH</span> | {claim['source']}<br>
                            {claim['text'][:150]}
                        </div>
                    """, unsafe_allow_html=True)
            
            if st.button("Approve & Add to KB", type="primary", key="approve_btn"):
                from fact_gatherer import approve_staged_claims
                from knowledge_base import refresh_knowledge_base
                
                approve_staged_claims()
                refresh_knowledge_base()
                st.success("Added to knowledge base!")
                st.rerun()
    
    # Report
    if st.session_state.gather_report:
        report = st.session_state.gather_report
        st.markdown("---")
        st.success(f"Gathered {report['gathering_report']['total_claims']} claims from official sources")

# ==================== FOOTER ====================
st.markdown("---")

# Status indicators
ollama_status = "✓ Local LLM Active" if check_ollama() else "○ Local LLM Offline"
ocr_status = "✓ OCR Available" if check_ocr() else "○ OCR Unavailable"
network_stats = get_network_stats()
network_status = f"✓ Network: {network_stats['total_claims']} claims" if network_stats else "○ Network Empty"

st.markdown(f"<p style='color: #6e7681; font-size: 12px;'>{ollama_status} | {ocr_status} | {network_status} | Sources: WHO, CDC, NHS</p>", 
            unsafe_allow_html=True)

st.markdown("""
    <div class="footer-text">
        <strong>How it works:</strong> Claims are checked against verified facts using semantic similarity, 
        local AI reasoning (Ollama), and network analysis (MinHash/LSH for near-duplicate detection). 
        Supports text input, URLs (including Twitter/X), and image uploads with OCR. No data is sent to external servers.
    </div>
""", unsafe_allow_html=True)