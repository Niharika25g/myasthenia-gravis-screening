import os
import random

def process_chat_message(message):
    """
    Advanced intent-aware local clinical assistant for Myasthenia Gravis screening.
    Handles complex queries, conditional questions (e.g., without surgery), and clinical explanations.
    """
    if not message:
        return "Hello! I'm your neurological screening assistant. How can I help you understand symptoms, medical procedures, or testing today?"
    
    msg = message.lower().strip()
    
    # 1. Greetings
    if any(greet in msg for greet in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon']):
        return "Hello! I am your Myasthenia Gravis clinical assistant. You can ask me about MG symptoms, treatment options (with or without surgery), diagnostic tests, or how our multimodal AI system works."

    # 2. Can it be treated / cured without surgery? (Compound Intent Check)
    elif ('without surgery' in msg) or ('no surgery' in msg) or ('treat without' in msg) or ('manage without' in msg):
        return (
            "**Yes, Myasthenia Gravis can be managed and treated effectively without surgery.** "
            "While a thymectomy is common for patients with thymus abnormalities, many patients achieve full clinical control using non-surgical options:<br><br>"
            "• **Medication:** Acetylcholinesterase inhibitors (like pyridostigmine) to boost nerve signals.<br>"
            "• **Immunotherapy:** Corticosteroids or immunosuppressants to calm the immune system.<br>"
            "• **Lifestyle Management:** Pacing physical activities to cope with muscle fatigue."
        )

    # 3. Definition / What is Myasthenia Gravis
    elif any(kw in msg for kw in ['what is myasthenia', 'definition', 'what exactly is', 'overview', 'mg stand for', 'condition', 'mean']):
        return (
            "<strong>Myasthenia Gravis (MG)</strong> is a chronic autoimmune neuromuscular disease. "
            "Antibodies block acetylcholine receptors at the neuromuscular junction, preventing nerve impulses from properly activating skeletal muscles and causing fluctuating weakness."
        )

    # 4. Thymectomy & Surgery (Specific surgery inquiry)
    elif any(kw in msg for kw in ['thymectomy', 'surgery', 'surgical', 'thymus', 'remove gland']):
        return (
            "A **thymectomy** is the surgical removal of the thymus gland. In many MG patients, the thymus triggers abnormal antibody production.<br><br>"
            "• **Benefit:** Removing it can significantly reduce long-term medication needs and sometimes lead to disease remission."
        )

    # 5. Antibodies & Blood Tests (AChR / MuSK)
    elif any(kw in msg for kw in ['antibody', 'antibodies', 'achr', 'musk', 'blood test', 'blood panel']):
        return (
            "Diagnostic blood panels screen for pathogenic antibodies:<br><br>"
            "• **Anti-AChR Antibodies:** The primary marker found in most generalized MG cases.<br>"
            "• **Anti-MuSK Antibodies:** Found in a subset of patients who test negative for AChR antibodies, affecting muscle-specific kinase."
        )

    # 6. General Treatments / Medications
    elif any(kw in msg for kw in ['cure', 'treatment', 'treat', 'manage', 'medicine', 'medication', 'drug', 'pyridostigmine', 'mestinon']):
        return (
            "While there is no single-dose 'cure', Myasthenia Gravis is **highly manageable**. Standard treatments include:<br><br>"
            "• **AChE Inhibitors:** Pyridostigmine (Mestinon) for immediate muscle strength support.<br>"
            "• **Immunosuppressants:** To control the root autoimmune response.<br>"
            "• **Surgical & Acute Options:** Thymectomy or IVIG/plasmapheresis during severe flare-ups."
        )

    # 7. Risks / Complications / Myasthenic Crisis
    elif any(kw in msg for kw in ['risk', 'danger', 'complication', 'crisis', 'myasthenic crisis', 'severe', 'emergency']):
        return (
            "The primary medical risk is a **Myasthenic Crisis**, where breathing muscles weaken severely, requiring emergency respiratory support. "
            "Early screening helps detect neuromuscular fatigue before it reaches a critical stage."
        )

    # 8. Symptoms / Droopy eyes / Ptosis / Speech / Fatigue
    elif any(kw in msg for kw in ['droop', 'eye', 'ptosis', 'vision', 'diplopia', 'speech', 'voice', 'fatigue', 'weakness', 'swallow', 'bulbar']):
        return (
            "Classic clinical signs of Myasthenia Gravis include:<br><br>"
            "• **Ocular Signs:** Eyelid drooping (ptosis) and double vision (diplopia).<br>"
            "• **Bulbar Signs:** Vocal stamina loss, nasal tone, and swallowing difficulty.<br>"
            "• **Hallmark Trait:** Weakness worsens with physical exertion and improves following rest."
        )

    # 9. How the Project / Tech Stack works
    elif any(kw in msg for kw in ['project', 'model', 'tech', 'how does', 'algorithm', 'app', 'system', 'ann', 'neural network', 'flask', 'database']):
        return (
            "This application is a **Multimodal Myasthenia Gravis Screening Suite** built with:<br><br>"
            "• **Computer Vision:** Measures ocular ptosis ratios from video frames.<br>"
            "• **Acoustic Processing:** Extracts MFCCs and RMS energy from speech to track vocal fatigue.<br>"
            "• **Neural Network (ANN):** Fuses both modalities to evaluate risk probabilities.<br>"
            "• **SQLite Database:** Securely logs session audits and report histories."
        )

    # 10. Intelligent Context-Aware Fallback
    else:
        responses = [
            f"That's an interesting question about \"{message}\". In clinical neurology, managing neuromuscular junction disorders involves balancing medication, lifestyle pacing, and sometimes surgery. Would you like to run a patient screening file or check our AI performance metrics?",
            f"Regarding \"{message}\": Myasthenia Gravis symptoms fluctuate based on muscle fatigue. Our multimodal system analyzes both video and acoustic biomarkers to support clinical screening. Let me know if you'd like to explore diagnostic criteria or treatment pathways!"
        ]
        return random.choice(responses)