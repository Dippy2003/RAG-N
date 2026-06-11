"""
Medical knowledge base for RAG retrieval.

Contains curated clinical guidelines covering drug interactions, allergy protocols,
and data integrity rules. Run this module directly to seed the Supabase
`medical_guidelines` table with embeddings.

Usage:
    uv run python knowledge_base.py
"""

import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Curated medical guidelines
# ---------------------------------------------------------------------------

GUIDELINES: list[dict] = [
    # ── Drug interactions ──────────────────────────────────────────────────
    {
        "guideline_id": "DI-001",
        "category": "drug_interaction",
        "title": "Warfarin + Aspirin: Major Bleeding Risk",
        "content": (
            "Concurrent use of warfarin and aspirin is a MAJOR interaction. "
            "Aspirin inhibits platelet aggregation and can displace warfarin from "
            "plasma proteins, potentiating anticoagulation. Risk of serious bleeding "
            "(GI haemorrhage, intracranial) increases 3-fold. "
            "Recommendation: Contraindicated unless under cardiologist supervision with "
            "frequent INR monitoring. If co-prescription is unavoidable, use lowest "
            "effective aspirin dose (≤100 mg/day) with gastroprotection."
        ),
        "severity": "critical",
        "tags": "warfarin aspirin bleeding anticoagulant NSAID INR haemorrhage",
    },
    {
        "guideline_id": "DI-002",
        "category": "drug_interaction",
        "title": "Warfarin + Ibuprofen: Anticoagulation Potentiation",
        "content": (
            "Ibuprofen (and all NSAIDs) significantly potentiate warfarin anticoagulation "
            "by inhibiting platelet function and displacing warfarin from protein binding sites. "
            "May also cause GI mucosal damage, increasing bleeding site risk. "
            "Recommendation: Avoid combination. Use paracetamol as analgesia for patients "
            "on warfarin. If NSAID unavoidable, reduce warfarin dose and monitor INR closely."
        ),
        "severity": "critical",
        "tags": "warfarin ibuprofen NSAID anticoagulant bleeding INR",
    },
    {
        "guideline_id": "DI-003",
        "category": "drug_interaction",
        "title": "Methotrexate + Aspirin: Methotrexate Toxicity",
        "content": (
            "NSAIDs including aspirin reduce renal clearance of methotrexate, raising plasma "
            "levels to potentially toxic concentrations. Risk of methotrexate toxicity includes "
            "bone marrow suppression, hepatotoxicity, and mucositis. "
            "Recommendation: Avoid concurrent use, especially at methotrexate doses >15 mg/week. "
            "If unavoidable, monitor FBC, LFTs, and methotrexate levels closely."
        ),
        "severity": "critical",
        "tags": "methotrexate aspirin NSAID toxicity bone marrow renal clearance",
    },
    {
        "guideline_id": "DI-004",
        "category": "drug_interaction",
        "title": "ACE Inhibitor + Potassium-Sparing Diuretic: Hyperkalaemia",
        "content": (
            "ACE inhibitors (e.g. enalapril, lisinopril) combined with potassium-sparing "
            "diuretics (e.g. spironolactone, amiloride) can cause dangerous hyperkalaemia. "
            "Potassium levels >6.0 mmol/L risk fatal arrhythmia. "
            "Recommendation: Monitor serum potassium within 1 week of starting combination "
            "and regularly thereafter. Avoid in patients with renal impairment."
        ),
        "severity": "high",
        "tags": "ACE inhibitor spironolactone potassium hyperkalaemia arrhythmia renal",
    },
    {
        "guideline_id": "DI-005",
        "category": "drug_interaction",
        "title": "Fluoxetine + Tramadol: Serotonin Syndrome Risk",
        "content": (
            "Combining SSRIs (fluoxetine, sertraline, paroxetine) with tramadol risks "
            "serotonin syndrome: hyperthermia, agitation, tremor, myoclonus, and in severe "
            "cases cardiovascular collapse. Tramadol also inhibits serotonin reuptake. "
            "Recommendation: Use alternative analgesia. If combination unavoidable, use "
            "lowest tramadol dose and counsel patient to report agitation, tremor, or fever."
        ),
        "severity": "high",
        "tags": "fluoxetine tramadol SSRI serotonin syndrome agitation hyperthermia",
    },

    # ── Allergy protocols ──────────────────────────────────────────────────
    {
        "guideline_id": "AL-001",
        "category": "allergy_mismatch",
        "title": "Penicillin Allergy: Cross-Reactivity with Cephalosporins and Carbapenems",
        "content": (
            "Patients with documented penicillin allergy have 1–2% cross-reactivity risk "
            "with cephalosporins (higher with early-generation, e.g. cephalexin) and "
            "<1% with carbapenems. Anaphylaxis risk is highest in patients with a history "
            "of anaphylaxis to penicillin. "
            "Recommendation: Avoid all penicillin-class drugs (amoxicillin, ampicillin, "
            "flucloxacillin, co-amoxiclav). Use azithromycin, clarithromycin, or doxycycline "
            "as alternatives for respiratory infections. If cephalosporin essential, do "
            "supervised skin test first."
        ),
        "severity": "critical",
        "tags": "penicillin allergy amoxicillin ampicillin cephalosporin cross-reactivity anaphylaxis",
    },
    {
        "guideline_id": "AL-002",
        "category": "allergy_mismatch",
        "title": "Sulfonamide Allergy: Cross-Reactivity with Thiazides and COX-2 Inhibitors",
        "content": (
            "Sulfonamide antibiotics (co-trimoxazole, sulfamethoxazole) allergy does not "
            "reliably predict cross-reactivity with non-antibiotic sulfonamides such as "
            "thiazide diuretics, furosemide, or celecoxib. However, in patients with "
            "history of severe sulfonamide reactions (SJS, TEN), caution is warranted. "
            "Recommendation: Document exact reaction type. Mild rash does not preclude "
            "non-antibiotic sulfonamides. Anaphylaxis or SJS warrants avoidance of all."
        ),
        "severity": "medium",
        "tags": "sulfonamide allergy trimethoprim thiazide furosemide SJS TEN cross-reactivity",
    },
    {
        "guideline_id": "AL-003",
        "category": "allergy_mismatch",
        "title": "Aspirin/NSAID Hypersensitivity: COX-1 Inhibition Reaction",
        "content": (
            "Aspirin-exacerbated respiratory disease (AERD) affects ~10% of adult asthmatics. "
            "NSAIDs that inhibit COX-1 (aspirin, ibuprofen, naproxen, diclofenac) trigger "
            "bronchoconstriction and nasal polyps in susceptible patients. "
            "Recommendation: All COX-1 inhibitors are contraindicated once NSAID hypersensitivity "
            "confirmed. Use paracetamol or selective COX-2 inhibitors (celecoxib) cautiously. "
            "Avoid in patients with nasal polyps + asthma even without confirmed allergy."
        ),
        "severity": "high",
        "tags": "aspirin NSAID allergy AERD asthma COX-1 ibuprofen naproxen respiratory",
    },
    {
        "guideline_id": "AL-004",
        "category": "allergy_mismatch",
        "title": "Contrast Media Allergy: Pre-medication Protocol",
        "content": (
            "Prior moderate/severe reaction to iodinated contrast requires pre-medication "
            "before repeat exposure. Protocol: oral prednisolone 50 mg at 13h, 7h, and 1h "
            "prior, plus cetirizine 10 mg 1h prior. Switch to low-osmolar non-ionic contrast. "
            "Recommendation: Flag allergy in radiology request. Ensure resuscitation equipment "
            "available. Mild prior reactions (urticaria only) do not mandate pre-medication "
            "but should be noted."
        ),
        "severity": "high",
        "tags": "contrast media allergy iodine radiology premedication prednisolone antihistamine",
    },

    # ── Data integrity rules ───────────────────────────────────────────────
    {
        "guideline_id": "DI-INT-001",
        "category": "data_integrity",
        "title": "Blood Type Discrepancy: Transfusion Safety Protocol",
        "content": (
            "Conflicting ABO or Rh blood type records are a critical patient safety risk. "
            "ABO-incompatible transfusion causes acute haemolytic reaction with mortality "
            "up to 10%. A single discrepancy between clinic, lab, and pharmacy records "
            "mandates re-verification before any transfusion or surgical procedure. "
            "Recommendation: Request fresh sample for re-grouping at the trusted lab. "
            "Freeze all cross-match and blood product orders until resolved. "
            "Document in patient's permanent record with date of resolution."
        ),
        "severity": "critical",
        "tags": "blood type ABO Rh transfusion haemolytic reaction grouping cross-match",
    },
    {
        "guideline_id": "DI-INT-002",
        "category": "data_integrity",
        "title": "Date of Birth Mismatch: Identity Verification Risk",
        "content": (
            "Discrepant date of birth across records indicates a possible patient identity "
            "mix-up or data entry error. This can lead to wrong-patient medication dispensing, "
            "incorrect age-based dosing, and missed age-specific screening. "
            "Recommendation: Verify identity using NIC (National Identity Card) as primary "
            "source in Sri Lanka. Cross-reference with biometric data if available. "
            "Resolve before prescribing any weight-based or age-dosed medications."
        ),
        "severity": "high",
        "tags": "date of birth identity mismatch NIC patient safety dosing age",
    },
    {
        "guideline_id": "DI-INT-003",
        "category": "data_integrity",
        "title": "Medication List Discrepancy: Reconciliation on Admission",
        "content": (
            "Discrepancies between medication lists from different sources (clinic, pharmacy, "
            "hospital) are one of the leading causes of preventable medication errors. "
            "The Sri Lanka College of Internal Medicine recommends structured medication "
            "reconciliation on every admission and discharge. "
            "Recommendation: Use the pharmacy dispensing record as ground truth for "
            "what the patient is actually taking. Verify with patient and/or carer. "
            "Resolve duplications, dose conflicts, and omissions before discharge summary."
        ),
        "severity": "high",
        "tags": "medication reconciliation admission discharge pharmacy dispensing error",
    },
    {
        "guideline_id": "DI-INT-004",
        "category": "data_integrity",
        "title": "Allergy Documentation Standards: Sri Lanka MoH Guidelines",
        "content": (
            "The Sri Lanka Ministry of Health mandates that allergy status be documented "
            "in all patient records using the NKDA (No Known Drug Allergies) flag or "
            "specific allergen + reaction type. Allergy should be propagated to all "
            "sub-systems: clinic, lab request forms, and pharmacy dispensing system. "
            "Recommendation: Missing allergy in one source should not be treated as "
            "'no allergy' — treat as unknown and verify with patient before prescribing "
            "high-risk agents."
        ),
        "severity": "medium",
        "tags": "allergy documentation NKDA MoH Sri Lanka standards propagation",
    },
    {
        "guideline_id": "DI-INT-005",
        "category": "data_integrity",
        "title": "Renal Function and Drug Dosing: eGFR-Based Adjustment",
        "content": (
            "Drugs with significant renal clearance (metformin, methotrexate, NSAIDs, "
            "aminoglycosides, vancomycin) require dose adjustment based on eGFR. "
            "Conflicting creatinine or eGFR values between sources invalidates dosing "
            "calculations and risks accumulation toxicity. "
            "Recommendation: Use most recent lab eGFR as authoritative. Re-dose or hold "
            "renally-cleared drugs if eGFR < 30 mL/min/1.73m² until confirmed."
        ),
        "severity": "high",
        "tags": "renal function eGFR creatinine dosing metformin NSAIDs vancomycin toxicity",
    },

    # ── Sri Lanka context ──────────────────────────────────────────────────
    {
        "guideline_id": "LK-001",
        "category": "data_integrity",
        "title": "Sri Lanka NIC-Based Identity Deduplication",
        "content": (
            "The Sri Lanka National Identity Card (NIC) is the primary patient identifier "
            "across MOH facilities. Old NIC format: 9 digits + V/X. New format: 12 digits. "
            "The same person may have both formats in different records. "
            "Recommendation: Normalise both NIC formats before matching. "
            "Old NIC 123456789V = New NIC 19XXXXXXXXXXV where XX encodes birth year. "
            "A mismatch in name or DOB with matching NIC should be treated as data entry "
            "error, not a different patient."
        ),
        "severity": "medium",
        "tags": "NIC Sri Lanka identity deduplication national identity card format normalisation",
    },
    {
        "guideline_id": "LK-002",
        "category": "drug_interaction",
        "title": "Herbal / Ayurvedic Interactions with Anticoagulants",
        "content": (
            "Sri Lankan patients frequently use Ayurvedic and herbal preparations alongside "
            "conventional medications. Notable interactions: Ginkgo biloba + warfarin "
            "(increased bleeding), garlic supplements + warfarin (potentiated anticoagulation), "
            "St John's Wort + warfarin/SSRIs (reduced efficacy via CYP3A4 induction). "
            "Recommendation: Always ask about traditional medicine use. Document in allergy/"
            "medications section. Assume interaction risk with any anticoagulant or narrow "
            "therapeutic index drug."
        ),
        "severity": "medium",
        "tags": "herbal Ayurvedic warfarin interaction ginkgo garlic St John Wort CYP3A4 Sri Lanka",
    },
]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_knowledge_base() -> None:
    """Embed all guidelines and upsert into Supabase medical_guidelines table."""
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    texts = [f"{g['title']}. {g['content']} {g['tags']}" for g in GUIDELINES]
    print(f"Embedding {len(texts)} guidelines...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for guideline, vector in zip(GUIDELINES, embeddings):
        supabase.table("medical_guidelines").upsert({
            "guideline_id": guideline["guideline_id"],
            "category":     guideline["category"],
            "title":        guideline["title"],
            "content":      guideline["content"],
            "severity":     guideline["severity"],
            "tags":         guideline["tags"],
            "embedding":    vector.tolist(),
        }, on_conflict="guideline_id").execute()
        print(f"  ✓ [{guideline['guideline_id']}] {guideline['title']}")

    print(f"\nDone. {len(GUIDELINES)} guidelines in Supabase.")


if __name__ == "__main__":
    seed_knowledge_base()
