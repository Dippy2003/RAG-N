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

    # ── Drug interactions (extended) ───────────────────────────────────────
    {
        "guideline_id": "DI-006",
        "category": "drug_interaction",
        "title": "Metformin + Iodinated Contrast Media: Lactic Acidosis Risk",
        "content": (
            "Metformin must be withheld before and 48 hours after iodinated contrast "
            "administration in patients with eGFR <60 mL/min/1.73m². Contrast-induced "
            "nephropathy can cause acute metformin accumulation leading to fatal lactic acidosis. "
            "Recommendation: Hold metformin the day of the procedure. Resume only after "
            "renal function confirmed normal at 48h. Ensure adequate IV hydration peri-procedure. "
            "Applies to CT scans, coronary angiography, and any contrast-enhanced imaging."
        ),
        "severity": "critical",
        "tags": "metformin contrast CT scan lactic acidosis nephropathy renal eGFR diabetes imaging",
    },
    {
        "guideline_id": "DI-007",
        "category": "drug_interaction",
        "title": "Digoxin + Amiodarone / Verapamil / Clarithromycin: Toxicity",
        "content": (
            "Digoxin has a narrow therapeutic index (target 0.5–2.0 ng/mL). "
            "Amiodarone, verapamil, diltiazem, and clarithromycin all increase digoxin "
            "plasma levels by reducing renal clearance and inhibiting P-glycoprotein. "
            "Toxicity: bradycardia, heart block, nausea, xanthopsia (yellow vision), arrhythmia. "
            "Recommendation: Reduce digoxin dose by 50% when starting amiodarone. "
            "Monitor digoxin levels and ECG within 1 week of any co-prescription. "
            "Avoid clarithromycin — use azithromycin as antibiotic alternative."
        ),
        "severity": "critical",
        "tags": "digoxin amiodarone verapamil clarithromycin toxicity bradycardia arrhythmia P-glycoprotein",
    },
    {
        "guideline_id": "DI-008",
        "category": "drug_interaction",
        "title": "Simvastatin / Atorvastatin + Macrolide Antibiotics: Rhabdomyolysis",
        "content": (
            "Clarithromycin, erythromycin, and azithromycin (to a lesser extent) inhibit "
            "CYP3A4, the primary metabolic pathway for simvastatin and atorvastatin. "
            "This raises statin plasma levels dramatically, risking myopathy and rhabdomyolysis "
            "(muscle breakdown → acute kidney injury, hyperkalaemia). "
            "Recommendation: Temporarily suspend simvastatin/atorvastatin during macrolide course. "
            "Use azithromycin instead of clarithromycin where possible. "
            "Rosuvastatin and pravastatin are safer alternatives (not CYP3A4-dependent)."
        ),
        "severity": "high",
        "tags": "simvastatin atorvastatin clarithromycin macrolide CYP3A4 rhabdomyolysis myopathy statin",
    },
    {
        "guideline_id": "DI-009",
        "category": "drug_interaction",
        "title": "Clopidogrel + Omeprazole/PPIs: Reduced Antiplatelet Effect",
        "content": (
            "Clopidogrel is a prodrug activated by CYP2C19. Omeprazole and esomeprazole "
            "strongly inhibit CYP2C19, reducing clopidogrel's active metabolite by ~45%, "
            "significantly blunting its antiplatelet effect and increasing stent thrombosis risk. "
            "Pantoprazole and rabeprazole have minimal CYP2C19 inhibition and are preferred. "
            "Recommendation: Switch to pantoprazole if PPI is necessary alongside clopidogrel. "
            "Routine PPI co-prescription is not recommended unless GI risk factors present."
        ),
        "severity": "high",
        "tags": "clopidogrel omeprazole PPI CYP2C19 antiplatelet stent thrombosis cardiac pantoprazole",
    },
    {
        "guideline_id": "DI-010",
        "category": "drug_interaction",
        "title": "Opioids + Benzodiazepines: Fatal Respiratory Depression",
        "content": (
            "Concurrent use of opioids (morphine, tramadol, codeine, fentanyl) and "
            "benzodiazepines (diazepam, clonazepam, alprazolam) or Z-drugs (zolpidem) "
            "dramatically increases risk of fatal respiratory depression. "
            "CNS depression is synergistic — each drug alone may be sub-lethal but together "
            "can cause apnoea. Risk is highest at night. "
            "Recommendation: Avoid combination unless absolutely necessary. "
            "If co-prescribed, use lowest effective doses and counsel patient/carer about "
            "apnoea risk. Provide naloxone prescription if opioid dose is high."
        ),
        "severity": "critical",
        "tags": "opioid morphine tramadol benzodiazepine diazepam respiratory depression apnoea naloxone",
    },
    {
        "guideline_id": "DI-011",
        "category": "drug_interaction",
        "title": "Ciprofloxacin / Fluoroquinolones + Antacids / Iron / Calcium: Absorption Block",
        "content": (
            "Divalent and trivalent cations (Mg²⁺, Al³⁺, Ca²⁺, Fe²⁺/Fe³⁺, Zn²⁺) chelate "
            "fluoroquinolones in the GI tract, reducing ciprofloxacin absorption by up to 90%. "
            "Common sources: antacids (Mg/Al hydroxide), calcium supplements, iron supplements, "
            "dairy products (calcium), multivitamins with zinc. "
            "Recommendation: Administer ciprofloxacin 2 hours before or 6 hours after "
            "any antacid, iron, calcium, or zinc supplement. "
            "This applies to all fluoroquinolones: levofloxacin, moxifloxacin, norfloxacin."
        ),
        "severity": "medium",
        "tags": "ciprofloxacin fluoroquinolone antacid iron calcium absorption chelation levofloxacin",
    },
    {
        "guideline_id": "DI-012",
        "category": "drug_interaction",
        "title": "Insulin / Metformin + Alcohol: Severe Hypoglycaemia",
        "content": (
            "Alcohol inhibits hepatic gluconeogenesis and potentiates the glucose-lowering "
            "effect of insulin and sulphonylureas (glibenclamide, glipizide, glimepiride). "
            "Risk of prolonged severe hypoglycaemia, particularly after fasting or heavy drinking. "
            "Metformin + alcohol also increases lactic acidosis risk. "
            "Recommendation: Counsel all diabetic patients on alcohol avoidance, especially "
            "on insulin or sulphonylureas. Advise never to skip meals when drinking. "
            "Educate carers on hypoglycaemia recognition and glucagon use."
        ),
        "severity": "high",
        "tags": "insulin metformin alcohol hypoglycaemia sulphonylurea glibenclamide lactic acidosis diabetes",
    },
    {
        "guideline_id": "DI-013",
        "category": "drug_interaction",
        "title": "Amoxicillin / Antibiotics + Methotrexate: Methotrexate Toxicity",
        "content": (
            "Penicillin-class antibiotics including amoxicillin reduce renal tubular secretion "
            "of methotrexate, increasing plasma levels and toxicity risk. "
            "Effects include severe mucositis, pancytopenia, and hepatotoxicity. "
            "Trimethoprim is also contraindicated with methotrexate (additive antifolate effect). "
            "Recommendation: Avoid amoxicillin, co-amoxiclav, and trimethoprim in patients "
            "on methotrexate. Use macrolides (azithromycin) or doxycycline as alternatives. "
            "If combination unavoidable, suspend methotrexate during antibiotic course."
        ),
        "severity": "critical",
        "tags": "amoxicillin methotrexate toxicity penicillin trimethoprim antifolate mucositis pancytopenia",
    },

    # ── Allergy protocols (extended) ───────────────────────────────────────
    {
        "guideline_id": "AL-005",
        "category": "allergy_mismatch",
        "title": "Latex Allergy: Cross-Reactivity with Foods (Latex-Fruit Syndrome)",
        "content": (
            "Latex allergy cross-reacts with kiwi, banana, avocado, chestnut, and papaya "
            "due to shared hevein-like proteins. Known as latex-fruit syndrome. "
            "Affects ~30–50% of latex-allergic patients. Risk in surgical and procedural settings. "
            "Recommendation: Flag latex allergy prominently before any surgical procedure. "
            "Use latex-free gloves, catheters, and equipment. "
            "Warn patient to avoid kiwi, banana, avocado if they develop oral allergy symptoms. "
            "Skin prick test before elective surgery in sensitised patients."
        ),
        "severity": "high",
        "tags": "latex allergy kiwi banana avocado cross-reactivity surgical gloves latex-fruit syndrome",
    },
    {
        "guideline_id": "AL-006",
        "category": "allergy_mismatch",
        "title": "Aspirin Allergy: Safety of Paracetamol (Acetaminophen)",
        "content": (
            "Patients with aspirin/NSAID hypersensitivity (AERD or urticarial type) can "
            "safely use paracetamol (acetaminophen) in standard doses (≤2g/day) as it "
            "minimally inhibits COX-1 at therapeutic doses. "
            "However, very high paracetamol doses (>2g) may trigger mild reactions in "
            "highly sensitive patients. Selective COX-2 inhibitors (celecoxib) are also "
            "generally safe but introduce cardiovascular risks. "
            "Recommendation: Paracetamol is first-line analgesia for aspirin-allergic patients. "
            "Avoid all NSAIDs, ibuprofen, naproxen, diclofenac, and ketorolac."
        ),
        "severity": "medium",
        "tags": "aspirin allergy paracetamol acetaminophen NSAID safe COX-2 celecoxib AERD",
    },
    {
        "guideline_id": "AL-007",
        "category": "allergy_mismatch",
        "title": "Egg Allergy: Propofol and Vaccine Safety",
        "content": (
            "Propofol is formulated in a lipid emulsion containing soybean oil and egg lecithin. "
            "However, egg lecithin (from egg yolk) rarely causes allergy even in egg-allergic "
            "patients — most egg allergy is to egg white proteins (ovalbumin). "
            "Influenza vaccines grown on eggs: most are safe even in egg-allergic patients "
            "with mild allergy; only those with anaphylaxis to egg require monitored setting. "
            "Recommendation: Egg allergy alone is NOT a contraindication to propofol or "
            "influenza vaccine. Document specific egg reaction type. "
            "For history of egg anaphylaxis: administer vaccine in hospital with resus available."
        ),
        "severity": "medium",
        "tags": "egg allergy propofol vaccine anaesthesia ovalbumin lecithin influenza anaphylaxis",
    },
    {
        "guideline_id": "AL-008",
        "category": "allergy_mismatch",
        "title": "Shellfish/Seafood Allergy: NOT a Contraindication to Iodinated Contrast",
        "content": (
            "The historical belief that shellfish allergy predicts iodinated contrast "
            "media reactions is a MYTH. Shellfish allergy is caused by proteins (tropomyosin), "
            "not iodine. Iodine itself is not allergenic. "
            "True risk factors for contrast reactions are: prior contrast reaction, asthma, "
            "atopy, anxiety, renal impairment, and cardiac disease. "
            "Recommendation: Do NOT withhold contrast or pre-medicate based solely on "
            "shellfish/seafood allergy. Assess for actual risk factors instead. "
            "Document to prevent unnecessary procedure delays."
        ),
        "severity": "low",
        "tags": "shellfish allergy iodine contrast CT scan myth seafood tropomyosin radiology",
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
    {
        "guideline_id": "LK-003",
        "category": "drug_interaction",
        "title": "Dengue Fever: NSAIDs and Aspirin Contraindicated",
        "content": (
            "In confirmed or suspected dengue fever, NSAIDs (ibuprofen, diclofenac, naproxen) "
            "and aspirin are STRICTLY CONTRAINDICATED. These drugs inhibit platelet function "
            "and increase bleeding risk, potentially precipitating dengue haemorrhagic fever (DHF) "
            "or dengue shock syndrome (DSS). Dengue is endemic in Sri Lanka with seasonal peaks. "
            "Recommendation: Use only paracetamol (≤4g/day in adults, ≤15mg/kg/dose in children) "
            "for fever management in dengue. Immediately stop any NSAID if dengue suspected. "
            "Monitor platelet count and haematocrit. Hospital admission if platelets <100,000/µL."
        ),
        "severity": "critical",
        "tags": "dengue fever NSAID aspirin ibuprofen contraindicated bleeding haemorrhagic platelet Sri Lanka",
    },
    {
        "guideline_id": "LK-004",
        "category": "data_integrity",
        "title": "Sri Lanka Diabetes Management: Metformin First-Line Protocol",
        "content": (
            "Per Sri Lanka Endocrine Society guidelines, metformin is first-line therapy "
            "for type 2 diabetes unless eGFR <30 mL/min (contraindicated) or eGFR 30–45 "
            "(use with caution, reduce dose). "
            "HbA1c targets: <7% for most patients; <8% for elderly or those with comorbidities. "
            "Metformin dose: start 500mg once daily with meals, titrate to 2000mg/day. "
            "Common Sri Lankan combinations: metformin + glibenclamide (risk of hypoglycaemia "
            "in elderly — prefer glimepiride). Insulin initiation threshold: HbA1c >10% "
            "or symptomatic hyperglycaemia."
        ),
        "severity": "medium",
        "tags": "diabetes metformin HbA1c insulin glibenclamide Sri Lanka endocrine first-line eGFR",
    },
    {
        "guideline_id": "LK-005",
        "category": "drug_interaction",
        "title": "Malaria Prophylaxis: Chloroquine + Antacids / Primaquine + G6PD",
        "content": (
            "Chloroquine absorption is reduced by 50% when taken with antacids (Mg/Al hydroxide). "
            "Space doses by at least 4 hours. "
            "Primaquine is CONTRAINDICATED in G6PD-deficient patients — causes severe haemolytic "
            "anaemia. G6PD deficiency is prevalent in Sri Lankan Tamil and Moor communities. "
            "Recommendation: Screen for G6PD before prescribing primaquine or dapsone. "
            "Use alternative regimens (chloroquine only) for G6PD-deficient patients with "
            "Plasmodium vivax malaria and arrange supervised radical cure when safe."
        ),
        "severity": "high",
        "tags": "malaria chloroquine primaquine G6PD deficiency haemolysis antacid Sri Lanka vivax",
    },
    {
        "guideline_id": "LK-006",
        "category": "data_integrity",
        "title": "Sri Lanka MoH Essential Medicines: Availability and Substitution",
        "content": (
            "The Sri Lanka National Medicines Regulatory Authority (NMRA) essential medicines "
            "list includes: paracetamol, amoxicillin, metformin, atenolol, amlodipine, "
            "enalapril, furosemide, omeprazole, cetirizine, prednisolone, salbutamol, "
            "digoxin, warfarin, metronidazole, co-trimoxazole, iron/folate, ORS. "
            "Branded equivalents may be prescribed when generics unavailable. "
            "Recommendation: Document if a prescribed drug is unavailable locally. "
            "Use nearest equivalent from NMRA list. Contact regional pharmacy for "
            "special access medicines not on essential list."
        ),
        "severity": "low",
        "tags": "essential medicines NMRA Sri Lanka MoH formulary availability generic substitution",
    },
    {
        "guideline_id": "LK-007",
        "category": "drug_interaction",
        "title": "Extended Ayurvedic / Sinhala Traditional Medicine Interactions",
        "content": (
            "Additional traditional medicine interactions relevant to Sri Lanka: "
            "Kohomba (Neem/Azadirachta indica) — hypoglycaemic effect, potentiates insulin/OHAs. "
            "Weniwalgeta (Coscinium fenestratum) — hepatotoxic in high doses, interacts with statins. "
            "Iramusu (Hemidesmus indicus) — antiplatelet effect, avoid with warfarin/aspirin. "
            "Pathpadagam (Oldenlandia corymbosa) — diuretic, may potentiate furosemide/ACE inhibitors. "
            "Recommendation: Always ask explicitly about Ayurveda, Siddha, Unani, and home remedies. "
            "Assume interaction risk with any anticoagulant, antidiabetic, or hepatically metabolised drug."
        ),
        "severity": "medium",
        "tags": "Ayurvedic Sinhala neem kohomba weniwalgeta iramusu traditional medicine warfarin insulin Sri Lanka",
    },

    # ── Pregnancy drug safety ──────────────────────────────────────────────
    {
        "guideline_id": "PG-001",
        "category": "drug_interaction",
        "title": "Pregnancy: Category D/X Drugs — Absolute Contraindications",
        "content": (
            "The following drugs are CONTRAINDICATED in pregnancy (Category D/X): "
            "Warfarin (teratogenic, fetal haemorrhage — use LMWH heparin instead); "
            "Methotrexate (teratogen, abortifacient — stop ≥3 months before conception); "
            "ACE inhibitors/ARBs in 2nd/3rd trimester (fetal renal agenesis, oligohydramnios); "
            "NSAIDs in 3rd trimester (premature ductus arteriosus closure); "
            "Valproate (neural tube defects, cognitive impairment); "
            "Isotretinoin (severe fetal malformations). "
            "Recommendation: Perform pregnancy test before prescribing Category D/X drugs "
            "to women of reproductive age. Counsel on contraception requirements."
        ),
        "severity": "critical",
        "tags": "pregnancy warfarin methotrexate ACE inhibitor NSAIDs valproate isotretinoin teratogen Category D X contraindicated",
    },
    {
        "guideline_id": "PG-002",
        "category": "drug_interaction",
        "title": "Pregnancy: Safe Drug Options for Common Conditions",
        "content": (
            "Safe drugs in pregnancy for common conditions: "
            "Pain/fever: Paracetamol (all trimesters); avoid NSAIDs (especially 3rd trimester). "
            "Antibiotic: Amoxicillin, erythromycin, azithromycin, cefalexin (safe); "
            "avoid tetracycline, fluoroquinolones, co-trimoxazole (1st trimester). "
            "Hypertension: Methyldopa, labetalol, nifedipine (safe); avoid ACE/ARBs. "
            "Diabetes: Insulin is preferred; metformin used in 2nd/3rd trimester under supervision. "
            "Anticoagulation: LMWH (enoxaparin, dalteparin) throughout; not warfarin. "
            "Nausea: Promethazine, metoclopramide, ondansetron (2nd/3rd trimester)."
        ),
        "severity": "high",
        "tags": "pregnancy safe drugs paracetamol amoxicillin labetalol methyldopa insulin LMWH enoxaparin antibiotic",
    },

    # ── Paediatric dosing ──────────────────────────────────────────────────
    {
        "guideline_id": "PD-001",
        "category": "data_integrity",
        "title": "Paediatric Dosing: Weight-Based Calculation Rules",
        "content": (
            "Paediatric drug doses are weight-based. Key rules: "
            "Paracetamol: 15 mg/kg/dose every 4–6h (max 60 mg/kg/day, max 4g/day in children >50kg). "
            "Amoxicillin: 25–50 mg/kg/day in 3 divided doses (max 3g/day). "
            "Ibuprofen: 5–10 mg/kg/dose every 6–8h (max 40 mg/kg/day; NOT for infants <3 months). "
            "Metformin: ≥10 years only, start 500mg once daily. "
            "Codeine: CONTRAINDICATED under 12 years (risk of fatal respiratory depression "
            "in ultra-rapid CYP2D6 metabolisers). "
            "Recommendation: Always verify dose against current weight. "
            "Never use adult doses for children. Use mg/kg calculation for all drugs <50kg."
        ),
        "severity": "critical",
        "tags": "paediatric dosing weight-based kg paracetamol amoxicillin ibuprofen codeine children metformin",
    },
    {
        "guideline_id": "PD-002",
        "category": "data_integrity",
        "title": "Paediatric Drug Contraindications: Age-Specific Restrictions",
        "content": (
            "Age-specific drug restrictions in children: "
            "Aspirin: CONTRAINDICATED under 16 years (Reye's syndrome risk). "
            "Codeine/tramadol: CONTRAINDICATED under 12 years; avoid under 18 after tonsillectomy. "
            "Tetracyclines/doxycycline: Avoid under 8 years (dental staining, bone effects). "
            "Fluoroquinolones: Generally avoid under 18 (cartilage damage in animal studies). "
            "Metoclopramide: Avoid under 1 year; use with caution under 20kg (dystonic reactions). "
            "Chloramphenicol: Avoid neonates (grey baby syndrome — fatal cardiovascular collapse). "
            "Recommendation: Always check age suitability before prescribing. "
            "If in doubt, consult BNF for Children or paediatric pharmacist."
        ),
        "severity": "critical",
        "tags": "paediatric aspirin Reye codeine tetracycline fluoroquinolone metoclopramide chloramphenicol age restriction child",
    },

    # ── Renal & hepatic dosing ─────────────────────────────────────────────
    {
        "guideline_id": "RN-001",
        "category": "data_integrity",
        "title": "Renal Dosing Table: Common Drugs Requiring eGFR-Based Adjustment",
        "content": (
            "eGFR-based dose adjustments for common drugs: "
            "Metformin: Caution eGFR 30–45; STOP eGFR <30. "
            "Methotrexate: Reduce dose eGFR <60; AVOID eGFR <30. "
            "Digoxin: Reduce dose eGFR <50; close monitoring eGFR <30. "
            "Co-trimoxazole: Halve dose eGFR 15–30; AVOID eGFR <15. "
            "Amoxicillin/clavulanate: Reduce frequency eGFR <30. "
            "NSAIDs: AVOID in all CKD (worsen renal function, risk of AKI). "
            "ACE inhibitors: Monitor K⁺ and creatinine closely; start low dose in CKD. "
            "Recommendation: Calculate eGFR (CKD-EPI formula) for all patients "
            "before prescribing renally-cleared drugs."
        ),
        "severity": "high",
        "tags": "renal dosing eGFR CKD metformin digoxin co-trimoxazole NSAIDs ACE inhibitor adjustment kidney",
    },
    {
        "guideline_id": "HP-001",
        "category": "data_integrity",
        "title": "Hepatic Impairment: Drugs to Avoid in Liver Disease",
        "content": (
            "Drugs with significant hepatic metabolism require dose reduction or avoidance "
            "in liver disease (Child-Pugh B/C): "
            "Statins: Contraindicated in active liver disease (hepatotoxicity risk). "
            "Methotrexate: Contraindicated in significant hepatic impairment. "
            "Paracetamol: Reduce to max 2g/day in liver disease (hepatotoxicity at lower doses). "
            "Warfarin: More sensitive — lower doses required, monitor INR more frequently. "
            "Opioids: Accumulate in liver failure — reduce dose and frequency significantly. "
            "Recommendation: Assess liver function (LFTs, INR, albumin) before prescribing "
            "hepatically-metabolised drugs. Alcohol history is critical context."
        ),
        "severity": "high",
        "tags": "hepatic liver impairment statins methotrexate paracetamol warfarin opioids Child-Pugh LFT",
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
