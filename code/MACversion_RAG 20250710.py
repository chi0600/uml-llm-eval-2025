import sys
import re
import numpy as np
import json
import subprocess
import tempfile
import os
import shutil
import platform
import logging
from difflib import SequenceMatcher
from typing import Tuple, Optional

# ----------------------------
# Platform-aware, hardcoded paths (macOS ARM vs Intel)
# ----------------------------
arch = platform.machine()
brew_prefix = '/opt/homebrew' if arch == 'arm64' else '/usr/local'

# Preferred Homebrew-installed Java and PlantUML JAR
preferred_java = os.path.join(brew_prefix, 'opt', 'openjdk@17', 'bin', 'java')
preferred_jar  = os.path.join(brew_prefix, 'opt', 'plantuml', 'libexec', 'plantuml.jar')

# Fallback to system java if Homebrew path not present
system_java = shutil.which('java')
if os.path.isfile(preferred_java):
    JAVA_PATH = preferred_java
elif system_java:
    JAVA_PATH = system_java
    logging.warning(f"Homebrew Java not found, falling back to system java: {JAVA_PATH}")
else:
    JAVA_PATH = preferred_java
PLANTUML_JAR_PATH = preferred_jar

OUTPUT_DIR = os.path.join(os.getcwd(), 'results')
REF_JSON = os.path.join('data', 'In-real-verifiaction-dataset-20250423.json')
GEN_JSON = os.path.join('data', 'all_few_shot_vector_and_llm_in_real_3example_20250619_raw.json')

# ----------------------------
# Setup logging
# ----------------------------
def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )

# ----------------------------
# Path validators
# ----------------------------
def validate_paths(java_path: str, plantuml_jar_path: str) -> None:
    if not os.path.isfile(java_path):
        raise FileNotFoundError(f"Java executable not found at: {java_path}")
    if not os.path.isfile(plantuml_jar_path):
        raise FileNotFoundError(f"PlantUML JAR not found at: {plantuml_jar_path}")

# ----------------------------
# PlantUML validation (verbose)
# ----------------------------
def validate_plantuml_diagram(
    plantuml_code: str,
    java_path: str,
    plantuml_jar_path: str,
    output_dir: str,
    timeout: int = 10
) -> Tuple[bool, Optional[str], str]:
    validate_paths(java_path, plantuml_jar_path)
    # write to temp file
    with tempfile.NamedTemporaryFile(suffix='.puml', mode='w', encoding='utf-8', delete=False) as tmp:
        tmp.write(plantuml_code)
        input_path = tmp.name
    try:
        os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(
            [java_path, '-jar', plantuml_jar_path, '-verbose', '-o', output_dir, input_path],
            capture_output=True, text=True, timeout=timeout
        )
        stdout = result.stdout or ''
        stderr = result.stderr or ''
        if result.returncode != 0:
            return False, None, f"PlantUML returned code {result.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        if stderr.strip():
            return False, None, f"PlantUML diagnostics:\n{stderr.strip()}"
        base = os.path.splitext(os.path.basename(input_path))[0]
        png_path = os.path.join(output_dir, f"{base}.png")
        if os.path.exists(png_path):
            return True, png_path, "Valid diagram generated"
        return False, None, "No output generated"
    except subprocess.TimeoutExpired:
        return False, None, "PlantUML processing timed out"
    except Exception as e:
        return False, None, f"Error during processing: {e}"
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass

# ----------------------------
# Load spaCy model
# ----------------------------
try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        import spacy.cli
        spacy.cli.download('en_core_web_lg')
        nlp = spacy.load('en_core_web_lg')
except ModuleNotFoundError:
    print("spaCy not installed. Please 'pip install spacy' and download 'en_core_web_lg'.")
    sys.exit(1)

# ----------------------------
# Comparison helpers
# ----------------------------
def extract_participants(diagram: str):
    pattern = re.compile(r'(actor|boundary|control|entity|database|collections|queue|participant)\s+"([^"]+)"\s+as\s+(\w+)', re.IGNORECASE)
    parts = {}
    for line in diagram.splitlines():
        m = pattern.search(line)
        if m:
            parts[m.group(3)] = m.group(2).strip()
    return parts

def compute_similarity(a: str, b: str) -> float:
    return nlp(a.lower()).similarity(nlp(b.lower()))

def create_dynamic_mapping(ref_diagram, gen_diagram, threshold=0.75):
    ref_participants = extract_participants(ref_diagram)
    gen_participants = extract_participants(gen_diagram)
    mapping = {}
    mapping_details = {}
    for gen_alias, gen_display in gen_participants.items():
        best_score = 0
        best_match = None
        details = []
        for ref_alias, ref_display in ref_participants.items():
            sim = compute_similarity(gen_display, ref_display)
            details.append((ref_alias, ref_display, sim))
            if sim > best_score:
                best_score = sim
                best_match = ref_alias
        mapping_details[gen_alias] = {
            'gen_display': gen_display,
            'best_match': best_match,
            'best_score': best_score,
            'comparisons': details
        }
        if best_score >= threshold:
            mapping[gen_alias] = best_match
        else:
            mapping[gen_alias] = gen_alias
    return mapping, mapping_details

def normalize_participants_dynamic(diagram, mapping):
    for old_name, new_name in mapping.items():
        diagram = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, diagram)
    return diagram

def parse_diagram(diagram):
    title_pattern = re.compile(r'^\s*title\s+(.*)$', re.IGNORECASE)
    message_pattern = re.compile(r'(\w+)\s*->\s*(-?>?)\s*(\w+)\s*:\s*(.+)')
    keyword_pattern = re.compile(
        r'^\s*(alt|else|opt|loop|par|break|critical|group|end|activate|deactivate|destroy)\b(?:\s+(.*))?', 
        re.IGNORECASE
    )
    note_start_pattern = re.compile(
        r'^\s*note\s+(left|right|top|bottom|over)(?:\s+of\s+(\w+))?(?::\s*(.*))?',
        re.IGNORECASE
    )
    note_end_pattern = re.compile(r'^\s*end\s+note\s*$', re.IGNORECASE)

    title = None
    messages = []
    keywords = []
    notes = []

    in_note = False
    current_note_position = None
    current_note_participant = None
    note_lines = []

    for line in diagram.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if title is None:
            title_match = title_pattern.search(line_stripped)
            if title_match:
                title = title_match.group(1).strip()

        if in_note:
            if note_end_pattern.search(line_stripped):
                full_note_text = "\n".join(note_lines)
                notes.append((current_note_position, current_note_participant, full_note_text))
                in_note = False
                current_note_position = None
                current_note_participant = None
                note_lines = []
            else:
                note_lines.append(line_stripped)
            continue

        note_start_match = note_start_pattern.search(line_stripped)
        if note_start_match:
            current_note_position = note_start_match.group(1).lower()
            current_note_participant = note_start_match.group(2)
            single_line_text = note_start_match.group(3)
            if single_line_text:
                notes.append((current_note_position, current_note_participant, single_line_text.strip()))
            else:
                in_note = True
            continue

        msg_match = message_pattern.search(line_stripped)
        if msg_match:
            sender, arrow, receiver, text = msg_match.groups()
            messages.append((sender, receiver, text.strip()))
            continue

        key_match = keyword_pattern.match(line_stripped)
        if key_match:
            keyword = key_match.group(1).lower()
            context = key_match.group(2) if key_match.group(2) else ""
            keywords.append((keyword, context.strip()))
            continue

    if in_note and note_lines:
        full_note_text = "\n".join(note_lines)
        notes.append((current_note_position, current_note_participant, full_note_text))

    return title, messages, keywords, notes

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def semantic_similarity(text1, text2):
    doc1 = nlp(preprocess_text(text1))
    doc2 = nlp(preprocess_text(text2))
    return doc1.similarity(doc2)

def structural_similarity(reference_messages, reference_keywords, test_messages, test_keywords):
    lifeline_commands = {"activate", "deactivate", "destroy"}
    ref_lifelines = []
    ref_non_lifelines = []
    for kw, ctx in reference_keywords:
        if kw in lifeline_commands:
            ref_lifelines.append(f"{kw}:{ctx}" if ctx else kw)
        else:
            ref_non_lifelines.append(f"{kw}:{ctx}" if ctx else kw)
    test_lifelines = []
    test_non_lifelines = []
    for kw, ctx in test_keywords:
        if kw in lifeline_commands:
            test_lifelines.append(f"{kw}:{ctx}" if ctx else kw)
        else:
            test_non_lifelines.append(f"{kw}:{ctx}" if ctx else kw)
    ref_non_lifelines_str = "|".join(ref_non_lifelines)
    test_non_lifelines_str = "|".join(test_non_lifelines)
    non_lifeline_matcher = SequenceMatcher(None, ref_non_lifelines_str, test_non_lifelines_str)
    non_lifeline_score = non_lifeline_matcher.ratio()
    ref_lifelines_str = "|".join(ref_lifelines)
    test_lifelines_str = "|".join(test_lifelines)
    lifeline_matcher = SequenceMatcher(None, ref_lifelines_str, test_lifelines_str)
    lifeline_score = lifeline_matcher.ratio()
    ref_str = "|".join([f"{msg[0]}->{msg[1]}" for msg in reference_messages])
    test_str = "|".join([f"{msg[0]}->{msg[1]}" for msg in test_messages])
    message_matcher = SequenceMatcher(None, ref_str, test_str)
    message_score = message_matcher.ratio()
    return non_lifeline_score, lifeline_score, message_score

def compare_titles(ref_title, gen_title):
    if not ref_title and not gen_title:
        return 1.0
    elif not ref_title or not gen_title:
        return 0.0
    else:
        return semantic_similarity(ref_title, gen_title)

def compare_notes(ref_notes, gen_notes):
    if not ref_notes and not gen_notes:
        return 1.0
    elif not ref_notes and gen_notes:
        return 0.5
    elif ref_notes and not gen_notes:
        return 0.0
    else:
        similarities = []
        for rnote in ref_notes:
            best_sim = 0
            for gnote in gen_notes:
                sim = semantic_similarity(rnote[2], gnote[2])
                if sim > best_sim:
                    best_sim = sim
            similarities.append(best_sim)
        return np.mean(similarities) if similarities else 0

def compare_diagrams(reference_diagram, generated_diagram):
    dynamic_mapping, mapping_details = create_dynamic_mapping(reference_diagram, generated_diagram, threshold=0.75)
    generated_diagram_norm = normalize_participants_dynamic(generated_diagram, dynamic_mapping)
    reference_diagram_norm = reference_diagram  # assume reference is normalized
    ref_title, ref_messages, ref_keywords, ref_notes = parse_diagram(reference_diagram_norm)
    gen_title, gen_messages, gen_keywords, gen_notes = parse_diagram(generated_diagram_norm)
    title_score = compare_titles(ref_title, gen_title)
    note_score = compare_notes(ref_notes, gen_notes)
    semantic_scores = []
    for ref_msg in ref_messages:
        best_score = 0
        for test_msg in gen_messages:
            score = semantic_similarity(ref_msg[2], test_msg[2])
            if score > best_score:
                best_score = score
        semantic_scores.append(best_score)
    semantic_avg = np.mean(semantic_scores) if semantic_scores else 0
    non_lifeline_score, lifeline_score, message_score = structural_similarity(
        ref_messages, ref_keywords,
        gen_messages, gen_keywords
    )
    combined_similarity = (0.5 * semantic_avg +
                           0.1 * non_lifeline_score +
                           0.1 * lifeline_score +
                           0.2 * message_score +
                           0.05 * title_score +
                           0.05 * note_score)
    # return everything
    return {
        "semantic_avg":       semantic_avg,
        "non_lifeline_score": non_lifeline_score,
        "lifeline_score":     lifeline_score,
        "message_score":      message_score,
        "title_score":        title_score,
        "note_score":         note_score,
        "combined_similarity": combined_similarity
    }

def sanitize_filename(filename):
    """Replace illegal filename characters with an underscore."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# ----------------------------
# Main
# ----------------------------
def main():
    setup_logging()
    logging.debug(f"Architecture: {arch}")
    logging.debug(f"JAVA_PATH: {JAVA_PATH}")
    logging.debug(f"JAR: {PLANTUML_JAR_PATH}")
    logging.debug(f"Output: {OUTPUT_DIR}")

    validate_paths(JAVA_PATH, PLANTUML_JAR_PATH)
    # load JSONs
    with open(REF_JSON, 'r', encoding='utf-8') as f:
        refs = json.load(f)
    ref_dict = {e['file_name']: e['output'] for e in refs}
    with open(GEN_JSON, 'r', encoding='utf-8') as f:
        gens = json.load(f)

    # build gen_dict
    gen_dict = {}
    for ent in gens:
        fn = ent.get('file_name')
        if not fn or fn not in ref_dict: continue
        for mdl, modes in ent.get('modes', {}).items():
            for md, info in modes.items():
                for alt in info.get('alternative', []):
                    u = alt.get('uml','').strip()
                    if u:
                        gen_dict.setdefault(fn, {}).setdefault(mdl, {}).setdefault(md, []).append(u)

    results = []
    for fn, models in gen_dict.items():
        ref_uml = ref_dict[fn]
        for mdl, modes in models.items():
            for md, ulist in modes.items():
                scores = []
                for idx, uml in enumerate(ulist,1):
                    valid, png, msg = validate_plantuml_diagram(uml, JAVA_PATH, PLANTUML_JAR_PATH, OUTPUT_DIR)
                    if valid and png:
                        tgt = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fn)[0]}_{mdl}_{md}_{idx}.png")
                        shutil.move(png, tgt)
                        logging.info(f"Generated: {tgt}")
                    else:
                        logging.error(f"Syntax error in {fn}: {msg}")
                    det = compare_diagrams(ref_uml, uml) if valid else {k:0.0 for k in ['semantic_avg','non_lifeline_score','lifeline_score','message_score','title_score','note_score','combined_similarity']}
                    scores.append(det['combined_similarity'])
                results.append({
                    'file_name': fn,
                    'model': mdl,
                    'mode': md,
                    'avg_score': float(np.mean(scores)) if scores else 0.0,
                    'individual_scores': scores
                })

    outp = os.path.join('data','evaluation_scores_all_few_shot_vector_and_llm_in_real_3example_20250619_raw.json')
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp,'w',encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Wrote {len(results)} entries to {outp}")

if __name__ == '__main__':
    main()
