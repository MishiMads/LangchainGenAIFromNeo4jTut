import pandas as pd
import re
import os

# --- CONFIGURATION ---
INPUT_FILENAME = R'C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\genai-integration-langchain\Scripts\CurriculumTestScriptsAndData\ResultsConsoleLogsFromCurriculumTest\TestOutputGPT4oMiniDeepEval2from11-12-26mTXT.txt'  # Put your log content in this file
OUTPUT_FILENAME = 'Math_RAG_Test_Results.xlsx'


# ---------------------

def parse_log_to_excel(input_file, output_file):
    # 1. Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: The file '{input_file}' was not found.")
        print("Please create a text file with that name and paste your log data into it.")
        return

    # 2. Read the file content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            log_data = f.read()
        print(f"✅ Successfully read {len(log_data)} characters from '{input_file}'.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # 3. Parsing Logic (Regex)
    # Split the log into individual test blocks based on the separator lines
    # We look for the "========== PASSED/FAILED/SKIPPED" lines or the start of a new test block
    test_cases = re.split(
        r'={10,}\s+(?:PASSED|FAILED|SKIPPED).*?\[\s*\d+%\]|(?=PASSED \[\s*\d+%\])|(?=FAILED \[\s*\d+%\])|(?=SKIPPED)',
        log_data)

    # Filter to ensure we only process blocks that actually contain a query
    test_cases = [tc for tc in test_cases if "Testing Query:" in tc]

    data_rows = []

    print(f"🔄 Processing {len(test_cases)} test cases...")

    for i, text_block in enumerate(test_cases, 1):
        row = {"Test ID": i}

        # -- Extract Status --
        # Check for explicit failure markers or Skip warnings
        if "❌ TEST FAILED" in text_block or "FAILED [" in text_block:
            row["Status"] = "FAILED"
        elif "Skipped:" in text_block:
            row["Status"] = "SKIPPED"
        else:
            row["Status"] = "PASSED"

        # -- Extract Query --
        q_match = re.search(r"Testing Query: (.*)", text_block)
        row["Query"] = q_match.group(1).strip() if q_match else "Unknown"

        # -- Extract Actual Output --
        act_match = re.search(r"Actual \(full\): (.*)", text_block)
        row["Actual Output"] = act_match.group(1).strip() if act_match else "N/A"

        # -- Extract Expected Output --
        exp_match = re.search(r"Expected: (.*)", text_block)
        row["Expected Output"] = exp_match.group(1).strip() if exp_match else ""

        # -- Extract Retrieval Context --
        # Captures text between "Retrieved Context:" and "Expected:"
        ctx_match = re.search(r"Retrieved Context: (.*?)Expected:", text_block, re.DOTALL)
        row["Retrieval Context"] = ctx_match.group(1).strip() if ctx_match else ""

        # -- Extract Faithfulness Metrics --
        faith_s = re.search(r"METRIC: FaithfulnessMetric\nScore: ([\d\.]+) \/ 1.00", text_block)
        faith_r = re.search(r"METRIC: FaithfulnessMetric.*?Judge's Reasoning:\n(.*?)\n={10,}", text_block, re.DOTALL)
        row["Faithfulness Score"] = float(faith_s.group(1)) if faith_s else None
        row["Faithfulness Reason"] = faith_r.group(1).strip().replace('\n', ' ') if faith_r else ""

        # -- Extract Pedagogical Metrics --
        ped_s = re.search(r"METRIC: Pedagogical Quality\nScore: ([\d\.]+) \/ 1.00", text_block)
        ped_r = re.search(r"METRIC: Pedagogical Quality.*?Judge's Reasoning:\n(.*?)\n={10,}", text_block, re.DOTALL)
        row["Pedagogical Score"] = float(ped_s.group(1)) if ped_s else None
        row["Pedagogical Reason"] = ped_r.group(1).strip().replace('\n', ' ') if ped_r else ""

        data_rows.append(row)

    # 4. Create DataFrame and Export
    if not data_rows:
        print("⚠️ No data found! Check if the text file contains the log output.")
        return

    df = pd.DataFrame(data_rows)

    # Reorder columns for better readability
    cols = [
        "Test ID", "Status", "Query", "Actual Output", "Expected Output",
        "Faithfulness Score", "Faithfulness Reason",
        "Pedagogical Score", "Pedagogical Reason", "Retrieval Context"
    ]
    # Only use columns that actually exist in the dataframe (in case some regex failed globally)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    try:
        df.to_excel(output_file, index=False)
        print(f"🎉 Success! Excel file created: {output_file}")
    except PermissionError:
        print(f"❌ Error: Could not write to {output_file}. Is the Excel file currently open?")


if __name__ == "__main__":
    parse_log_to_excel(INPUT_FILENAME, OUTPUT_FILENAME)