import pandas as pd
import re
import os
import json

# --- CONFIGURATION ---
# Use raw string (r'...') for Windows paths to handle backslashes correctly
INPUT_FILENAME = r"C:\Users\jakob\Desktop\Git Repos\LangchainGenAIFromNeo4jTut\Jakob's_Stuff\User_Model_Txt_Files\UserModelTestOutput 5.txt"
OUTPUT_FILENAME = 'RAG_Results_5.xlsx'


# ---------------------

def parse_log_to_excel(input_file, output_file):
    # 1. Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: The file '{input_file}' was not found.")
        print("Please check the path or create the file.")
        return

    # 2. Read the file content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            log_data = f.read()
        print(f"✅ Successfully read {len(log_data)} characters from file.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # 3. Parsing Logic
    # Split the log by the "=== Running Test" header
    # We use a lookahead or just split and ignore the preamble
    test_blocks = re.split(r'=== Running Test (\d+)/50 ===', log_data)

    # re.split with capturing group returns [preamble, id1, block1, id2, block2...]
    # We skip the first element (preamble)
    if len(test_blocks) < 2:
        print("⚠️ No test blocks found. Check the file format.")
        return

    data_rows = []

    # Iterate through pairs (Test ID, Content Block)
    for i in range(1, len(test_blocks), 2):
        test_id = test_blocks[i]
        block = test_blocks[i + 1]

        row = {"Test ID": int(test_id)}

        # -- Extract Question --
        q_match = re.search(r"📘 Question:\s*(.*)", block)
        row["Question"] = q_match.group(1).strip() if q_match else "Unknown"

        # -- Extract Category --
        cat_match = re.search(r"📚 Category:\s*(.*)", block)
        row["Category"] = cat_match.group(1).strip() if cat_match else "Unknown"

        # -- Extract Model Answer --
        # Captures text between the Robot emoji and the Checkered Flag emoji
        ans_match = re.search(r"🤖 Model Answer:\s*(.*?)\s*🏁 Evaluation:", block, re.DOTALL)
        row["Model Answer"] = ans_match.group(1).strip() if ans_match else "N/A"

        # -- Extract Context --
        # Captures text between "Context retrieved:" (or Category end) and Model Answer
        # We look for the section before the model answer
        ctx_match = re.search(r"(?:Context retrieved:|Error in retrieving context)(.*?)🤖 Model Answer:", block,
                              re.DOTALL)
        if ctx_match:
            row["Retrieval Context"] = ctx_match.group(1).strip()
        else:
            row["Retrieval Context"] = "Error or Empty"

        # -- Extract Evaluation (JSON parsing) --
        # We find the JSON block after "Evaluation:"
        eval_match = re.search(r"🏁 Evaluation:\s*(\{.*\})", block, re.DOTALL)

        if eval_match:
            json_str = eval_match.group(1)
            try:
                # Parse the JSON string
                metrics = json.loads(json_str)

                # Semantic Alignment (Pedagogical)
                sem = metrics.get("semantic_alignment", {})
                row["Semantic Score"] = sem.get("score", None)
                row["Semantic Passed"] = sem.get("passed", None)
                row["Semantic Reason"] = sem.get("reason", "")

                # Faithfulness
                faith = metrics.get("faithfulness", {})
                row["Faithfulness Score"] = faith.get("score", None)
                row["Faithfulness Passed"] = faith.get("passed", None)
                row["Faithfulness Reason"] = faith.get("reason", "")

            except json.JSONDecodeError:
                row["Semantic Score"] = "JSON Error"
                row["Faithfulness Score"] = "JSON Error"
        else:
            row["Semantic Score"] = None
            row["Faithfulness Score"] = None

        data_rows.append(row)

    print(f"🔄 Processed {len(data_rows)} test cases.")

    # 4. Create DataFrame and Export
    if not data_rows:
        print("⚠️ No data parsed.")
        return

    df = pd.DataFrame(data_rows)

    # Reorder columns for better readability
    desired_order = [
        "Test ID", "Category", "Question", "Model Answer",
        "Semantic Score", "Semantic Passed", "Semantic Reason",
        "Faithfulness Score", "Faithfulness Passed", "Faithfulness Reason",
        "Retrieval Context"
    ]

    # Filter only columns that exist (in case of errors)
    cols = [c for c in desired_order if c in df.columns]
    df = df[cols]

    try:
        df.to_excel(output_file, index=False)
        print(f"🎉 Success! Excel file created: {output_file}")
    except PermissionError:
        print(f"❌ Error: Could not write to {output_file}. Is the Excel file currently open?")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


if __name__ == "__main__":
    parse_log_to_excel(INPUT_FILENAME, OUTPUT_FILENAME)