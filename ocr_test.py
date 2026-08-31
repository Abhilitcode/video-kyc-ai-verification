import easyocr
import re


# -----------------------------
# 1. Initialize EasyOCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=True)


# -----------------------------
# 2. Read the ID image
# -----------------------------
image_path = r"dummy_id.png"

text_lines = reader.readtext(image_path, detail=0)

# Combine all OCR lines into one text
text = "\n".join(text_lines)

print("\n========== RAW OCR TEXT ==========")
print(text)
print("===================================\n")


# -----------------------------
# 3. Extract NAME
# -----------------------------
name = None

for i, line in enumerate(text_lines):
    clean_line = line.strip()

    if clean_line.lower().startswith("name"):
        if ":" in clean_line:
            value_after_colon = clean_line.split(":", 1)[1].strip()

            if value_after_colon:
                name = value_after_colon
            elif i + 1 < len(text_lines):
                name = text_lines[i + 1].strip()

        break


# -----------------------------
# 4. Extract DOB
# -----------------------------
dob = None

dob_pattern = r"\b\d{2}[-/]\d{2}[-/]\d{4}\b"

dob_match = re.search(dob_pattern, text)

if dob_match:
    dob = dob_match.group()


# -----------------------------
# 5. Extract ID number
# -----------------------------
id_number = None

for i, line in enumerate(text_lines):
    if line.lower().strip().startswith("id"):
        if ":" in line:
            id_number = line.split(":", 1)[1].strip()

        elif i + 1 < len(text_lines):
            id_number = text_lines[i + 1].strip()

        break


# -----------------------------
# 6. Extract ADDRESS
# -----------------------------
address = None

address_lines = []

for i, line in enumerate(text_lines):
    clean_line = line.strip()

    if clean_line.lower().startswith("address"):

        if ":" in clean_line:
            value_after_colon = clean_line.split(":", 1)[1].strip()

            if value_after_colon:
                address_lines.append(value_after_colon)

        for following_line in text_lines[i + 1:]:
            clean_line = following_line.strip()

            if not clean_line:
                continue

            # Stop when we reach the footer/noise
            if clean_line.lower().startswith(
                (
                    "name",
                    "dob",
                    "id",
                    "date of birth",
                    "for demonstration"
                )
            ):
                break

            if clean_line.lower().startswith(
                ("dattsanziv", "datt", "8", "1")
            ):
                break

            address_lines.append(clean_line)

        break


if address_lines:
    address = ", ".join(address_lines)
    address = address.replace(";", "")
    address = address.replace(",,", ",")


# -----------------------------
# 7. Create structured data
# -----------------------------
id_information = {
    "name": name,
    "dob": dob,
    "id_number": id_number,
    "address": address
}


# -----------------------------
# 8. Display final result
# -----------------------------
print("========== STRUCTURED ID DATA ==========")

for key, value in id_information.items():
    print(f"{key}: {value}")

print("=========================================")