import streamlit as st
import google.generativeai as genai
from PIL import Image
import pytesseract
import PyPDF2

# Set up Google Gemini API
genai.configure(api_key="AIzaSyCcndZ9HXsQv11Q5floh5EHtpNNE3WRl2E")

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Streamlit app layout
st.title("Medical Agent - Prescription and Symptom Analysis")

# Input section on the left side
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload Prescription (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
user_problem = st.sidebar.text_area("Describe your problem or symptoms:")
user_allergy = st.sidebar.text_input("Any specific allergies?")

# Output section on the right side
st.header("Analysis Results")

if uploaded_file and user_problem:
    prescription_text = ""

    if uploaded_file.type == "application/pdf":
        prescription_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        prescription_text = extract_text_from_image(image)
    else:
        st.error("Unsupported file type. Please upload a PDF or image.")
        st.stop()

    # Updated prompt with strict formatting instructions
    prompt = f"""
    Prescription Data:
    {prescription_text}

    User Problem:
    {user_problem}

    User Allergy:
    {user_allergy}

    Analyze the prescription and user inputs. Provide your response in the exact format below, using the numbered section headers as shown (e.g., '1. Prescription Analysis:'). Do not deviate from this structure:

    1. Prescription Analysis: Extract medicine names, dosage, and patient age if mentioned.
    2. Usage of Medicines: Explain how and when to use each medicine.
    3. Medicine Details: Explain which medicine works for which problem.
    4. Problem Analysis: Identify the patient's problem related to the medication.
    5. Healthcare Tips: Suggest home remedies or natural remedies.
    6. Physical Activity: Suggest exercises or physical activities.
    7. Side Effects: Mention any side effects of the medicines.
    8. Food to Avoid: List foods to avoid based on the user's problem.
    9. Food to Prioritize: List foods that can help recovery.
    """

    # Call the Google Gemini API
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-002')
        response = model.generate_content(prompt)

        # Debug: Display raw response
        st.write("Raw API Response (for debugging):")
        st.text(response.text)

        # Parse response
        full_text = response.text
        sections = {
            "Prescription Analysis": "1. Prescription Analysis:",
            "Usage of Medicines": "2. Usage of Medicines:",
            "Medicine Details": "3. Medicine Details:",
            "Problem Analysis": "4. Problem Analysis:",
            "Healthcare Tips": "5. Healthcare Tips:",
            "Physical Activity": "6. Physical Activity:",
            "Side Effects": "7. Side Effects:",
            "Food to Avoid": "8. Food to Avoid:",
            "Food to Prioritize": "9. Food to Prioritize:"
        }

        for title, marker in sections.items():
            st.subheader(title)
            try:
                if title == "Food to Prioritize":
                    start_text = full_text.split(marker)[1].strip()
                else:
                    next_marker = list(sections.values())[list(sections.keys()).index(title) + 1]
                    start_text = full_text.split(marker)[1].split(next_marker)[0].strip()
                st.write(start_text)
            except IndexError:
                st.write("Unable to parse this section. Check raw response above.")

    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
else:
    st.warning("Please upload a prescription and describe your problem to get analysis.")