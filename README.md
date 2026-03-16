⚙️ How to Run the Project
Follow the steps below to set up and run the **COVID-19 Misinformation Detection System** on your local machine.

📋 Prerequisites
Before running the project, make sure the following are installed on your system:

* Python 3.8 or higher
* pip (Python package manager)
* Minimum 4 GB RAM
* Ollama installed for local LLM verification

📥 Step 1: Clone or Download the Repository
Clone the repository using Git:
git clone https://github.com/your-username/covid-misinformation-detection.git

Then navigate to the project folder:
cd covid-misinformation-detection

🧪 Step 2: Create a Virtual Environment
Create a virtual environment to manage dependencies.
python -m venv venv

Activate the environment:
Windows:
venv\Scripts\activate

📦 Step 3: Install Required Dependencies
Install all the required Python libraries using:
pip install -r requirements.txt

🧠 Step 4: Install spaCy Language Model
The system uses **spaCy for entity recognition and text processing**.
Run the following command:
python -m spacy download en_core_web_sm

🤖 Step 5: Install Ollama (Required for Full Functionality)
This project uses Ollama for local AI verification.
1️⃣ Download Ollama
Download from:
https://ollama.ai/download

2️⃣ Pull the Required AI Model
After installation, open the terminal and run:
ollama pull gemma2:2b

If the model above is not available, you can use alternatives:

ollama pull qwen2.5:3b
ollama pull phi4-mini
ollama pull llama3.1:8b

If it is working correctly, the application header will display **"LLM ON"**.

▶️ Step 6: Run the Application
Start the Streamlit application with the following command:
streamlit run app.py

🌐 Step 7: Open the Application in Browser
After running the command, Streamlit will generate a local URL like this:
http://localhost:8501

Open this URL in your browser to access the application.

🚀 Using the Application
Verify Mode (Default)
1. Enter a claim in the text box
   Example:
Garlic cures COVID-19

2. Click Verify

3. The system will classify the claim as:
* CORRECT
* MISINFORMATION
* UNVERIFIED

4. The system will also display the explanation and matched facts.

Image Verification Mode
1. Go to the Image tab
2. Upload an image (PNG or JPG)
3. Click Verify Image
4. The system extracts text from the image using OCR and analyzes it.


Knowledge Gathering Mode
1. Switch to Gather Mode
2. Click Fetch Facts
3. The system downloads verified facts from trusted sources like WHO and CDC.

🛠 Troubleshooting
ModuleNotFoundError
Run:
pip install -r requirements.txt

spaCy Model Not Found
Run:
python -m spacy download en_core_web_sm

OCR Not Working
Install EasyOCR:
pip install easyocr

"LLM OFF" Showing in App
Check if Ollama is installed and running:
ollama --version
ollama list

If no model is available:
ollama pull gemma2:2b

