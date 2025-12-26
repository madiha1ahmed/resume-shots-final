from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
import os
from flask import Flask, session
from flask_session import Session
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from flask_socketio import SocketIO
import time
#import redis
from dotenv import load_dotenv
load_dotenv()
#from google import genai
import os

from threading import Thread
import uuid

JOBS = {}  # job_id -> {"status": ..., "progress": ..., "total": ..., "emails_data": [], "error": None}


print("🔧 OPENAI_API_KEY present?:", bool(os.getenv("OPENAI_API_KEY")))

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'myapp:'
#app.config['SESSION_REDIS'] = redis.StrictRedis(host='localhost', port=6379, db=0)
app.config['SESSION_FILE_DIR'] = './flask_session_files'
Session(app)

socketio = SocketIO(app, cors_allowed_origins="*")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your_secret_key'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
#client = genai.Client(api_key="AIzaSyDz3IIqX2E74NaYnXK3CmnKRBOCZekOa8A")

def initialize_llm(llm_provider):
    if llm_provider == "deepseek":
        return ChatOpenAI(model="deepseek/deepseek-chat",openai_api_key=OPENROUTER_API_KEY,openai_api_base="https://openrouter.ai/api/v1", temperature=0)
    elif llm_provider == 'gemini':
        return ChatOpenAI(model="google/gemini-2.0-flash-001",openai_api_key=OPENROUTER_API_KEY,openai_api_base="https://openrouter.ai/api/v1", temperature=0)
    # Replace with DeepSeek's chat model
    return ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0) # Default to OpenAI


#llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Extract text from resume

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

current_date = datetime.today().strftime('%B %d, %Y')

def fetch_website_info(website_url):
    """Fetches the main content of a company's website to personalize the cover letter."""
    if not website_url or not website_url.startswith("http"):
        return None  # Ensure the website URL is valid

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(website_url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            if paragraphs:
                return paragraphs[0].get_text().strip()
            headers = soup.find_all(["h1", "h2"])
            if headers:
                return headers[0].get_text().strip()

    except requests.RequestException:
        return None

    return None

# Generate AI-based cover letter
def generate_cover_letter(company_name, job_position, job_description, website_info, resume_text):
    #print(f"🚀 Starting cover letter for {company_name} - {job_position}")
    """
    Generates a professional and truthful cover letter based strictly on resume details.
    """
    llm_provider = session.get('llm_provider', 'openai')  # Default to OpenAI if not set
    llm = initialize_llm(llm_provider)

    # Ensure that website information is included properly if available
    #website_section = (
       # f"\nI reviewed the information available on {company_name}'s website and was particularly drawn to {website_info}. "
        #if website_info and website_info.lower() != "no website available"
        #else ""
    #)

    if website_info and website_info.strip():
        website_section = f"I reviewed {company_name}'s website and was particularly drawn to {website_info}."
    else:
        website_section = ""  # Leave blank if no website info


    # Get the current date
    current_date = datetime.today().strftime('%B %d, %Y')

    prompt = ChatPromptTemplate.from_template(f"""
    You are a **highly precise and fact-based** job application assistant. 
    Your task is to generate a **truthful, concise, and professional** cover letter based **ONLY on the provided resume**.

    **Candidate Details:**
    - **Name:** [Name from {resume_text}]
    - **Address:** [Address from {resume_text}]
    - **Phone Number:** [Phone Number from {resume_text}]
    - **Date:** {current_date}

    **Job Application Details:**
    - **Company:** {company_name}
    - **Position:** {job_position}
    - **Job Description:** {job_description}

    **Candidate's Resume:**
    {resume_text}

    {website_section}

    **STRICT RULES:**
    - **DO NOT fabricate any experiences, degrees, or skills that are NOT present in the resume.**
    - **Only extract relevant skills and qualifications from the resume text above.**
    - **If a required skill is missing, acknowledge it politely and express eagerness to learn.**
    - **No generic statements like "passionate about teaching" unless proven from the resume.**
    - **If website insights are available, integrate them naturally.**
    - **Do not include LinkedIn Profile in the cover letter**
    - **No more than TWO paragraphs.**

    **Example Cover Letter Structure:**
    
    Dear Hiring Team of {company_name},

    I am excited to apply for the {job_position} role at {company_name}. My background in **[relevant expertise from {resume_text}]** has enabled me to **[explain how your experience aligns with job responsibilities]**. {website_section}

    I look forward to the opportunity to discuss how my expertise in **[relevant experience in {resume_text}]** can contribute to **[company's goal or project mentioned in job description]**.

    Sincerely,  
    [Name from {resume_text}] 
    [City, Country from Address in {resume_text}]
    Phone: [Phone number from {resume_text}]
    
    """)

    # Create an LLM Chain using LangChain
    #chain = prompt | llm
    #response = chain.invoke({
        #"company_name": company_name,
        #"job_position": job_position,
        #"job_description": job_description,
        #"resume_text": resume_text,  # Explicitly passing the extracted resume
        #"website_info_integration": website_section,
    #})

    #return response.content  # Extract the generated cover letter

    if llm_provider == "deepseek":
        chain = prompt | llm

    # Pass input variables to the model
        response = chain.invoke({
        "company_name": company_name,
        "job_position": job_position,
        "job_description": job_description,
        "resume_text": resume_text,  # Explicitly passing the extracted resume
        "website_info_integration": website_section,
        })

        #print(f"✅ Finished cover letter for {company_name} - {job_position}")

        return response.content
    
    elif llm_provider == "gemini":
        chain = prompt | llm

    # Pass input variables to the model
        response = chain.invoke({
        "company_name": company_name,
        "job_position": job_position,
        "job_description": job_description,
        "resume_text": resume_text,  # Explicitly passing the extracted resume
        "website_info_integration": website_section,
        })

        return response.content
    
    else:
        chain = prompt | llm
        response = chain.invoke({
        "company_name": company_name,
        "job_position": job_position,
        "job_description": job_description,
        "resume_text": resume_text,  # Explicitly passing the extracted resume
        "website_info_integration": website_section,
    })
        return response.content


import json

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        session.clear()

        if 'email_file' not in request.files or 'resume_files' not in request.files:
            return jsonify({"error": "Missing files"}), 400

        email_file = request.files['email_file']
        resume_files = request.files.getlist('resume_files')

        if email_file.filename == "" or len(resume_files) == 0:
            return jsonify({"error": "No files selected"}), 400

        email_path = os.path.join(app.config['UPLOAD_FOLDER'], email_file.filename)
        email_file.save(email_path)

        resume_texts = {}
        resume_paths = {}

        for resume_file in resume_files:
            if resume_file.filename == "":
                continue  # Ignore empty filenames

            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(resume_path)

            resume_texts[resume_file.filename] = extract_text_from_pdf(resume_path)
            resume_paths[resume_file.filename] = resume_path

        # Save resume texts to a JSON file instead of the session
        with open("resume_texts.json", "w") as f:
            json.dump(resume_texts, f)

        session['email_path'] = email_path
        session['resume_paths'] = resume_paths  # Keep only file paths in session

        return jsonify({"success": True, "message": "Files uploaded successfully!"})

    return render_template('upload.html')

def is_relevant_resume(job_description, resume_text):
    """
    Determines if the resume is relevant to the job description by checking domain-specific skills.
    If the job description contains IT-related skills, and the resume has overlapping skills, it is considered relevant.
    """
    llm = initialize_llm(session.get('llm_provider', 'openai'))

    prompt = f"""
    You are a strict hiring assistant.

    Your job is to determine if a **resume is relevant** for a given **job description** based on industry skills.
    
    **Instructions:**
    - If the resume contains **relevant technical skills** (e.g., Python, SQL, Cloud Computing, AI, IT Security) for an IT-related job, it should be marked as "YES".
    - If it is in a **completely different field** (e.g., receptionist for IT Manager), it should be marked as "NO".
    - Consider **data engineers, software engineers, and AI researchers relevant for IT Management** roles if they have IT infrastructure knowledge.

    **Examples:**
    - Receptionist applying for IT Manager → "NO"
    - Civil Engineer applying for Software Engineer → "NO"
    - Data Scientist applying for AI Researcher → "YES"
    - Data Engineer applying for IT Manager → "YES"
    - Python Developer applying for IT Consultant → "YES"
    - Doctor applying for Cybersecurity Analyst → "NO"

    **Job Description:**
    {job_description}

    **Resume:**
    {resume_text}  # Limiting token usage

    Answer with only "YES" or "NO".
    """

    try:
        result = llm.invoke(prompt).content.strip().upper()
        return result == "YES"
    except Exception as e:
        print(f"❌ Filter error: {e}")
        return False



def select_best_resume(job_description, resume_texts):
    """
    Uses LLM to select the most relevant resume for the given job description.
    No pre-filtering, purely prompt-based selection.
    """
    llm_provider = session.get('llm_provider', 'openai')
    llm = initialize_llm(llm_provider)

    # Ensure there are resumes to choose from
    if not resume_texts:
        print("❌ No resumes available for selection.")
        return None

    # Generate the selection prompt
    combined_prompt = f"""
    You are an expert recruiter.

    Your task is to select the **best and most closely (even remotely closest)** resume for a given job description.
    Below are multiple resumes. Choose the **most relevant one** based on education, experience, skills, volunteering, interest and industry match.

    **Job Description:**
    {job_description}

    **Available Resumes:**
    {json.dumps(resume_texts, indent=2)}

    **Important Instructions:**
    - Choose the resume that **best aligns with the job description**.
    - If multiple resumes match, select the most experienced candidate.
    - **Do NOT fabricate or assume skills that are not present in the resume.**
    - **Output should be ONLY the filename of the best resume. No explanations, no extra text.**

    **Example Output Format:**
    Resume-2025.pdf
    """

    try:
        best_filename = llm.invoke(combined_prompt).content.strip()
        return best_filename if best_filename in resume_texts else None
    except Exception as e:
        print(f"❌ Resume selection error: {e}")
        return None

def process_cover_letter_job(job_id, email_path, resume_paths, llm_provider):
    """
    Background worker: reads the Excel, picks resumes, calls LLM,
    and fills JOBS[job_id]["emails_data"].
    """
    try:
        # Load Excel/CSV
        df = pd.read_excel(email_path) if email_path.endswith('.xlsx') else pd.read_csv(email_path)
        total_jobs = len(df)

        # Load resume texts from JSON
        try:
            with open("resume_texts.json", "r") as f:
                resume_texts = json.load(f)
        except Exception as e:
            print(f"❌ Error loading resume texts: {e}")
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = f"Error loading resume texts: {e}"
            return

        emails_data = []

        JOBS[job_id]["total"] = total_jobs
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["progress"] = 0

        for index, row in df.iterrows():
            company_name = row['Company Name']
            job_position = row['Job Position']
            job_description = row.get('Job Description', '').strip()
            recipient_email = row['Email']
            company_website = row.get('Website', '')

            # Pick best resume for this job
            best_resume_filename = select_best_resume(job_description, resume_texts)
            print(f"🔍 Selected Resume for {job_position}: {best_resume_filename}")

            best_resume_text = resume_texts.get(best_resume_filename, "")

            # Optional website info
            website_info = fetch_website_info(company_website) if company_website else None

            # Generate cover letter
            cover_letter = generate_cover_letter(
                company_name,
                job_position,
                job_description,
                website_info,
                best_resume_text
            )

            emails_data.append({
                'recipient_email': recipient_email,
                'company_name': company_name,
                'job_position': job_position,
                'job_description': job_description or "No job description available.",
                'selected_resume': best_resume_filename,
                'cover_letter': cover_letter
            })

            # Update progress
            JOBS[job_id]["progress"] = index + 1

        JOBS[job_id]["emails_data"] = emails_data
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["error"] = None

        print(f"✅ Job {job_id} completed with {len(emails_data)} emails.")

    except Exception as e:
        print(f"❌ Unexpected error in job {job_id}: {e}")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.route('/generate_cover_letters', methods=['POST'])
def generate_cover_letters():
    """
    Starts a background job and returns a job_id immediately.
    The heavy work is done in process_cover_letter_job.
    """
    email_path = session.get('email_path')
    resume_paths = session.get('resume_paths', {})
    llm_provider = session.get('llm_provider', 'openai')

    if not email_path or not resume_paths:
        return jsonify({"success": False, "message": "Please upload Excel and resumes first."}), 400

    # Create a new job id
    job_id = str(uuid.uuid4())

    # Initialize job state
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "total": 0,
        "emails_data": [],
        "error": None,
    }

    # Start a background thread to process this job
    t = Thread(
        target=process_cover_letter_job,
        args=(job_id, email_path, resume_paths, llm_provider),
        daemon=True
    )
    t.start()

    # Return immediately – frontend will poll /job_status/<job_id>
    return jsonify({"success": True, "job_id": job_id})




@app.route('/review/<job_id>', methods=['GET'])
def review_emails(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return redirect(url_for('upload_files'))

    emails_data = job["emails_data"]

    # Also put in session so send_email keeps working as before
    session['emails_data'] = emails_data

    return render_template('review.html', emails_data=emails_data, job_id=job_id)


 # "dclo ewei hyrg ltar"

from flask import Flask, render_template, request, redirect, url_for, session, jsonify

@app.route('/send_email', methods=['POST'])
def send_email():
    sender_email = "madiha1ahmed@gmail.com"
    sender_password = "uxim fruv ijqv hwdi"  # Use a Gmail app password
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    emails_data = session.get('emails_data', [])
    resume_paths = session.get('resume_paths', {})  # ✅ Now using stored resume paths

    approved_indexes = request.form.getlist("approve_")  # Get checked indexes
    print("✅ Approved indexes:", approved_indexes)

    filtered_emails = [emails_data[int(i) - 1] for i in approved_indexes]
    print(f"✅ {len(filtered_emails)} emails to be sent.")

    if not filtered_emails:
        print("❌ No emails selected for sending!")
        return jsonify({"success": False, "message": "No emails were selected!"})

    for i, email in enumerate(filtered_emails):
        edited_cover_letter = request.form.get(f"cover_letter_{int(approved_indexes[i])}")
        if edited_cover_letter:
            email["cover_letter"] = edited_cover_letter.strip()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        print("✅ Successfully logged into SMTP server.")

        for email in filtered_emails:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = email['recipient_email']
            msg['Bcc'] = sender_email  # ✅ Adds sender in BCC
            msg['Subject'] = f"Job Application for {email['job_position']}"

            msg.attach(MIMEText(email['cover_letter'], 'plain'))

            # ✅ Retrieve the correct resume path
            resume_filename = email.get("selected_resume")  # The selected resume filename
            resume_path = resume_paths.get(resume_filename)  # Get actual path from stored resumes

            if not resume_path:
                print(f"❌ Resume path not found for {resume_filename}")
                continue  # Skip this email if resume is missing

            with open(resume_path, 'rb') as resume_file:
                attach_file = MIMEBase('application', 'octet-stream')
                attach_file.set_payload(resume_file.read())
                encoders.encode_base64(attach_file)
                attach_file.add_header('Content-Disposition', f'attachment; filename={os.path.basename(resume_path)}')
                msg.attach(attach_file)

            server.send_message(msg)
            print(f"✅ Email sent to {email['recipient_email']} for {email['job_position']}")

        server.quit()
        print("✅ SMTP server connection closed.")

    except Exception as e:
        print(f"❌ Failed to send emails. Error: {e}")
        return jsonify({"success": False, "message": f"Error: {e}"})

    return jsonify({"success": True, "redirect_url": "/success"})


from flask import jsonify

@app.route("/test_openai")
def test_openai():
    print("🧪 /test_openai hit")
    try:
        from openai import OpenAI
        client = OpenAI()  # will use OPENAI_API_KEY from env

        resp = client.chat.completions.create(
            model="gpt-4o",  # or your actual model
            messages=[{"role": "user", "content": "Say 'ok' only."}],
            max_tokens=5,
            timeout=20,  # very strict, just for test
        )
        msg = resp.choices[0].message.content
        print("🧪 OpenAI test response:", msg)
        return jsonify({"status": "success", "reply": msg})
    except Exception as e:
        print("❌ OpenAI test error:", repr(e))
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/success')
def success():
    return render_template('success.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/view_resume/<filename>')
def view_resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/job_status/<job_id>')
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "message": "Job not found"}), 404

    return jsonify({
        "success": True,
        "status": job["status"],      # "queued" | "running" | "done" | "error"
        "progress": job["progress"],  # rows processed
        "total": job["total"],        # total rows
        "error": job["error"],
    })




#if __name__ == '__main__':
    #socketio.run(app,debug=True)
    s#ocketio.run(app, debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
