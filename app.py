import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid Tcl/Tk issues
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template, send_file, session, redirect, url_for, flash
from io import BytesIO
import base64
from pathlib import Path
import os
import zipfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session and flash messages

# Register Times New Roman font
# Update the path to times.ttf on your system
try:
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'C:/Windows/Fonts/times.ttf'))
except:
    # Fallback to a similar font if Times New Roman is not available
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'Helvetica'))  # Replace with path to times.ttf if needed

# Hardcoded teacher credentials with first-login flag
TEACHER_CREDENTIALS = {
    'username': 'teacher',
    'password': generate_password_hash('password123'),  # Hashed password
    'first_login': True  # Flag to track first login
}

def load_data(file):
    try:
        df = pd.read_csv(file)
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        return f"Error loading file: {e}"

def analyze_performance(df):
    print("DataFrame before processing:")
    print(df)
    print("Columns:", df.columns.tolist())

    # Explicitly define score columns based on the CSV headings
    score_cols = [col for col in df.columns if col in ['BEEE', 'Chemistry', 'Physics']]
    print("Score columns:", score_cols)

    if not score_cols:
        raise ValueError("No score columns (e.g., 'BEEE', 'Chemistry', 'Physics') found in the CSV file.")

    # Convert score columns to numeric, coercing errors to NaN
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['AverageScore'] = df[score_cols].mean(axis=1)
    print("AverageScore column:")
    print(df['AverageScore'])

    if df['AverageScore'].isna().all():
        raise ValueError("All AverageScore values are NaN. Check if score columns contain numeric data.")

    # Compute class-level statistics
    stats = {
        'class_avg': df['AverageScore'].mean(),
        'class_median': df['AverageScore'].median(),
        'top_performers': df.nlargest(3, 'AverageScore')[['Name', 'AverageScore']].to_dict('records'),
        'bottom_performers': df.nsmallest(3, 'AverageScore')[['Name', 'AverageScore']].to_dict('records'),
        'failing_students': df[df['AverageScore'] < 60][['Name', 'AverageScore']].to_dict('records'),
        'score_distribution': df['AverageScore'].dropna().tolist(),
        'subject_averages': {col: df[col].mean() for col in score_cols}  # Average score per subject
    }

    stats['correlation'] = df['Attendance'].corr(df['AverageScore']) if 'Attendance' in df.columns else None
    weakest_subject = df[score_cols].mean().idxmin() if score_cols else None
    stats['weakest_subject'] = weakest_subject

    return stats, df

def generate_charts(stats, df, output_dir='reports'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Grade distribution bar chart
    grade_dist = None
    if stats['score_distribution'] and not all(np.isnan(stats['score_distribution'])):
        plt.figure(figsize=(8, 6))
        plt.hist(stats['score_distribution'], bins=10, color='skyblue', edgecolor='black')
        plt.title('Grade Distribution')
        plt.xlabel('Average Score')
        plt.ylabel('Number of Students')
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        grade_dist = base64.b64encode(img.getvalue()).decode('utf-8')
    else:
        print("Warning: No valid data for grade distribution plot.")

    # Attendance vs performance scatter with trend line
    attendance_plot = None
    if stats['correlation'] is not None and not df['Attendance'].isna().all() and not df['AverageScore'].isna().all():
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Attendance'], df['AverageScore'], color='green', label='Data Points', alpha=0.6)
        z = np.polyfit(df['Attendance'], df['AverageScore'], 1)
        p = np.poly1d(z)
        plt.plot(df['Attendance'], p(df['Attendance']), color='blue', linestyle='--', label='Trend Line')
        plt.title('Attendance vs Performance')
        plt.xlabel('Attendance (%)')
        plt.ylabel('Average Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        attendance_plot = base64.b64encode(img.getvalue()).decode('utf-8')
    else:
        print("Warning: No valid data for attendance vs performance plot.")

    return grade_dist, attendance_plot

def generate_grade_card(student, score_cols, class_stats, output_dir='grade_cards'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Sanitize student name for filename
    student_name = student['Name'].replace(' ', '_')
    pdf_path = output_dir / f"grade_card_{student_name}.pdf"

    # Create PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Watermark
    c.saveState()
    c.setFont("TimesNewRoman", 60)
    c.setFillColor(colors.grey, alpha=0.1)
    c.translate(width/2, height/2)
    c.rotate(45)
    c.drawCentredString(0, 0, "Student Analytics")
    c.restoreState()

    # Header Background (Grayscale)
    c.setFillColor(colors.grey)
    c.rect(0, height - 80, width, 80, fill=1, stroke=0)

    # Header Title
    c.setFont("TimesNewRoman", 24)
    c.setFillColor(colors.black)
    c.drawCentredString(width/2, height - 60, "Student Grade Card")

    # Content Border (Sharp Corners)
    c.setStrokeColor(colors.black)
    c.setLineWidth(2)
    c.rect(40, 100, width - 80, height - 220, stroke=1, fill=0)

    # Student Info
    c.setFont("TimesNewRoman", 16)
    c.setFillColor(colors.black)
    c.drawString(60, height - 120, f"Student ID: {student['StudentID']}")
    c.drawString(60, height - 150, f"Name: {student['Name']}")

    # Scores Table
    c.setFont("TimesNewRoman", 14)
    c.drawString(60, height - 190, "Scores:")
    
    # Prepare table data
    table_data = [["Subject", "Score"]]
    for col in score_cols:
        table_data.append([col, f"{student[col]:.2f}"])
    table_data.append(["Average Score", f"{student['AverageScore']:.2f}"])
    table_data.append(["Attendance", f"{student['Attendance']:.2f}%"])

    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    # Draw table
    table.wrapOn(c, width - 120, height)
    table.drawOn(c, 60, height - 190 - len(table_data) * 25)

    # Pass/Fail Indicator
    y_position = height - 190 - len(table_data) * 25 - 50
    status = "Pass" if student['AverageScore'] >= 60 else "Fail"
    c.setFillColor(colors.black)
    c.circle(80, y_position + 10, 15, fill=0, stroke=1)
    c.setFont("TimesNewRoman", 20)
    if status == "Pass":
        c.drawCentredString(80, y_position + 5, "✓")
    else:
        c.drawCentredString(80, y_position + 5, "✗")
    c.setFont("TimesNewRoman", 14)
    c.setFillColor(colors.black)
    c.drawString(110, y_position + 5, f"Status: {status}")

    # Insights Section
    y_position -= 40
    c.setFont("TimesNewRoman", 14)
    c.drawString(60, y_position, "Insights:")
    y_position -= 20

    # Generate insights
    insights = []
    class_avg = class_stats['class_avg']
    subject_averages = class_stats['subject_averages']

    # Insight 1: Performance relative to class average
    if student['AverageScore'] > class_avg:
        insights.append(f"Your average score ({student['AverageScore']:.2f}) is above the class average ({class_avg:.2f}).")
    else:
        insights.append(f"Your average score ({student['AverageScore']:.2f}) is below the class average ({class_avg:.2f}).")

    # Insight 2: Strongest and weakest subjects
    scores = {col: student[col] for col in score_cols}
    strongest_subject = max(scores, key=scores.get)
    weakest_subject = min(scores, key=scores.get)
    insights.append(f"Strongest subject: {strongest_subject} ({scores[strongest_subject]:.2f}).")
    insights.append(f"Weakest subject: {weakest_subject} ({scores[weakest_subject]:.2f}).")

    # Insight 3: Recommendation based on performance and attendance
    if student['AverageScore'] < 60:
        insights.append("Recommendation: Consider seeking extra help in weaker subjects.")
    elif student['Attendance'] < 75:
        insights.append("Recommendation: Improve attendance to potentially boost performance.")
    else:
        insights.append("Recommendation: Keep up the good work!")

    # Draw insights
    c.setFont("TimesNewRoman", 12)
    for insight in insights:
        c.drawString(80, y_position, f"- {insight}")
        y_position -= 20

    # Footer
    c.setFont("TimesNewRoman", 10)
    c.setFillColor(colors.black)
    c.drawCentredString(width/2, 50, f"Generated on {datetime.now().strftime('%Y-%m-%d')}")
    c.setFont("TimesNewRoman", 10)
    c.drawCentredString(width/2, 30, "Student Performance Analytics Tool")

    c.showPage()
    c.save()
    return pdf_path

def generate_all_grade_cards(df, score_cols, class_stats, output_dir='grade_cards'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pdf_files = []
    for _, student in df.iterrows():
        pdf_path = generate_grade_card(student, score_cols, class_stats, output_dir)
        pdf_files.append(pdf_path)

    # Create a ZIP file containing all grade cards
    zip_path = output_dir / "grade_cards.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pdf_file in pdf_files:
            zipf.write(pdf_file, pdf_file.name)

    # Clean up individual PDF files
    for pdf_file in pdf_files:
        os.remove(pdf_file)

    return zip_path

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if (username == TEACHER_CREDENTIALS['username'] and 
            check_password_hash(TEACHER_CREDENTIALS['password'], password)):
            session['logged_in'] = True
            session['username'] = username
            # Check if it's the first login
            if TEACHER_CREDENTIALS['first_login']:
                flash('Please change your password.', 'info')
                return redirect(url_for('change_password'))
            flash('Login successful!', 'success')
            return redirect(url_for('upload_file'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Verify current password
        if not check_password_hash(TEACHER_CREDENTIALS['password'], current_password):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('change_password'))

        # Check if new passwords match
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return redirect(url_for('change_password'))

        # Update the password and first_login flag
        TEACHER_CREDENTIALS['password'] = generate_password_hash(new_password)
        TEACHER_CREDENTIALS['first_login'] = False
        flash('Password changed successfully!', 'success')
        return redirect(url_for('upload_file'))

    return render_template('change_password.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'error')
            return redirect(url_for('upload_file'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('upload_file'))

        df = load_data(file)
        if isinstance(df, str):
            flash(df, 'error')
            return redirect(url_for('upload_file'))

        try:
            stats, df = analyze_performance(df)
            grade_dist, attendance_plot = generate_charts(stats, df)

            insights = []
            if stats['failing_students']:
                insights.append(f"{stats['failing_students'][0]['Name']} has the lowest average ({stats['failing_students'][0]['AverageScore']:.2f}) and may need intervention.")
            if stats['weakest_subject']:
                insights.append(f"The class struggled most with {stats['weakest_subject']}, indicating that topic needs review.")

            # Store DataFrame and stats in session
            session['df'] = df.to_dict('records')
            session['stats'] = stats

            return render_template('results.html', stats=stats, grade_dist=grade_dist, 
                                 attendance_plot=attendance_plot, insights=insights)
        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('upload_file'))

    return render_template('upload.html')

@app.route('/download_grade_cards', methods=['POST'])
def download_grade_cards():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Retrieve DataFrame and stats from session
    if 'df' not in session or 'stats' not in session:
        flash('No data available to generate grade cards.', 'error')
        return redirect(url_for('upload_file'))

    df_data = session['df']
    class_stats = session['stats']
    df = pd.DataFrame(df_data)

    # Define score columns
    score_cols = [col for col in df.columns if col in ['BEEE', 'Chemistry', 'Physics']]

    # Generate grade cards and ZIP file
    zip_path = generate_all_grade_cards(df, score_cols, class_stats)

    # Clear session data after download
    session.pop('df', None)
    session.pop('stats', None)

    # Send the ZIP file for download
    return send_file(
        zip_path,
        as_attachment=True,
        download_name='grade_cards.zip',
        mimetype='application/zip'
    )

if __name__ == '__main__':
    app.run(debug=True)