from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, make_response, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import os
from datetime import datetime, timezone
from dateutil import parser  
from proctoring_system import ProctoringSystem
from dotenv import load_dotenv
from functools import wraps
from werkzeug.utils import safe_join
import base64
import numpy as np
from deepface import DeepFace
import cv2
import pickle
import traceback
from proctoring_face_analyzer import analyze_frame
import uuid
import re
from werkzeug.utils import secure_filename
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# SQLite Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///exam_portal.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# File upload configuration
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
EXAM_FILES_FOLDER = 'static/uploads/exam_files'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize proctoring system
proctoring_system = ProctoringSystem()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(EXAM_FILES_FOLDER, exist_ok=True)

# Flask-Admin setup
admin = Admin(app, name='Proctoring Admin', template_mode='bootstrap4')

class SecureModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated and getattr(current_user, 'role', None) == 'admin'
    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('login'))

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    name = db.Column(db.String(100))
    aadhar_number = db.Column(db.String(12), unique=True)
    photo_path = db.Column(db.String(200))
    role = db.Column(db.String(20), nullable=False)  # 'student', 'faculty', 'admin'
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Faculty specific fields
    department = db.Column(db.String(100))
    designation = db.Column(db.String(100))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Integer, nullable=False)  # in minutes
    total_marks = db.Column(db.Integer, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20), nullable=False)  # 'mcq', 'theory'
    marks = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    answer_text = db.Column(db.Text)
    marks_obtained = db.Column(db.Float)
    evaluated_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    evaluated_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ProctoringLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    event_type = db.Column(db.String(50), nullable=False)  # 'face_detected', 'face_not_detected', 'multiple_faces', etc.
    confidence = db.Column(db.Float)
    image_path = db.Column(db.String(200))

class ExamSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    anomalies = db.Column(db.Text)  # Store anomalies as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ExamFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # 'question_paper', 'keywords'
    file_path = db.Column(db.String(500), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class ExamKeyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    keyword = db.Column(db.String(100), nullable=False)
    weight = db.Column(db.Float, default=1.0)  # Importance weight for the keyword
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StudentInvitation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    student_email = db.Column(db.String(120), nullable=False)
    invitation_code = db.Column(db.String(50), unique=True, nullable=False)
    is_accepted = db.Column(db.Boolean, default=False)
    accepted_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnswerFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Frontend Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            if user.is_verified:
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('verification_pending'))
        else:
            flash('Invalid email or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('verification_pending'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'student':
        exams = Exam.query.filter(Exam.end_time > datetime.utcnow()).all()
    else:
        exams = Exam.query.filter_by(created_by=current_user.id).all()
    print(f"Dashboard: current_user.photo_path: {current_user.photo_path}") # Debug log
    return render_template('dashboard.html', exams=exams)

@app.route('/verification-pending')
def verification_pending():
    return render_template('verification_pending.html')

# Faculty Routes
@app.route('/faculty-register', methods=['GET'])
def faculty_register_page():
    return render_template('faculty_register.html')

@app.route('/faculty-login', methods=['GET'])
def faculty_login_page():
    return render_template('faculty_login.html')

@app.route('/faculty-dashboard')
@login_required
def faculty_dashboard():
    if current_user.role != 'faculty':
        return redirect(url_for('dashboard'))
    exams = Exam.query.filter_by(created_by=current_user.id).all()
    print(f"Faculty Dashboard: current_user.photo_path: {current_user.photo_path}") # Debug log
    return render_template('faculty_dashboard.html', exams=exams)

# Faculty API Routes
@app.route('/api/faculty/register', methods=['POST'])
def api_faculty_register():
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')
        password = data.get('password')
        username = data.get('username')
        department = data.get('department')
        designation = data.get('designation')
        photo = data.get('photo')  # Base64 encoded photo

        if not all([email, name, password, username, department, designation, photo]):
            return jsonify({'error': 'All fields are required'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already taken'}), 400

        # Save photo
        filename = None
        photo_path_db = None
        if photo:
            try:
                faculty_upload_dir = os.path.join(app.root_path, 'static', 'uploads', 'faculty')
                os.makedirs(faculty_upload_dir, exist_ok=True)
                filename = f"faculty_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                photo_full_path = os.path.join(faculty_upload_dir, filename)
                
                # Convert base64 to image and save
                image_data = base64.b64decode(photo)
                with open(photo_full_path, 'wb') as f:
                    f.write(image_data)
                print(f"Photo saved: {filename}")  # Debug log
                photo_path_db = os.path.join('uploads', 'faculty', filename)
                print(f"Photo path in DB: {photo_path_db}")  # Debug log
            except Exception as e:
                print(f"Error saving photo: {str(e)}")  # Debug log
                return jsonify({'error': f'Error saving photo: {str(e)}'}), 500

        user = User(
            email=email,
            name=name,
            username=username,
            department=department,
            designation=designation,
            photo_path=photo_path_db,
            role='faculty'
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        print(f"Faculty registered successfully: {name} ({email})")  # Debug log
        
        return jsonify({'message': 'Registration successful! Please wait for admin verification.'}), 201
    except Exception as e:
        print(f"Registration error: {str(e)}")  # Debug log
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/faculty/login', methods=['POST'])
def api_faculty_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email, role='faculty').first()
    
    if user and user.check_password(password):
        if user.is_verified:
            login_user(user)
            return jsonify({
                'message': 'Login successful',
                'redirect': '/faculty-dashboard'
            }), 200
        else:
            return jsonify({
                'error': 'Your account is pending verification',
                'status': 'pending',
                'redirect': '/verification-pending'
            }), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401

# API Routes
@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        print("Student registration request received:", data)  # Debug log
        email = data.get('email')
        name = data.get('name')
        password = data.get('password')
        username = data.get('username')
        aadhar_number = data.get('aadhar_number')
        photo = data.get('photo')  # Base64 encoded photo

        if not all([email, name, password, username, aadhar_number, photo]):
            print("Missing required fields")  # Debug log
            return jsonify({'error': 'All fields are required'}), 400

        if User.query.filter_by(email=email).first():
            print(f"Email already registered: {email}")  # Debug log
            return jsonify({'error': 'Email already registered'}), 400

        if User.query.filter_by(username=username).first():
            print(f"Username already taken: {username}")  # Debug log
            return jsonify({'error': 'Username already taken'}), 400

        if User.query.filter_by(aadhar_number=aadhar_number).first():
            print(f"Aadhar number already registered: {aadhar_number}")  # Debug log
            return jsonify({'error': 'Aadhar number already registered'}), 400

        # Save photo
        filename = None
        photo_path_db = None
        if photo:
            try:
                student_upload_dir = os.path.join(app.root_path, 'static', 'uploads', 'students')
                os.makedirs(student_upload_dir, exist_ok=True)
                filename = f"{aadhar_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                photo_full_path = os.path.join(student_upload_dir, filename)
                
                # Convert base64 to image and save
                image_data = base64.b64decode(photo)
                with open(photo_full_path, 'wb') as f:
                    f.write(image_data)
                print(f"Photo saved: {filename}")  # Debug log
                photo_path_db = os.path.join('uploads', 'students', filename)
                print(f"Photo path in DB: {photo_path_db}")  # Debug log
            except Exception as e:
                print(f"Error saving photo: {str(e)}")  # Debug log
                return jsonify({'error': f'Error saving photo: {str(e)}'}), 500

        user = User(
            email=email,
            name=name,
            username=username,
            aadhar_number=aadhar_number,
            photo_path=photo_path_db,
            role='student'  # Default role for new registrations
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        print(f"Student registered successfully: {name} ({email})")  # Debug log
        
        return jsonify({'message': 'Registration successful! Please wait for verification.'}), 201
    except Exception as e:
        print(f"Registration error: {str(e)}")  # Debug log
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    username = data.get('username')

    user = None
    # Try to find user by email first
    if email:
        user = User.query.filter_by(email=email).first()
    # For admin, also allow login by username
    if not user and username:
        user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        if user.role == 'admin':
            login_user(user)
            return jsonify({
                'message': 'Login successful',
                'redirect': '/admin-dashboard'
            }), 200
        elif user.is_verified:
            login_user(user)
            return jsonify({
                'message': 'Login successful',
                'redirect': '/dashboard' if user.role == 'student' else '/faculty-dashboard'
            }), 200
        else:
            return jsonify({
                'error': 'Your account is pending verification',
                'status': 'pending',
                'redirect': '/verification-pending'
            }), 200
    else:
        return jsonify({'error': 'Invalid email/username or password'}), 401

@app.route('/api/exam/start/<int:exam_id>', methods=['POST'])
@login_required
def start_exam(exam_id):
    exam = Exam.query.get_or_404(exam_id)
    
    # Check if exam is currently available
    now = datetime.utcnow()
    if not (exam.start_time <= now <= exam.end_time):
        return jsonify({'error': 'Exam is not currently available'}), 400
    
    # Start proctoring
    proctoring_system.start_monitoring(current_user.id)
    
    # Create exam session
    session = ExamSession(
        user_id=current_user.id,
        exam_id=exam_id,
        start_time=now
    )
    db.session.add(session)
    db.session.commit()
    
    return jsonify({'message': 'Exam started', 'session_id': session.id}), 200

@app.route('/api/exam/end/<int:session_id>', methods=['POST'])
@login_required
def end_exam(session_id):
    session = ExamSession.query.get_or_404(session_id)
    
    if session.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Stop proctoring
    proctoring_system.stop_monitoring()
    
    # Update session
    session.end_time = datetime.utcnow()
    session.anomalies = str(proctoring_system.get_anomalies())
    db.session.commit()
    
    return jsonify({'message': 'Exam ended successfully'}), 200

@app.route('/api/exam/anomalies/<int:session_id>', methods=['GET'])
@login_required
def get_exam_anomalies(session_id):
    session = ExamSession.query.get_or_404(session_id)
    
    if session.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify({'anomalies': session.anomalies}), 200

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Admin Routes
@app.route('/admin-login', methods=['GET'])
def admin_login_page():
    return render_template('admin_login.html')

@app.route('/admin-register', methods=['GET'])
def admin_register_page():
    return render_template('admin_register.html')

@app.route('/admin-dashboard')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

# Admin API Routes
@app.route('/api/admin/users')
@login_required
@admin_required
def get_users():
    print("Fetching students...")  # Debug log
    users = User.query.filter_by(role='student').all()
    print(f"Found {len(users)} students")  # Debug log
    for user in users:  # Debug log
        print(f"Student: {user.name} ({user.email}) - Role: {user.role}")
    return jsonify([{
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'username': user.username,
        'is_verified': user.is_verified,
        'created_at': user.created_at.isoformat() if user.created_at else None
    } for user in users])

@app.route('/api/admin/users/<int:user_id>')
@login_required
@admin_required
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    # Ensure photo_path is correctly formatted for display
    display_photo_path = None
    if user.photo_path:
        if user.role == 'student':
            # If the path doesn't already include 'uploads/students', add it
            if 'uploads/students' not in user.photo_path:
                display_photo_path = os.path.join('uploads', 'students', os.path.basename(user.photo_path))
            else:
                display_photo_path = user.photo_path
        else:
            display_photo_path = user.photo_path

    print(f"Student photo path: {display_photo_path}")  # Debug log
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'username': user.username,
        'is_verified': user.is_verified,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'photo_path': display_photo_path
    })

@app.route('/api/admin/users/<int:user_id>/verify', methods=['POST'])
@login_required
@admin_required
def verify_student(user_id):
    user = User.query.get_or_404(user_id)
    user.is_verified = True
    db.session.commit()
    return jsonify({'message': 'User verified successfully'})

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_student(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted successfully'})

@app.route('/api/admin/faculty')
@login_required
@admin_required
def get_faculty():
    print("Fetching faculty...")  # Debug log
    faculty = User.query.filter_by(role='faculty').all()
    print(f"Found {len(faculty)} faculty members")  # Debug log
    for user in faculty:  # Debug log
        print(f"Faculty: {user.name} ({user.email}) - Role: {user.role}")
    return jsonify([{
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'username': user.username,
        'department': user.department,
        'designation': user.designation,
        'is_verified': user.is_verified,
        'created_at': user.created_at.isoformat() if user.created_at else None
    } for user in faculty])

@app.route('/api/admin/faculty/<int:user_id>')
@login_required
@admin_required
def get_faculty_member(user_id):
    user = User.query.get_or_404(user_id)
    # Ensure photo_path is correctly formatted for display
    display_photo_path = None
    if user.photo_path:
        if user.role == 'faculty':
            # If the path doesn't already include 'uploads/faculty', add it
            if 'uploads/faculty' not in user.photo_path:
                display_photo_path = os.path.join('uploads', 'faculty', os.path.basename(user.photo_path))
            else:
                display_photo_path = user.photo_path
        else:
            display_photo_path = user.photo_path

    print(f"Faculty photo path: {display_photo_path}")  # Debug log
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'username': user.username,
        'department': user.department,
        'designation': user.designation,
        'is_verified': user.is_verified,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'photo_path': display_photo_path
    })

@app.route('/api/admin/faculty/<int:user_id>/verify', methods=['POST'])
@login_required
@admin_required
def verify_faculty(user_id):
    user = User.query.get_or_404(user_id)
    user.is_verified = True
    db.session.commit()
    return jsonify({'message': 'Faculty member verified successfully'})

@app.route('/api/admin/faculty/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_faculty(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'Faculty member deleted successfully'})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Create default admin account if it doesn't exist
def create_default_admin():
    try:
        # First check if admin exists
        admin = User.query.filter_by(email='mslatha0707@gmail.com').first()
        if not admin:
            print("Creating default admin account...")
            # Create new admin user
            admin = User(
                username='san',
                email='mslatha0707@gmail.com',
                role='admin',
                is_verified=True,
                name='Admin'  # Adding name field
            )
            admin.set_password('san')
            db.session.add(admin)
            db.session.commit()
            print("Default admin account created successfully!")
        else:
            # Update existing admin credentials
            admin.username = 'san'
            admin.set_password('san')
            admin.is_verified = True
            db.session.commit()
            print("Default admin account updated!")
    except Exception as e:
        print(f"Error creating/updating admin account: {str(e)}")
        db.session.rollback()

def fix_photo_paths():
    """Fix photo paths in the database to use forward slashes."""
    try:
        users = User.query.all()
        for user in users:
            if user.photo_path:
                # Convert backslashes to forward slashes
                user.photo_path = user.photo_path.replace('\\', '/')
        db.session.commit()
        print("Photo paths fixed successfully!")
    except Exception as e:
        print(f"Error fixing photo paths: {str(e)}")
        db.session.rollback()

# Initialize database and create admin account
def init_db():
    with app.app_context():
        db.create_all()
        create_default_admin()
        fix_photo_paths()  # Fix photo paths after initialization

@app.route('/api/demo_exam/verify', methods=['POST'])
@login_required
def demo_exam_verify():
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400
    try:
        # 1. Save the reference image
        os.makedirs('reference_faces', exist_ok=True)
        ref_path = f'reference_faces/{current_user.username}.jpg'
        image_data = base64.b64decode(image_b64)
        with open(ref_path, 'wb') as f:
            f.write(image_data)
        
        # 1a. Check for exactly one face using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_count = len(faces)
        if face_count != 1:
            return jsonify({'error': f'Exactly one face must be present in the frame. Detected: {face_count}'}), 400
        
        # 2. Generate embedding for reference image
        try:
            ref_embedding = DeepFace.represent(img_path=ref_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
        except Exception as e:
            return jsonify({'error': 'Face not detected in captured image. Please try again with better lighting and face clearly visible.'}), 400
        # 3. Load profile picture and generate embedding
        profile_path = os.path.join(app.root_path, 'static', current_user.photo_path)
        try:
            profile_embedding = DeepFace.represent(img_path=profile_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
        except Exception as e:
            return jsonify({'error': 'Face not detected in profile image. Please contact support.'}), 400
        # 4. Cosine similarity
        similarity = np.dot(ref_embedding, profile_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(profile_embedding))
        threshold = 0.6
        if similarity > threshold:
            # Optionally, store the embedding for future use
            os.makedirs('embeddings', exist_ok=True)
            with open(f'embeddings/{current_user.username}.pkl', 'wb') as f:
                pickle.dump(ref_embedding, f)
            return jsonify({'success': True, 'similarity': similarity}), 200
        else:
            return jsonify({'success': False, 'similarity': similarity, 'error': 'Face does not match profile'}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo_exam/proctor', methods=['POST'])
@login_required
def demo_exam_proctor():
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        proctoring_system.start_monitoring(current_user.id)
        return jsonify({'success': True, 'message': 'Proctoring started.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo_exam/status', methods=['GET'])
@login_required
def demo_exam_status():
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(proctoring_system.get_live_status()), 200

@app.route('/api/proctoring/analyze', methods=['POST'])
def proctoring_analyze():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze frame
        result = analyze_frame(frame)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New API routes for exam creation and management
@app.route('/api/exam/create', methods=['POST'])
@login_required
def create_exam():
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        data = request.get_json()
        # Parse as local time string (no UTC conversion)
        start_time = datetime.fromisoformat(data['start_time'])
        end_time = datetime.fromisoformat(data['end_time'])
        exam = Exam(
            title=data['title'],
            description=data.get('description', ''),
            start_time=start_time,
            end_time=end_time,
            duration=data['duration'],
            total_marks=data['total_marks'],
            created_by=current_user.id
        )
        db.session.add(exam)
        db.session.commit()
        return jsonify({
            'success': True,
            'exam_id': exam.id,
            'message': 'Exam created successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/upload-file', methods=['POST'])
@login_required
def upload_exam_file(exam_id):
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_type = request.form.get('file_type', 'question_paper')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{exam_id}_{file_type}_{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(EXAM_FILES_FOLDER, unique_filename)
        file.save(file_path)
        
        # Create exam file record
        exam_file = ExamFile(
            exam_id=exam_id,
            file_type=file_type,
            file_path=file_path,
            original_filename=filename
        )
        
        db.session.add(exam_file)
        db.session.commit()
        
        # Determine if a keywords file already exists for this exam
        existing_keywords_file = ExamFile.query.filter_by(exam_id=exam_id, file_type='keywords').first()
        # Extract text and keywords only if:
        # - This upload is a keywords file, OR
        # - This upload is a question paper AND there is no keywords file for this exam
        extracted_keywords = []
        if (file_type == 'keywords') or (file_type == 'question_paper' and not existing_keywords_file):
            file_extension = filename.rsplit('.', 1)[1].lower();
            if file_extension == 'pdf':
                text = extract_text_from_pdf(file_path)
            elif file_extension == 'docx':
                text = extract_text_from_docx(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            extracted_keywords = extract_keywords_from_text(text)
        
        return jsonify({
            'success': True,
            'file_id': exam_file.id,
            'extracted_keywords': extracted_keywords,
            'message': 'File uploaded successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/keywords', methods=['POST'])
@login_required
def save_exam_keywords(exam_id):
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        keywords = data.get('keywords', [])
        
        # Clear existing keywords
        ExamKeyword.query.filter_by(exam_id=exam_id).delete()
        
        # Add new keywords
        for keyword_data in keywords:
            keyword = ExamKeyword(
                exam_id=exam_id,
                keyword=keyword_data['keyword'],
                weight=keyword_data.get('weight', 1.0)
            )
            db.session.add(keyword)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Keywords saved successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/invite-students', methods=['POST'])
@login_required
def invite_students(exam_id):
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        student_emails = data.get('student_emails', [])
        
        invitations = []
        for email in student_emails:
            # Check if invitation already exists
            existing_invitation = StudentInvitation.query.filter_by(
                exam_id=exam_id, 
                student_email=email
            ).first()
            
            if not existing_invitation:
                invitation = StudentInvitation(
                    exam_id=exam_id,
                    student_email=email,
                    invitation_code=uuid.uuid4().hex[:12].upper()
                )
                db.session.add(invitation)
                invitations.append(invitation)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'invitations_created': len(invitations),
            'message': f'Invitations sent to {len(invitations)} students'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/invitations', methods=['GET'])
@login_required
def get_exam_invitations(exam_id):
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    invitations = StudentInvitation.query.filter_by(exam_id=exam_id).all()
    
    return jsonify({
        'invitations': [{
            'id': inv.id,
            'student_email': inv.student_email,
            'invitation_code': inv.invitation_code,
            'is_accepted': inv.is_accepted,
            'accepted_at': inv.accepted_at.astimezone(timezone.utc).isoformat() if inv.accepted_at else None,
            'created_at': inv.created_at.astimezone(timezone.utc).isoformat(),
            'start_time': exam.start_time.astimezone(timezone.utc).isoformat(),
            'end_time': exam.end_time.astimezone(timezone.utc).isoformat(),
        } for inv in invitations]
    })

@app.route('/api/exam/<int:exam_id>/files', methods=['GET'])
@login_required
def get_exam_files(exam_id):
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    files = ExamFile.query.filter_by(exam_id=exam_id).all()
    
    return jsonify({
        'files': [{
            'id': file.id,
            'file_type': file.file_type,
            'original_filename': file.original_filename,
            'uploaded_at': file.uploaded_at.isoformat()
        } for file in files]
    })

@app.route('/api/exam/<int:exam_id>/keywords', methods=['GET'])
@login_required
def get_exam_keywords(exam_id):
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    keywords = ExamKeyword.query.filter_by(exam_id=exam_id).all()
    
    return jsonify({
        'keywords': [{
            'id': kw.id,
            'keyword': kw.keyword,
            'weight': kw.weight
        } for kw in keywords]
    })

@app.route('/api/exam/accept-invitation', methods=['POST'])
@login_required
def accept_exam_invitation():
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        invitation_code = data.get('invitation_code')
        
        if not invitation_code:
            return jsonify({'error': 'Invitation code is required'}), 400
        
        # Find invitation
        invitation = StudentInvitation.query.filter_by(
            invitation_code=invitation_code,
            student_email=current_user.email
        ).first()
        
        if not invitation:
            return jsonify({'error': 'Invalid invitation code or email mismatch'}), 404
        
        if invitation.is_accepted:
            return jsonify({'error': 'Invitation already accepted'}), 400
        
        # Accept invitation
        invitation.is_accepted = True
        invitation.accepted_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Invitation accepted successfully',
            'exam_id': invitation.exam_id
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/available', methods=['GET'])
@login_required
def get_available_exams():
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get exams where student has accepted invitation
        accepted_invitations = StudentInvitation.query.filter_by(
            student_email=current_user.email,
            is_accepted=True
        ).all()
        
        exam_ids = [inv.exam_id for inv in accepted_invitations]
        exams = Exam.query.filter(Exam.id.in_(exam_ids)).all()
        
        return jsonify({
            'exams': [{
                'id': exam.id,
                'title': exam.title,
                'description': exam.description,
                'start_time': exam.start_time.astimezone(timezone.utc).isoformat(),
                'end_time': exam.end_time.astimezone(timezone.utc).isoformat(),
                'duration': exam.duration,
                'total_marks': exam.total_marks
            } for exam in exams]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/pending-invitations', methods=['GET'])
@login_required
def get_pending_invitations():
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get pending invitations for the student
        pending_invitations = StudentInvitation.query.filter_by(
            student_email=current_user.email,
            is_accepted=False
        ).all()
        
        invitations_with_exam = []
        for invitation in pending_invitations:
            exam = Exam.query.get(invitation.exam_id)
            if exam:
                invitations_with_exam.append({
                    'invitation_id': invitation.id,
                    'invitation_code': invitation.invitation_code,
                    'exam_id': exam.id,
                    'exam_title': exam.title,
                    'exam_description': exam.description,
                    'created_at': invitation.created_at.astimezone(timezone.utc).isoformat(),
                    'start_time': exam.start_time.astimezone(timezone.utc).isoformat(),
                    'end_time': exam.end_time.astimezone(timezone.utc).isoformat(),
                })
        
        return jsonify({
            'pending_invitations': invitations_with_exam
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/start', methods=['POST'])
@login_required
def start_exam_session(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        # Check if student has access to this exam
        invitation = StudentInvitation.query.filter_by(
            exam_id=exam_id,
            student_email=current_user.email,
            is_accepted=True
        ).first()
        if not invitation:
            return jsonify({'error': 'No access to this exam'}), 403
        # Check if exam is currently active
        exam = Exam.query.get_or_404(exam_id)
        now = datetime.now()  # Use local time
        if now < exam.start_time:
            return jsonify({'error': 'Exam has not started yet'}), 400
        if now > exam.end_time:
            return jsonify({'error': 'Exam has ended'}), 400
        # Check if student already has an active or ended session
        any_session = ExamSession.query.filter_by(
            user_id=current_user.id,
            exam_id=exam_id
        ).first()
        if any_session:
            return jsonify({'error': 'You have already attempted this exam and cannot restart it.'}), 400
        # Check if identity verification is required for this exam
        if session.get('exam_verified') != str(exam_id):
            return jsonify({'error': 'Identity verification required'}), 403
        # Create new exam session
        exam_session = ExamSession(
            user_id=current_user.id,
            exam_id=exam_id,
            start_time=now
        )
        db.session.add(exam_session)
        db.session.commit()
        return jsonify({
            'success': True,
            'session_id': exam_session.id,
            'message': 'Exam session started successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/question-paper', methods=['GET'])
@login_required
def get_exam_question_paper(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Check if student has access to this exam
        invitation = StudentInvitation.query.filter_by(
            exam_id=exam_id,
            student_email=current_user.email,
            is_accepted=True
        ).first()
        
        if not invitation:
            return jsonify({'error': 'No access to this exam'}), 403
        
        # Get question paper file
        question_file = ExamFile.query.filter_by(
            exam_id=exam_id,
            file_type='question_paper'
        ).first()
        
        if not question_file:
            return jsonify({'error': 'Question paper not found'}), 404
        
        # Read file content based on type
        file_extension = question_file.original_filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'pdf':
            text = extract_text_from_pdf(question_file.file_path)
        elif file_extension == 'docx':
            text = extract_text_from_docx(question_file.file_path)
        else:
            # For txt files, read directly
            with open(question_file.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        return jsonify({
            'success': True,
            'question_paper': text,
            'filename': question_file.original_filename,
            'file_type': file_extension
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/submit-answer', methods=['POST'])
@login_required
def submit_exam_answer(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        answer_text = data.get('answer_text', '')
        
        # Check if student has an active session
        active_session = ExamSession.query.filter_by(
            user_id=current_user.id,
            exam_id=exam_id,
            end_time=None
        ).first()
        
        if not active_session:
            return jsonify({'error': 'No active exam session'}), 400
        
        # Create answer record
        answer = Answer(
            exam_id=exam_id,
            question_id=1,  # For now, we'll use a single question per exam
            student_id=current_user.id,
            answer_text=answer_text
        )
        db.session.add(answer)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Answer submitted successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/end', methods=['POST'])
@login_required
def end_exam_session(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Find active session
        active_session = ExamSession.query.filter_by(
            user_id=current_user.id,
            exam_id=exam_id,
            end_time=None
        ).first()
        
        if not active_session:
            return jsonify({'error': 'No active exam session'}), 400
        
        # End session
        active_session.end_time = datetime.utcnow().replace(tzinfo=timezone.utc)  # Make UTC-aware
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Exam session ended successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/session-status', methods=['GET'])
@login_required
def get_exam_session_status(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Check if student has access to this exam
        invitation = StudentInvitation.query.filter_by(
            exam_id=exam_id,
            student_email=current_user.email,
            is_accepted=True
        ).first()
        
        if not invitation:
            return jsonify({'error': 'No access to this exam'}), 403
        
        # Get active session
        active_session = ExamSession.query.filter_by(
            user_id=current_user.id,
            exam_id=exam_id,
            end_time=None
        ).first()
        
        if active_session:
            return jsonify({
                'has_active_session': True,
                'session_id': active_session.id,
                'start_time': active_session.start_time.astimezone(timezone.utc).isoformat()  # Ensure UTC
            })
        else:
            return jsonify({
                'has_active_session': False
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def ensure_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo:
        return dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=timezone.utc)

@app.route('/exam/<int:exam_id>')
@login_required
def exam_interface(exam_id):
    if current_user.role != 'student':
        return redirect(url_for('dashboard'))
    
    try:
        # Check if student has access to this exam
        invitation = StudentInvitation.query.filter_by(
            exam_id=exam_id,
            student_email=current_user.email,
            is_accepted=True
        ).first()
        
        if not invitation:
            flash('You do not have access to this exam')
            return redirect(url_for('dashboard'))
        
        # Get exam details
        exam = Exam.query.get_or_404(exam_id)
        
        # Check if exam is currently active
        now = datetime.now()  # Use local time
        if now < exam.start_time:
            flash('This exam has not started yet')
            return redirect(url_for('dashboard'))
        
        if now > exam.end_time:
            flash('This exam has ended')
            return redirect(url_for('dashboard'))
        
        # Pass datetimes as is (local time)
        exam_data = exam.__dict__.copy()
        exam_data['start_time'] = exam.start_time
        exam_data['end_time'] = exam.end_time
        return render_template('exam_interface.html', exam=exam_data)
    except Exception as e:
        flash('Error accessing exam')
        return redirect(url_for('dashboard'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except ImportError:
        return "PyPDF2 not installed. Please install it to process PDF files."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except ImportError:
        return "python-docx not installed. Please install it to process DOCX files."
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def extract_keywords_from_text(text):
    """Extract potential keywords from text"""
    # Simple keyword extraction - can be enhanced with NLP
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    for word in words:
        if word not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top 20 most frequent words as potential keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:20]]

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# After all model definitions and after db = SQLAlchemy(app)
admin.add_view(SecureModelView(User, db.session))
admin.add_view(SecureModelView(Exam, db.session))
admin.add_view(SecureModelView(Question, db.session))

@app.route('/api/exam/<int:exam_id>', methods=['DELETE'])
@login_required
def delete_exam(exam_id):
    if current_user.role != 'faculty':
        return jsonify({'error': 'Unauthorized'}), 403
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        # Delete related ExamFile records and files
        exam_files = ExamFile.query.filter_by(exam_id=exam_id).all()
        for file in exam_files:
            try:
                if file.file_path and os.path.exists(file.file_path):
                    os.remove(file.file_path)
            except Exception:
                pass
            db.session.delete(file)
        # Delete related ExamKeyword records
        ExamKeyword.query.filter_by(exam_id=exam_id).delete()
        # Delete related StudentInvitation records
        StudentInvitation.query.filter_by(exam_id=exam_id).delete()
        # Delete related ExamSession records
        ExamSession.query.filter_by(exam_id=exam_id).delete()
        # Delete related Answer records
        Answer.query.filter_by(exam_id=exam_id).delete()
        # Delete the exam itself
        db.session.delete(exam)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Exam deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/verify_identity', methods=['POST'])
@login_required
def verify_exam_identity(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400
    try:
        # Save reference image
        os.makedirs('reference_faces', exist_ok=True)
        ref_path = f'reference_faces/{current_user.username}_exam.jpg'
        image_data = base64.b64decode(image_b64)
        with open(ref_path, 'wb') as f:
            f.write(image_data)
        # Generate embedding for reference image
        ref_embedding = DeepFace.represent(img_path=ref_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
        # Generate embedding for profile picture
        profile_path = os.path.join(app.root_path, 'static', current_user.photo_path)
        profile_embedding = DeepFace.represent(img_path=profile_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
        # Cosine similarity
        similarity = np.dot(ref_embedding, profile_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(profile_embedding))
        threshold = 0.6
        if similarity > threshold:
            # Mark verification as passed (in session)
            session['exam_verified'] = f'{exam_id}'
            return jsonify({'success': True, 'similarity': similarity}), 200
        else:
            return jsonify({'success': False, 'similarity': similarity, 'error': 'Face does not match profile'}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/upload-answer', methods=['POST'])
@login_required
def upload_answer_script(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    exam = Exam.query.get_or_404(exam_id)
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        # Save file
        answers_folder = os.path.join(app.root_path, 'static', 'uploads', 'answers')
        os.makedirs(answers_folder, exist_ok=True)
        filename = secure_filename(file.filename)
        unique_filename = f"{exam_id}_{current_user.id}_{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(answers_folder, unique_filename)
        file.save(file_path)
        # Save record in AnswerFile
        answer_file = AnswerFile(
            exam_id=exam_id,
            student_id=current_user.id,
            file_path=os.path.join('uploads', 'answers', unique_filename),
            original_filename=filename
        )
        db.session.add(answer_file)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Answer script uploaded successfully', 'file_path': answer_file.file_path})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Endpoint to list uploaded answer files for the current student and exam
@app.route('/api/exam/<int:exam_id>/my-answers', methods=['GET'])
@login_required
def list_my_uploaded_answers(exam_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    exam = Exam.query.get_or_404(exam_id)
    now = datetime.now()
    # Only show files if the exam is still active
    if now > exam.end_time:
        return jsonify({'files': []})
    files = AnswerFile.query.filter_by(exam_id=exam_id, student_id=current_user.id).order_by(AnswerFile.uploaded_at.desc()).all()
    return jsonify({'files': [
        {
            'id': f.id,
            'original_filename': f.original_filename,
            'file_path': f.file_path,
            'uploaded_at': f.uploaded_at.isoformat()
        } for f in files
    ]})

@app.route('/api/exam/<int:exam_id>/delete-answer/<int:file_id>', methods=['DELETE'])
@login_required
def delete_uploaded_answer(exam_id, file_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Unauthorized'}), 403
    answer_file = AnswerFile.query.filter_by(id=file_id, exam_id=exam_id, student_id=current_user.id).first()
    if not answer_file:
        return jsonify({'error': 'File not found'}), 404
    try:
        # Delete file from disk
        file_path = os.path.join(app.root_path, 'static', answer_file.file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        db.session.delete(answer_file)
        db.session.commit()
        return jsonify({'success': True, 'message': 'File deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()  # Initialize database and create admin account
    app.run(debug=True) 
    