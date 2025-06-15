from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import os
from datetime import datetime
from proctoring_system import ProctoringSystem
from dotenv import load_dotenv
from functools import wraps
from werkzeug.utils import safe_join

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

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize proctoring system
proctoring_system = ProctoringSystem()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        photo_path_db = None # Initialize photo_path_db
        if photo:
            try:
                import base64
                # Ensure the faculty uploads directory exists within static/uploads
                faculty_upload_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], 'faculty')
                os.makedirs(faculty_upload_dir, exist_ok=True)
                filename = f"faculty_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                photo_full_path = os.path.join(faculty_upload_dir, filename)
                
                # Convert base64 to image and save
                image_data = base64.b64decode(photo)
                with open(photo_full_path, 'wb') as f:
                    f.write(image_data)
                # Store path relative to the static folder (for url_for('static'))
                photo_path_db = os.path.join('uploads', 'faculty', filename) 
            except Exception as e:
                return jsonify({'error': f'Error saving photo: {str(e)}'}), 500

        user = User(
            email=email,
            name=name,
            username=username,
            department=department,
            designation=designation,
            photo_path=photo_path_db, # Use the correctly constructed relative path
            role='faculty'
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'message': 'Registration successful! Please wait for admin verification.'}), 201
    except Exception as e:
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
        if photo:
            try:
                import base64
                student_upload_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], 'students')
                os.makedirs(student_upload_dir, exist_ok=True)
                filename = f"{aadhar_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                photo_full_path = os.path.join(student_upload_dir, filename)
                
                # Convert base64 to image and save
                image_data = base64.b64decode(photo)
                with open(photo_full_path, 'wb') as f:
                    f.write(image_data)
                print(f"Photo saved: {filename}")  # Debug log
                photo_path_db = os.path.join('uploads', 'students', filename)
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
    user = User.query.filter_by(email=email).first()
    
    if user and user.check_password(password):
        if user.is_verified:
            login_user(user)
            return jsonify({
                'message': 'Login successful',
                'redirect': '/dashboard'
            }), 200
        else:
            return jsonify({
                'error': 'Your account is pending verification',
                'status': 'pending',
                'redirect': '/verification-pending'
            }), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401

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

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email, role='admin').first()
    
    if user and user.check_password(password):
        login_user(user)
        return jsonify({
            'message': 'Login successful',
            'redirect': '/admin-dashboard'
        }), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/api/admin/register', methods=['POST'])
def api_admin_register():
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')
        password = data.get('password')
        username = data.get('username')

        if not all([email, name, password, username]):
            return jsonify({'error': 'All fields are required'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already taken'}), 400

        user = User(
            email=email,
            name=name,
            username=username,
            role='admin',
            is_verified=True  # Admins are automatically verified
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'message': 'Admin registration successful!'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

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
        if user.role == 'student' and 'uploads/students/' not in user.photo_path:
            display_photo_path = os.path.join('uploads', 'students', os.path.basename(user.photo_path))
        else:
            display_photo_path = user.photo_path # Already has the correct prefix or is faculty

    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'username': user.username,
        'is_verified': user.is_verified,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'photo_path': display_photo_path # Use the correctly formatted path
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
        if user.role == 'faculty' and 'uploads/faculty/' not in user.photo_path:
            display_photo_path = os.path.join('uploads', 'faculty', os.path.basename(user.photo_path))
        else:
            display_photo_path = user.photo_path # Already has the correct prefix or is student

    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'username': user.username,
        'department': user.department,
        'designation': user.designation,
        'is_verified': user.is_verified,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'photo_path': display_photo_path # Use the correctly formatted path
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
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database path:", os.path.abspath("exam_portal.db"))

    app.run(debug=True) 
    