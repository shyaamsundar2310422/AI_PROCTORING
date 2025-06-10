from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import os
from datetime import datetime
from proctoring_system import ProctoringSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# SQLite Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///exam_portal.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/students'
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
    role = db.Column(db.String(20), nullable=False)  # 'student', 'faculty', 'admin'
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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
    return render_template('dashboard.html', exams=exams)

@app.route('/verification-pending')
def verification_pending():
    return render_template('verification_pending.html')

# API Routes
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    email = data.get('email')
    name = data.get('name')
    password = data.get('password')
    aadhar_number = data.get('aadhar_number')
    photo = data.get('photo')  # Base64 encoded photo

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400

    if User.query.filter_by(aadhar_number=aadhar_number).first():
        return jsonify({'error': 'Aadhar number already registered'}), 400

    # Save photo
    if photo:
        filename = f"{aadhar_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Convert base64 to image and save
        # Implementation needed

    user = User(
        email=email,
        name=name,
        aadhar_number=aadhar_number,
        photo_path=filename
    )
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'Registration successful! Please wait for verification.'}), 201

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 