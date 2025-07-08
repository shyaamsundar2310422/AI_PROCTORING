# AI_PROCTORING
This is new repository to save all the files related to the IFSP project.





# Exam Creation and Student Invitation Functionality

This document describes the new functionality added to the facial verification proctoring system that allows faculties to create exams and invite students.

## New Features

### 1. Exam Creation with File Upload
- **Question Paper Upload**: Faculties can upload question papers in PDF, DOCX, or TXT formats
- **Keywords File Upload**: Optional keywords file can be uploaded separately
- **Automatic Keyword Extraction**: The system automatically extracts potential keywords from uploaded question papers
- **Exam Details**: Set exam title, description, start/end times, duration, and total marks

### 2. Student Invitation System
- **Bulk Invitation**: Faculties can invite multiple students by entering their email addresses
- **Unique Invitation Codes**: Each invitation gets a unique code for security
- **Invitation Management**: View all sent invitations and their acceptance status
- **Student Acceptance**: Students can accept invitations using the provided codes

### 3. Enhanced Dashboard
- **Faculty Dashboard**: Manage exams, upload files, and invite students
- **Student Dashboard**: View pending invitations and available exams
- **Real-time Status**: See exam status (Upcoming, Active, Completed)

## Database Models Added

### ExamFile
- Stores uploaded question papers and keywords files
- Supports multiple file types (PDF, DOCX, TXT)
- Links files to specific exams

### ExamKeyword
- Stores keywords extracted from question papers
- Includes weight/importance for each keyword
- Used for potential plagiarism detection

### StudentInvitation
- Manages student invitations to exams
- Tracks invitation acceptance status
- Generates unique invitation codes

## API Endpoints

### Faculty Endpoints
- `POST /api/exam/create` - Create a new exam
- `POST /api/exam/{exam_id}/upload-file` - Upload question paper or keywords file
- `POST /api/exam/{exam_id}/keywords` - Save exam keywords
- `POST /api/exam/{exam_id}/invite-students` - Invite students to exam
- `GET /api/exam/{exam_id}/files` - Get exam files
- `GET /api/exam/{exam_id}/keywords` - Get exam keywords
- `GET /api/exam/{exam_id}/invitations` - Get exam invitations

### Student Endpoints
- `POST /api/exam/accept-invitation` - Accept exam invitation
- `GET /api/exam/available` - Get available exams
- `GET /api/exam/pending-invitations` - Get pending invitations

## File Processing

### Supported Formats
- **PDF**: Uses PyPDF2 for text extraction
- **DOCX**: Uses python-docx for text extraction
- **TXT**: Direct text reading

### Keyword Extraction
- Automatically extracts keywords from uploaded question papers
- Filters out common stop words
- Returns top 20 most frequent words as potential keywords
- Can be enhanced with NLP libraries for better extraction

## Usage Workflow

### For Faculties
1. **Create Exam**: Fill in exam details and optionally upload question paper
2. **Upload Files**: Upload question paper and/or keywords file
3. **Review Keywords**: System extracts keywords automatically, can be modified
4. **Invite Students**: Enter student email addresses to send invitations
5. **Monitor**: Track invitation acceptance and exam status

### For Students
1. **View Invitations**: Check dashboard for pending exam invitations
2. **Accept Invitation**: Use invitation code to accept exam invitation
3. **Access Exams**: View available exams after accepting invitations
4. **Take Exam**: Start exam when it becomes active (future feature)

## Installation

### New Dependencies
```bash
pip install PyPDF2==3.0.1 python-docx==0.8.11
```

### Database Migration
The new models will be automatically created when the application starts. Run:
```bash
python app.py
```

## Testing

Run the test script to verify functionality:
```bash
python test_exam_functionality.py
```

## Security Features

- **File Type Validation**: Only allowed file types can be uploaded
- **Unique Invitation Codes**: Each invitation has a unique, secure code
- **Email Verification**: Invitations are tied to specific email addresses
- **Access Control**: Only exam creators can manage their exams
- **File Size Limits**: 16MB maximum file size for uploads

## Future Enhancements

1. **Email Notifications**: Send email invitations to students
2. **Advanced Keyword Extraction**: Use NLP for better keyword detection
3. **Plagiarism Detection**: Use keywords for answer analysis
4. **Exam Taking Interface**: Complete exam interface for students
5. **Answer Evaluation**: Automatic and manual answer evaluation
6. **Analytics Dashboard**: Detailed exam analytics and reports

## File Structure

```
static/uploads/exam_files/  # Uploaded exam files
├── {exam_id}_question_paper_{uuid}_{filename}
└── {exam_id}_keywords_{uuid}_{filename}
```

## Error Handling

- File upload errors are caught and reported
- Invalid invitation codes are rejected
- Database transaction rollback on errors
- User-friendly error messages in the UI

## Browser Compatibility

- Modern browsers with ES6+ support
- File upload API support
- Bootstrap 5 for UI components 