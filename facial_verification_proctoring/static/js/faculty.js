document.addEventListener('DOMContentLoaded', function() {
    // Handle faculty login form submission
    const facultyLoginForm = document.getElementById('facultyLoginForm');
    if (facultyLoginForm) {
        facultyLoginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch('/api/faculty/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    if (data.status === 'pending') {
                        alert(data.error);
                        window.location.href = data.redirect;
                    } else if (data.redirect) {
                        window.location.href = data.redirect;
                    } else {
                        window.location.href = '/faculty-dashboard';
                    }
                } else {
                    alert(data.error || 'Login failed. Please try again.');
                }
            } catch (error) {
                console.error('Login error:', error);
                alert('An error occurred during login. Please try again.');
            }
        });
    }
    
    // Handle faculty registration form submission
    const facultyRegisterForm = document.getElementById('facultyRegisterForm');
    if (facultyRegisterForm) {
        facultyRegisterForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const username = document.getElementById('username').value;
            const department = document.getElementById('department').value;
            const designation = document.getElementById('designation').value;
            const photoFile = document.getElementById('photo').files[0];
            
            // Convert photo to base64
            const photoBase64 = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result.split(',')[1]);
                reader.onerror = error => reject(error);
                reader.readAsDataURL(photoFile);
            });
            
            try {
                const response = await fetch('/api/faculty/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name,
                        email,
                        password,
                        username,
                        department,
                        designation,
                        photo: photoBase64
                    }),
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert(data.message);
                    window.location.href = '/faculty-login';
                } else {
                    alert(data.error || 'Registration failed');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during registration');
            }
        });
    }
}); 